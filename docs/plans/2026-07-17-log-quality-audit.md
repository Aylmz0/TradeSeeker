# Log Çıktı Kalitesi İyileştirme Planı

**Tarih:** 2026-07-17
**Kapsam:** Mevcut loguru altyapısı (4 sink) iyi — içerik dağıtımı ve temizlik sorunlu.
**Felsefe:** Trading bot'ta tüm veri değerli → **hiçbir log kaybolmaz**, sadece doğru yere taşınır.

---

## Mevcut Durum
- 4 sink aktif: console (renkli), .log (günlük), structured.jsonl, crash_reports.jsonl
- 1 cycle ~600+ satır: ~580'i LLM reasoning dump'ı, ~20'si operasyonel bilgi
- LiteLLM stdlib logging'i bypass ediyor → format'sız stderr satırları
- structured.jsonl `"text": "I5O\n"` bozuk (serialize=True + {time:ISO} uyumsuz)
- `_ensure_exit_plan` WARNING her 30sn'de tekrar (~14×/cycle)
- Duplicate "LLM response received" mesajı

---

## Aşama A: Operasyonel Log Temizliği (P0)

**Hedef:** Ana `.log` dosyası sadece operasyonel bilgi (sinyaller, kararlar, PnL, hatalar) içersin. Reasoning/prompt ayrı yere gider.

### Değişiklikler

**`src/main.py:408`** — Prompt basma
```python
# ÖNCE
logger.info("Prompt: \n{}", prompt)
# SONRA
logger.debug("Prompt size: {} chars", len(prompt))
```

**`src/main.py:479-483`** — REASONING ve DECISIONS ayrımı
```python
# ÖNCE: ikisi de INFO
logger.info("REASONING:\n{}", thoughts)
logger.info("DECISIONS:\n{}", json.dumps(decisions, ...))

# SONRA: reasoning ayrı yere, decisions INFO'da kalır
logger.debug("REASONING: {} chars, {} decisions", len(thoughts), len(decisions))
logger.bind(kind="ai_reasoning").debug("REASONING FULL:\n{}", thoughts)
logger.info("DECISIONS:\n{}", json.dumps(decisions, indent=2) if decisions else "{}")
```

**`src/core/log_config.py`** — Yeni sink: `ai_reasoning.log`
```python
logger.add(
    os.path.join(log_dir, "ai_reasoning.log"),
    level="DEBUG",
    rotation="1 day",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{line} | {message}",
    filter=lambda record: record["extra"].get("kind") == "ai_reasoning",
    compression="gz",
)
```

### Doğrulama
- `.log` dosyasında "REASONING:" dump'ı yok, sadece compact summary
- `ai_reasoning.log` dosyasında tam reasoning mevcut

---

## Aşama B: structured.jsonl Onarımı (P0)

**Hedef:** JSON log `jq` ile sorgulanabilir olsun.

### Değişiklikler

**`src/core/log_config.py:61-68`** — serialize=True sorunu
```python
# ÖNCE
logger.add(
    os.path.join(log_dir, "structured.jsonl"),
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
    format="{time:ISO}",
    serialize=True,
)

# SONRA
logger.add(
    os.path.join(log_dir, "structured.jsonl"),
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
    format="{time:YYYY-MM-DD}T{time:HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
    compression="gz",
)
```
`serialize=True` kaldırıldı → `format` direkt uygulanır, `text` alanı temiz çıkar.

### Doğrulama
- `tail -1 data/logs/structured.jsonl` → `| INFO | src.core...` format'ında okunabilir satır

---

## Aşama C: LiteLLM & xgboost Intercept (P1)

**Hedef:** Ham stderr satırları loguru'ya çekilsin veya bastırılsın.

### Değişiklikler

**`src/core/log_config.py`** — stdlib logging köprüsü (yeni class)
```python
import logging

class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        level_map = {
            logging.DEBUG: "DEBUG", logging.INFO: "INFO",
            logging.WARNING: "WARNING", logging.ERROR: "ERROR",
        }
        level = level_map.get(record.levelno, "INFO")
        logger.opt(depth=6).log(level, record.getMessage())
```

**`src/core/log_config.py`** — `setup_logging` içine ekle
```python
# LiteLLM ve xgboost stdlib logging intercept
for name in ("litellm", "httpx", "httpcore", "LiteLLM Router"):
    logging.getLogger(name).handlers = [InterceptHandler()]
    logging.getLogger(name).setLevel(logging.WARNING)
```

**`src/ai/deepseek_api.py`** — litellm verbose kapat
```python
litellm.suppress_debug_info = True
litellm.verbose = False  # EKLE
```

**`src/services/ml_service.py`** — xgboost warning filtre
```python
import warnings
warnings.filterwarnings("ignore", message=".*Unknown file format.*")
```

### Doğrulama
- Konsolda `[92m...LiteLLM Router...` satırı YOK
- `grep "Unknown file format" data/logs/*.log | wc -l` → 0

---

## Aşama D: WARNING Spam Dedup (P1)

**Hedef:** `_ensure_exit_plan` her cycle'da sadece 1 kez loglansın.

### Değişiklikler

**`src/core/account_service.py`** — `_ensure_exit_plan` dedup
```python
# ÖNCE: her çağrıda basıyor
logger.warning("Missing profit_target for {} - using default exit plan offsets.", coin)

# SONRA: class-level dedup set
class AccountService:
    _warned_missing_exit_plan: set[str] = set()

    def _ensure_exit_plan(self, ...):
        if "profit_target" not in exit_plan:
            if coin not in self._warned_missing_exit_plan:
                logger.warning("Missing profit_target for {} - using default exit plan offsets.", coin)
                self._warned_missing_exit_plan.add(coin)
            # fallback devam eder...
```

### Doğrulama
- `grep "Missing profit_target" data/logs/*.log | wc -l` → 1-2 (cycle başına, spam yok)

---

## Aşama E: Duplicate & Seviye Düzeltmeleri (P2)

### Değişiklikler

**`src/ai/deepseek_api.py:475/488`** — duplicate sil
İkinci `logger.info("LLM response received.")` satırını kaldır.

**`src/core/portfolio_manager.py:4441`** — `log_func` seviye ayrımı
```python
# ÖNCE
def log_func(category, message, details=None):
    logger.info("{}", message)

# SONRA
def log_func(category, message, details=None):
    match category:
        case "OK":
            logger.success("{}", message)
        case "WARN" | "WATCH":
            logger.warning("{}", message)
        case "ERR":
            logger.error("{}", message)
        case _:
            logger.info("{}", message)
```

### Doğrulama
- Konsolda `[OK]` satırları yeşil (success), `[WATCH]` sarı (warning)

---

## Aşama F: Context Binding (P2)

**Hedef:** `structured.jsonl`'da cycle/phase bilgisi olsun (`jq` ile sorgulanabilmeli).

### Değişiklikler

**`src/main.py`** — cycle başında bind
```python
# run_trading_cycle başında
log = logger.bind(cycle=cycle_number, phase="trading_cycle")
```

**`src/core/log_config.py`** — file format'a `{extra}` ekle
```python
format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message} | cycle={extra[cycle]:->4}",
```

### Doğrulama
- `grep "cycle=" data/logs/2026-07-17.log | head -3` → `cycle=  13` format'ında

---

## Aşama G: ml_training.log Entegrasyonu (P3, Düşük Öncelik)

**Hedef:** Eski stdlib `logging` dosyalarını loguru'ya taşı.

### Değişiklikler
- `admin_server_flask.py:33` → `logging.basicConfig()` kaldır, `from loguru import logger` kullan
- `utils.py:19` → `logging.basicConfig()` kaldır (zaten loguru var)
- `ml_training.log` / `performance_refresh.log` artık oluşmayacak

---

## Doğrulama Checksum

| # | Kontrol | Beklenen |
|---|---------|----------|
| 1 | `.log` dosyasında "REASONING FULL" dump'ı | YOK |
| 2 | `ai_reasoning.log` dolu | Evet |
| 3 | `tail -1 structured.jsonl` | `| INFO | src.core...` format |
| 4 | Konsolda `[92mLiteLLM...` satırı | YOK |
| 5 | `grep "Missing profit_target" data/logs/*.log \| wc -l` | ≤ 2 |
| 6 | `grep "cycle=" data/logs/*.log \| head -3` | cycle bilgisi görünür |
| 7 | `ty check src/` | zero errors |
| 8 | `ruff check src/` | zero errors |
| 9 | 1 cycle sorunsuz | Tamam |

---

## Öncelik Sıralaması

| Aşama | Öncelik | Sorun | Efor |
|:------|:--------|:------|:-----|
| A | P0 | Reasoning operasyonel logu kirletiyor | 15 dk |
| B | P0 | structured.jsonl bozuk | 5 dk |
| C | P1 | LiteLLM ham stderr | 15 dk |
| D | P1 | Warning spam | 10 dk |
| E | P2 | Duplicate + seviye | 10 dk |
| F | P2 | Context binding eksik | 10 dk |
| G | P3 | Eski logging alt sistemi | 10 dk |
