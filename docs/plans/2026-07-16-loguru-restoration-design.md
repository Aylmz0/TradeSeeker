# loguru Full Loglama Restorasyonu — Tasarım Document

**Tarih:** 2026-07-16
**Yazar:** Grandmaster (Maestro)
**Durum:** Onaylandı → Implementasyonda

## Problem

Mevcut loglama kaotik: 468 `print()` satırı, 16 dosya, tutarsız format, rotation yok, analiz edilebilirlik yok. Geçmişe dönüp "bu saatte ne oldu" sorusunun cevabı yok.

## Hedef

- Tüm `print()`'leri `logger`'a dönüştür
- Tutarlı, yapılandırılmış loglama
- Otomatik rotation (günlük dosya, 30 gün saklama)
- JSON output (`jq` ile sorgulanabilir)
- Crash reports (ERROR ve üzeri)
- Sıfır bilgi kaybı — mevcut tüm çıktılar korunur

## Yaklaşım

**loguru** (MIT license, ~100KB) — drop-in replacement, otomatik rotation, structured JSON.

## Aşamalar

### Aşama 0: Tasarım Document ✅
Bu dosya.

### Aşama 1: Bağımlılık Kurulumu
**Dosyalar:** `requirements.txt`
**Efor:** 5 dakika

- `loguru>=0.7.0` ekle
- `pip install loguru` ile kur
- Hiçbir mevcut kod değişmez

### Aşama 2: Merkezi Konfigürasyon
**Dosyalar:** `src/core/log_config.py` (YENİ)
**Efor:** 15 dakika

- 4 sink: console (renkli), dosya (günlük rotation), JSON (10MB rotation), crash (JSONL)
- `setup_logging(log_level)` fonksiyonu
- `data/logs/` dizinini otomatik oluştur

### Aşama 3: Konfigürasyon + .gitignore
**Dosyalar:** `config/config.py`, `.gitignore`
**Efor:** 10 dakika

- `LOG_DIR` config'e ekle
- `data/logs/` .gitignore'a ekle

### Aşama 4: main.py Restorasyonu (107 print)
**Dosyalar:** `src/main.py`
**Efor:** 45 dakika

- `setup_logging()` çağrısını `main()`'e ekle
- Tüm `print()`'leri `logger.info/success/warning/error/debug`'a dönüştür
- `logging.basicConfig()` kaldır
- Context binding: `logger.bind(cycle=N, phase="...")`
- Crash report: `_log_crash()` fonksiyonu

### Aşama 5: portfolio_manager.py (89 print)
**Dosyalar:** `src/core/portfolio_manager.py`
**Efor:** 40 dakika

### Aşama 6: account_service.py (71 print)
**Dosyalar:** `src/core/account_service.py`
**Efor:** 35 dakika

### Aşama 7: market_data.py (37 print)
**Dosyalar:** `src/core/market_data.py`
**Efor:** 20 dakika

### Aşama 8: deepseek_api.py (35 print)
**Dosyalar:** `src/ai/deepseek_api.py`
**Efor:** 20 dakika

### Aşama 9: performance_monitor.py (34 print)
**Dosyalar:** `src/core/performance_monitor.py`
**Efor:** 20 dakika

### Aşama 10: cache_manager.py (22 print)
**Dosyalar:** `src/core/cache_manager.py`
**Efor:** 15 dakika

### Aşama 11: enhanced_context_provider.py (21 print)
**Dosyalar:** `src/ai/enhanced_context_provider.py`
**Efor:** 15 dakika

### Aşama 12: data_engine.py (12 print)
**Dosyalar:** `src/core/data_engine.py`
**Efor:** 10 dakika

- `logging.basicConfig()` kaldır

### Aşama 13: strategy_analyzer + ai_service + indicators (30 print)
**Dosyalar:** `src/core/strategy_analyzer.py`, `src/core/ai_service.py`, `src/core/indicators.py`
**Efor:** 20 dakika

### Aşama 14: ml_service + binance (8 print)
**Dosyalar:** `src/services/ml_service.py`, `src/services/binance.py`
**Efor:** 10 dakika

### Aşama 15: Temizlik
**Dosyalar:** Tümü
**Efor:** 15 dakika

- Tüm `logging.basicConfig()` kaldır (utils.py, data_engine.py, admin_server_flask.py)
- Logger import'larını kontrol et
- Kullanılmayan `import logging`'leri temizle

### Aşama 16: Doğrulama
**Efor:** 15 dakika

- `ruff check src/` — lint temiz
- `python3 src/main.py` — runtime test
- `ls data/logs/` — dosyalar oluşmuş mu
- `tail -1 data/logs/structured.jsonl | jq .` — JSON看得见
- Kasıtlı hata → crash_reports.jsonl'a yazmalı

## Log Seviye Haritası

| Mevcut Pattern | → Log Seviyesi | Örnek |
|----------------|----------------|-------|
| `[INFO]` | `logger.info` | `[INFO] Fetching market data...` |
| `[OK]` | `logger.success` | `[OK] Prices: ETH: $1918...` |
| `[WARN]` | `logger.warning` | `[WARN] No API key found...` |
| `[ERR]` | `logger.error` | `[ERR] CRITICAL CYCLE ERROR...` |
| `[TIME]` | `logger.debug` | `[TIME] Timers: market 694ms...` |
| `[WAIT]` | `logger.debug` | `[WAIT] Waiting for response...` |
| Debug detayları | `logger.debug` | Cache hit/miss, indikatör hesaplama |

## Context Binding Stratejisi

```python
# Döngü seviyesi
log = logger.bind(cycle=cycle_number, phase="trading_cycle")

# Coin seviyesi
log = logger.bind(coin="ETH", interval="15m")

# API çağrısı
log = logger.bind(model="openrouter/gemini", payload_size=len(prompt))

# ML tahmini
log = logger.bind(coin="SOL", prediction="BUY", confidence=0.42)
```

## Dosya Yapısı

```
data/logs/
├── 2026-07-16.log          # İnsanoğlu için (günlük rotation, 30 gün)
├── 2026-07-15.log.gz        # Sıkıştırılmış eski loglar
├── structured.jsonl          # Makine için (10MB rotation, 7 gün)
└── crash_reports.jsonl       # Sadece hatalar (90 gün)
```

## Kullanım

```bash
# Son hataları gör
tail -5 data/logs/crash_reports.jsonl | jq '.exception_type, .message'

# Bu haftaki logları filtrele
cat data/logs/structured.jsonl | jq 'select(.record.text | contains("ETH"))'

# En çok tekrar eden hata
cat data/logs/crash_reports.jsonl | jq -s 'group_by(.exception_type) | map({type: .[0].exception_type, count: length}) | sort_by(-.count)'

# Döngü 42'yi ara
cat data/logs/structured.jsonl | jq 'select(.record.extra.cycle == 42)'
```
