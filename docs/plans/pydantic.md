# Pydantic Entegrasyonu — Implementation Plan

## 1. 🎯 Objective
Config, Position ve AI Decision veri yapılarını Pydantic modelleriyle tip-güvenli hale getirmek. Runtime crash'lerini (TypeError, KeyError, "N/A" propagation) compile-time'a taşımak.

## 2. 🏗️ Tech Strategy
- **Pattern:** Gradual adoption — Config → Position → AI → TradeHistory (her aşama bağımsız deploy edilebilir)
- **State:** Mevcut dict yapısını koru, modelleri "wrapper" olarak ekle. Dict↔Model dönüşümü `.model_validate(dict)` / `.model_dump()` ile
- **Constraints:** `pydantic-settings` config için, ana `pydantic` geri kalan için. Mevcut `dict[str, Any]` usage'ları zorla değiştirme — sadece creation point'leri model kullanacak

## 3. 📂 File Changes

| Action | File Path | Brief Purpose |
|:-------|:----------|:--------------|
| [MOD]  | `requirements.txt` | `pydantic>=2.0`, `pydantic-settings>=2.0` ekle |
| [NEW]  | `src/schemas/config.py` | `Settings(BaseSettings)` — .env loading + validation |
| [NEW]  | `src/schemas/position.py` | `Position`, `ExitPlan`, `TrailingMeta` modelleri |
| [NEW]  | `src/schemas/trade.py` | `TradeHistoryEntry`, `CycleHistoryEntry` modelleri |
| [NEW]  | `src/schemas/ai.py` | `AIDecision`, `MLConsensus`, `ExecutionReport` modelleri |
| [NEW]  | `src/schemas/__init__.py` | Tüm modelleri export et |
| [MOD]  | `config/config.py` | `Settings` instance oluştur, `Config` attribute'larını ondan oku |
| [MOD]  | `src/core/portfolio_manager.py` | Position oluşturma noktalarında `Position.model_validate()` kullan |
| [MOD]  | `src/core/ai_service.py` | `_clean_ai_decisions` içinde `AIDecision.model_validate()` |
| [MOD]  | `src/core/account_service.py` | History entry oluşturma noktalarında `TradeHistoryEntry` |
| [MOD]  | `src/main.py` | Config import'unu güncelle |

## 4. 👣 Execution Sequence

### Aşama 1: Schema Tanımları (hiçbir mevcut kod değişmez)
1. `src/schemas/__init__.py` oluştur
2. `src/schemas/config.py` — `Settings(BaseSettings)` tanımla
   - Tüm `.env` field'larını ekle (tip, default, validation)
   - `model_config = SettingsConfigDict(env_file=".env", extra="ignore")`
3. `src/schemas/position.py` — `ExitPlan`, `TrailingMeta`, `Position` tanımla
   - `ExitPlan`: stop_loss/profit_target `float | None = None`
   - `Position`: direction `Literal["long", "short"]`, margin_usd `float = 0.0`
4. `src/schemas/trade.py` — `TradeHistoryEntry`, `CycleHistoryEntry`
5. `src/schemas/ai.py` — `AIDecision`, `MLConsensus`, `ExecutionReport`

### Aşama 2: Config Dönüşümü
1. `requirements.txt`'e pydantic ekle
2. `config/config.py`'de:
   - `from src.schemas.config import Settings`
   - `Settings()` instance oluştur (module level)
   - Mevcut `Config` class'ının attribute'larını `settings.ATTRIBUTE`'dan oku
   - Geriye uyumluluk: `Config` class'ını koru, sadece içini değiştir

### Aşama 3: Position Dönüşümü
1. `portfolio_manager.py` — `_execute_order_payload` içinde position dict yerine `Position.model_validate(dict)` çağrısı ekle
2. `account_service.py` — `_merge_live_positions` içinde position'ları validate et
3. **Kritik:** Position dict hâlâ `dict` olarak saklanır (JSON serialization için), ama creation'da validate edilir

### Aşama 4: AI Decision Dönüşümü
1. `ai_service.py` — `_clean_ai_decisions` içinde `AIDecision.model_validate()` ekle
2. Validation hatalarını yakala → `logger.warning` + hold'a çevir
3. `prompt_json_schemas.py`'deki `pass` statement'ını kaldır, Pydantic validator ile değiştir

### Aşama 5: Trade History Dönüşümü
1. History entry oluşturma noktalarında `TradeHistoryEntry` kullan
2. `"N/A"` fallback'lerini kaldır — default float/int değerleri kullan

### Aşama 6: Doğrulama
1. `ruff check src/` — lint temiz
2. Hatalı `.env` testi — `MAX_LEVERAGE=abc` → graceful error
3. `python3 src/main.py` — runtime test
4. Mevcut `portfolio_state.json`'u yükle — model validation geçmeli

## 5. Blast Radius

| Değişiklik | Etkilenen Modüller | Risk |
|------------|-------------------|------|
| Config → Settings | Tüm modüller (Config her yerde import ediliyor) | Yüksek — dikkatli geçiş |
| Position modeli | portfolio_manager, account_service, main | Orta — dict yapısı korunuyor |
| AIDecision modeli | ai_service, deepseek_api | Düşük — zaten cleaning var |
| TradeHistoryEntry | portfolio_manager, main | Düşük — sadece oluşturma |

## 6. ✅ Verification Standards

- [ ] `ruff check src/` — zero errors
- [ ] `python3 -c "from src.schemas import Settings, Position, AIDecision"` — import OK
- [ ] Hatalı `.env` → `ValidationError` (crash değil, graceful)
- [ ] `Position.model_validate(portfolio_state.json["positions"]["SOL"])` → geçer
- [ ] `python3 src/main.py` — 1 cycle sorunsuz çalışır
- [ ] `data/logs/structured.jsonl` — loglar doğru format'ta

## 7. Öncelik Sırası

| Öncelik | Aşama | Efor | Impact |
|---------|-------|------|--------|
| P0 | Aşama 1 (Schema tanımları) | 30 dk | Foundation |
| P0 | Aşama 2 (Config) | 20 dk | .env hatalarını yakalar |
| P1 | Aşama 3 (Position) | 30 dk | "N/A" bug'ını çözer |
| P2 | Aşama 4 (AI Decision) | 20 dk | LLM garbini yakalar |
| P3 | Aşama 5 (TradeHistory) | 15 dk | Type safety |
| P1 | Aşama 6 (Doğrulama) | 15 dk | Tüm aşamaları test eder |
