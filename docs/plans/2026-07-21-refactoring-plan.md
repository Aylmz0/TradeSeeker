# Refactoring Plan — TradeSeeker (Güncellenmiş)

**Tarih:** 2026-07-21 (v2)
**Sıra:** Docstring ✅ → Test ✅ → **Refactoring** (bu plan)
**Kapsam:** Tüm src/ kodu

---

## 1. Doğrulanmış Mevcut Durum

| Sorun | Sayı | Doğrulandı |
|-------|------|------------|
| Dead import | 3 (confirmed) | ✅ |
| Redundant import | 19 (local Config re-import) | ✅ |
| Bare except | 1 | ✅ |
| HTF_INTERVAL duplication | 9 dosya | ✅ |
| Kod tekrarı | 4 kritik | ✅ |
| Magic number | 30+ (27 confidence + exit tiers + risk guards) | ✅ |
| Uzun fonksiyonlar (>100 satır) | 4 | ✅ |
| God Object | 2 (PortfolioManager 4512, AccountService 1701) | ✅ |

---

## 2. Faz 1 — Düşük Asmalı Meyveler (1-2 saat)

### 2.1 Dead Importları Temizle

| Dosya | Import | Durum | Eylem |
|-------|--------|-------|-------|
| `account_service.py:10` | `TradeHistoryEntry` | DEAD | Sil |
| `portfolio_manager.py:16` | `Literal` | DEAD | Import listesinden çıkar |
| `portfolio_manager.py:10` | `import os` | DEAD | Sil |

**NOT:** `ai_service.py`'deki `warnings` ve `EnhancedContextProvider` importları KULLANILIyor — dokunma.

### 2.2 Redundant Importları Temizle
- `portfolio_manager.py` — 19 yerel `from config.config import Config` → Modül seviyesinde zaten var, 19'unu sil

### 2.3 Bare `except:` Düzelt
- `ai_service.py:1412` — `except:` → `except (TypeError, ValueError, ZeroDivisionError):`

### 2.4 HTF_INTERVAL Tek Noktada Tanımla
- 9 dosyada aynı kod tekrarlanıyor
- **Çözüm:** `constants.py`'ye ekle: `HTF_INTERVAL = getattr(Config, "HTF_INTERVAL", "1h") or "1h"`
- 9 dosyadan sil, `from src.core.constants import HTF_INTERVAL` ekle

---

## 3. Faz 2 — Paylaşılan Utility'leri Çıkar (2-4 saat)

### 3.1 Trend Yön Hesaplama Birleştir
- `portfolio_manager.py:2049` (`_calculate_trend_direction`) — identical logic
- `market_data.py:926` (`_determine_trend`) — identical logic
- **Çözüm:** `indicators.py`'ye `determine_trend(price, ema20, neutral_band)` fonksiyonu ekle
- Her iki dosyadan mevcut kodu sil, ortak fonksiyonu kullan

### 3.2 Volume Quality Score Birleştir
- `strategy_analyzer.py:28` ve `portfolio_manager.py:2207` — identical scoring
- **Çözüm:** Tek implementasyon `StrategyAnalyzer`'da kalsın
- `PortfolioManager.calculate_volume_quality_score()`'ı sil, `StrategyAnalyzer`'ı çağır

### 3.3 execute_live_close Tekillik
- `portfolio_manager.py:1570` — stripped-down, stale version
- `account_service.py:422` — mature, complete version
- **Çözüm:** `PortfolioManager.execute_live_close()`'ı sil
- `main.py` zaten `1AccountService`'e yönlendiriyor — portfolio_manager versiyonunu temizle

### 3.4 Retry Logic Tekrarını Azalt
- `market_data.py:148-277` — aynı retry+circuit-breaker kodu 3 kez tekrarlanmış
- **Çözüm:** `_handle_api_error(attempt, max_retries, symbol, interval, exception_type)` helper çıkar
- Circuit breaker mantığını da helper'a taşı

---

## 4. Faz 3 — Magic Number'ları Constants'a Taşı (2-3 saat)

### 4.1 Confidence Multiplier Sabitleri
`constants.py`'ye ekle:
```python
# Confidence Adjustments
MOMENTUM_WEAK_PENALTY = 0.90
MOMENTUM_STRONG_BONUS = 1.10
CT_RISK_PENALTY = 0.90
TF_REWARD_BONUS = 1.10
VWAP_ADJUSTMENT = 0.95
VOLUME_FLOOR_PENALTY = 0.85
LOW_VOLUME_PENALTY = 0.92
ADX_STRONG_BONUS = 1.05
ADX_WEAK_PENALTY = 0.90
INDICATOR_ALIGNMENT_BONUS = 1.05
INDICATOR_CLASH_PENALTY = 0.90
TREND_CLASH_PENALTY = 0.85
FLIP_GUARD_PENALTY = 0.97
ML_STRONG_BONUS = 1.10
ML_WEAK_PENALTY = 0.90
TREND_STRONG_BONUS = 1.10
TREND_NEUTRAL = 1.00
TREND_WEAK_PENALTY = 0.90
CHOPPY_PENALTY = 0.70
CONFIDENCE_DECAY_098 = 0.98
```

### 4.2 Exit Tiers Yapılandırması
`constants.py`'ye ekle:
```python
EXIT_TIERS = {
    50: {"level1": 0.008, "level2": 0.008, "level3": 0.008, ...},
    100: {...},
    250: {...},
    500: {...},
    1000: {...},
}
```

### 4.3 Risk Guard Sabitleri
`constants.py`'ye ekle:
```python
MAX_MARGIN_CASH_FRACTION = 0.40
MIN_CASH_GUARD_PCT = 0.10
CYCLES_PER_DAY = 720
```

---

## 5. Faz 4 — God Object'leri Parçala (8-12 saat)

### 5.1 PortfolioManager → 5 Sınıfa Böl

Mevcut: **4512 satır**, 98 metod.

| Yeni Sınıf | Sorumluluk | Tahmini Satır |
|------------|------------|---------------|
| `PortfolioStateManager` | State persistence (load/save, history) | ~200 |
| `DirectionalBiasManager` | Bias logic, win/loss handling | ~400 |
| `CooldownManager` | Cooldown tracking, activation | ~150 |
| `TrendStateManager` | Trend state, flip guards, intraday | ~500 |
| `TradeExecutor` | Order execution, sizing, margin | ~400 |
| `PortfolioManager` (facade) | Koordinasyon, delegasyon | ~300 |

**Yaklaşım:** Alt sınıfları oluştur, `PortfolioManager`'ı facade olarak yeniden yaz. Mevcut API korunur.

### 5.2 AccountService → 2 Sınıfa Böl

Mevcut: **1701 satır**.

| Yeni Sınıf | Sorumluluk | Tahmini Satır |
|------------|------------|---------------|
| `EnhancedExitStrategy` | TP/SL, trailing, partial close, graduated loss | ~600 |
| `AccountService` | Binance entegrasyonu, balance sync, live init | ~500 |

---

## 6. Faz 5 — Uzun Metodları Basitleştir (4-6 saat)

### 6.1 `check_and_execute_tp_sl` (409 satır → ~100 satır)
- `_handle_partial_close_simulation()` çıkar
- `_handle_partial_close_live()` çıkar
- `_handle_full_close_simulation()` çıkar
- `_handle_full_close_live()` çıkar
- `_handle_graduated_stop_loss()` çıkar

### 6.2 `_evaluate_trailing_stop` (293 satır → ~120 satır)
- `_calculate_trailing_stop_long()` çıkar
- `_calculate_trailing_stop_short()` çıkar
- `_get_trailing_context()` çıkar (indicator fetching)

### 6.3 `get_features_for_ml` (234 satır → ~100 satır)
- `_compute_price_features()` çıkar
- `_compute_momentum_features()` çıkar
- `_compute_volatility_features()` çıkar
- `_compute_structure_features()` çıkar
- Inline ADX hesaplamasını mevcut `calculate_adx()` ile değiştir

### 6.4 `generate_alpha_arena_prompt_json` (280 satır → ~120 satır)
- `_compute_ml_predictions()` çıkar
- `_filter_tradeable_coins()` çıkar
- Prompt string template'ini ayır

---

## 7. Uygulama Sırası

```
Faz 1 (1-2s)  → Faz 2 (2-4s)  → Faz 3 (2-3s)  → Faz 4 (8-12s)  → Faz 5 (4-6s)
Dead imports    Utility'ler      Constants        God Objects      Uzun metodlar
Bare except     Trend birleşik   Exit tiers       Account split    Method decomposition
HTF_INTERVAL    Volume score     Risk guards      PM facade
Local imports   execute_close
Retry logic
```

**Toplam tahmini süre:** 17-27 saat

---

## 8. Risk Yönetimi

- Her fazdan önce `git commit` ile mevcut durumu kaydet
- Her fazdan sonra `ruff check` + `ty check` + `pytest tests/ -v` çalıştır
- Faz 4 (God Object) en riskli — incremental yaklaş, her alt sınıf ayrı test et
- `graphify update ./src` ile graph'i güncelle

---

## 9. Docstring ve Test Kuralları

Refactoring sırasında:
- Her yeni fonksiyona docstring ekle (Google-style)
- Her extract edilen fonksiyonun imzasını netleştir
- Modül seviyesinde `__all__` tanımla
- Yeni fonksiyonlar için test ekle
- Mevcut testleri güncelle (kırılan testleri düzelt)

---

## 10. Değişiklik Özeti (v1 → v2)

| Değişiklik | Açıklama |
|------------|----------|
| Dead import listesi güncellendi | `warnings` ve `EnhancedContextProvider` KULLANILIyor — çıkarıldı |
| Bare except konumu düzeltildi | `1340` → `1412` (doğru satır) |
| God Object satır sayıları güncellendi | PM: 4522→4512, AS: 1560→1701 |
| Long method satır sayıları güncellendi | check_and_execute_tp_sl: 403→409, _evaluate_trailing_stop: 275→293 |
| Retry logic detayı eklendi | Circuit breaker da tekrarlanıyor |
| Exit tier magic numbers detaylandı | Tüm threshold ve take percentage'ler hardcoded |
