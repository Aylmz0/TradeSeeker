# Refactoring Plan — TradeSeeker

**Tarih:** 2026-07-21
**Sıra:** Docstring → Test → **Refactoring** (bu plan)
**Kapsam:** Tüm src/ kodu

---

## 1. Mevcut Durum Analizi

| Sorun | Sayı | Etki |
|-------|------|------|
| Kod tekrarı | 5 kritik | Yüksek |
| Uzun fonksiyonlar (>100 satır) | 7 | Yüksek |
| God Object | 2 (PortfolioManager 4522 satır) | Yüksek |
| Magic number | 30+ | Yüksek |
| Karmaşık conditionals | 4 | Orta |
| Ölü importlar | 6 | Düşük |
| Tutarlı olmayan pattern'ler | 6 | Orta |

---

## 2. Faz 1 — Düşük Asmalı Meyveler (1-2 saat)

**Hedef:** Hızlı temizlik, minimum risk.

### 2.1 Ölü Importları Temizle
| Dosya | Import | Durum |
|-------|--------|-------|
| `ai_service.py:4` | `import warnings` | Sadece deprecated fonksiyonda kullanılıyor |
| `ai_service.py:11` | `EnhancedContextProvider` | Kullanım kaldırıldı ama import duruyor |
| `account_service.py:10` | `TradeHistoryEntry` | Hiç kullanılmıyor |
| `portfolio_manager.py:16` | `Literal` | Hiç kullanılmıyor |
| `portfolio_manager.py:10` | `import os` | Hiç kullanılmıyor |
| `portfolio_manager.py` (19 yer) | Yerel `from config.config import Config` | Modül seviyesinde import zaten var |

**Eylem:** Tümünü sil.

### 2.2 Bare `except:` Düzelt
- `ai_service.py:1340` — `except:` → `except (ZeroDivisionError, TypeError, ValueError):`

### 2.3 HTF_INTERVAL Tek Noktada Tanımla
- 9 dosyada aynı kod: `HTF_INTERVAL = getattr(Config, "HTF_INTERVAL", "1h") or "1h"`
- **Çözüm:** `constants.py`'ye ekle, her yerden import et.

---

## 3. Faz 2 — Paylaşılan Utility'leri Çıkar (2-4 saat)

**Hedef:** Kod tekrarını azalt.

### 3.1 Trend Yön Hesaplama Birleştir
- `portfolio_manager.py:1994-2014` (`_calculate_trend_direction`)
- `market_data.py:777-783` (`_determine_trend`)
- **Çözüm:** Tek `determine_trend(price, ema20)` fonksiyonu, `utils.py`'ye veya `indicators.py`'ye.

### 3.2 Volume Quality Score Birleştir
- `strategy_analyzer.py:21-54` ve `portfolio_manager.py:2164-2204`
- **Çözüm:** Tek implementasyon `StrategyAnalyzer`'da kal, `PortfolioManager`'dan çağrılır.

### 3.3 execute_live_close Tekillik
- `portfolio_manager.py:1527-1567` — ölü kod
- `account_service.py:373-446` — tam implementasyon
- **Çözüm:** `PortfolioManager.execute_live_close()`'ı sil, tüm canlı close'lar `AccountService`'den geçsin.

### 3.4 Retry Logic Tekrarını Azalt
- `market_data.py:191-243` — aynı retry kodu 3 kez tekrarlanmış
- **Çözüm:** `_handle_api_error(attempt, max_retries, symbol, interval, exception_type)` çıkar.

---

## 4. Faz 3 — Magic Number'ları Constants'a Taşı (2-3 saat)

**Hedef:** Yapılandırma mümkünluğu, okunabilirlik.

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
```

### 4.2 Exit Tiers Yapılandırması
```python
EXIT_TIERS = {
    50: {"level1": 0.008, "level2": 0.012, "level3": 0.018, ...},
    100: {...},
    250: {...},
    500: {...},
    1000: {...},
}
```

### 4.3 Risk Guard Sabitleri
```python
MAX_MARGIN_CASH_FRACTION = 0.40
MIN_CASH_GUARD_PCT = 0.10
CYCLES_PER_DAY = 720  # veya Config'den hesapla
```

---

## 5. Faz 4 — God Object'leri Parçala (8-12 saat)

**Hedef:** Sorumluluk ayrımı, bakım kolaylığı.

### 5.1 PortfolioManager → 5 Sınıfa Böl

Mevcut: 4522 satır, 70+ metod.

| Yeni Sınıf | Sorumluluk | Tahmini Satır |
|------------|------------|---------------|
| `PortfolioStateManager` | State persistence (load/save, history) | ~200 |
| `DirectionalBiasManager` | Bias logic, win/loss handling | ~400 |
| `CooldownManager` | Cooldown tracking, activation | ~150 |
| `TrendStateManager` | Trend state, flip guards, intraday | ~500 |
| `TradeExecutor` | Order execution, sizing, margin | ~400 |
| `PortfolioManager` (facade) | Koordinasyon, delegasyon | ~300 |

**Yaklaşım:** Alt sınıfları oluştur, `PortfolioManager`'ı facade olarak yeniden yaz. Mevcut API korunur (dışarıdan aynı metodlar çağrılır).

### 5.2 AccountService → 2 Sınıfa Böl

Mevcut: 1560 satır.

| Yeni Sınıf | Sorumluluk | Tahmini Satır |
|------------|------------|---------------|
| `EnhancedExitStrategy` | TP/SL, trailing, partial close, graduated loss | ~600 |
| `AccountService` | Binance entegrasyonu, balance sync, live init | ~500 |

---

## 6. Faz 5 — Uzun Metodları Basitleştir (4-6 saat)

**Hedef:** Okunabilirlik, test edilebilirlik.

### 6.1 `check_and_execute_tp_sl` (403 satır → ~100 satır)
- `_handle_partial_close_simulation()` çıkar
- `_handle_partial_close_live()` çıkar
- `_handle_full_close_simulation()` çıkar
- `_handle_full_close_live()` çıkar

### 6.2 `_evaluate_trailing_stop` (275 satır → ~120 satır)
- `_calculate_trailing_stop_long()` çıkar
- `_calculate_trailing_stop_short()` çıkar
- `_get_trailing_context()` çıkar (indicator fetching)

### 6.3 `get_features_for_ml` (227 satır → ~100 satır)
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
Low-hanging     Utility'ler      Constants        God Objects      Uzun metodlar
fruit extraction extraction decomposition
```

**Toplam tahmini süre:** 17-27 saat

---

## 8. Risk Yönetimi

- Her fazdan önce `git commit` ile mevcut durumu kaydet
- Her fazdan sonra `ruff check` + `ty check` çalıştır
- Faz 4 (God Object) en riskli — incremental yaklaş, test ile doğrula
- `graphify update ./src` ile graph'i güncelle

---

## 9. Docstring ve Test İçin Hazırlık

Refactoring sırasında:
- Her yeni fonksiyona docstring ekle (Google-style)
- Her extract edilen fonksiyonun imzasını netleştir
- Modül seviyesinde __all__ tanımla

Bu, sonraki B (docstring) ve A (test) fazlarını kolaylaştırır.
