# Test Gap Fix Plan — TradeSeeker

**Tarih:** 2026-07-21
**Amaç:** Mevcut 128 testi mükemmelleştir + eksik testleri ekle

---

## 1. conftest.py Düzeltmeleri

- `sample_ohlcv` timestamp generation: `pl.Series([...]).__mul__(1)` → düzgün `pl.Series("timestamp", [...])`

## 2. test_data_engine.py

- Shadow eden `sample_ohlcv` fixture'ını sil, conftest'dekini kullan (veya düzelt)

## 3. Yeni Test Dosyaları

### 3.1 `tests/unit/test_utils_extended.py`
- `test_rate_limiter`: decorator çalışır
- `test_retry_manager`: session oluşturur
- `test_data_validator`: dataframe validasyonu
- `test_cleanup_stale_temp_files`: eski tmp dosyaları temizler
- `test_safe_file_read_cached`: cache hit/miss
- `test_safe_file_read_invalid_json`: bozuk JSON → default

### 3.2 `tests/unit/test_performance_extended.py`
- `test_generate_adaptive_suggestions`: öneriler üretilir
- `test_generate_reversal_recommendations`: reversal önerileri üretilir
- `test_analyze_performance`: tam analiz çalışır
- `test_analyze_performance_no_data`: veri yoksa info döner

### 3.3 `tests/unit/test_schemas_extended.py`
- `test_trailing_meta`: TrailingMeta schema
- `test_trend_context`: TrendContext schema
- `test_directional_bias`: DirectionalBias schema
- `test_cycle_history_entry`: CycleHistoryEntry schema

## 4. Mevcut Dosyalara Edge Case Ekleme

### 4.1 test_indicators.py
- `test_rsi_period_too_large`: period > len(prices)
- `test_ema_period_one`: period=1
- `test_adx_single_point`: tek veri noktası
- `test_bb_insufficient_data`: < period data

### 4.2 test_regime_detector.py
- `test_classify_price_equals_ema`: price == ema20
- `test_overall_regime_with_averaged_ers`: averaged ER parametresi
- `test_strength_all_bearish`: tümü bearish → strength=1.0

### 4.3 test_enhanced_ctx.py
- `test_performance_insights`: performance insights
- `test_market_regime_context`: regime context
- `test_empty_positions`: pozisyon yoksa
- `test_bearish_suggestions`: bearish + 3 pozisyon

### 4.4 test_strategy.py
- `test_bearish_regime`: bearish indicators
- `test_zero_avg_volume`: avg_volume=0
- `test_unclear_regime`: hata varsa UNCLEAR

### 4.5 test_deepseek_api.py
- `test_ratelimit_error`: RateLimit → cached decisions
- `test_cached_decisions_with_ghost`: ghost entry blocking

### 4.6 test_market_data.py
- `test_drawdown_same_prices`: tüm fiyatlar aynı
- `test_validate_negative_volume`: negatif hacim

---

## 5. Uygulama Sırası

| Adım | İşlem | Süre |
|------|-------|------|
| 1 | conftest.py düzelt | 10 dk |
| 2 | test_data_engine.py fixture düzelt | 5 dk |
| 3 | test_utils_extended.py | 30 dk |
| 4 | test_performance_extended.py | 30 dk |
| 5 | test_schemas_extended.py | 20 dk |
| 6 | Edge case ekleme (mevcut dosyalar) | 1 saat |
| 7 | Final test çalıştırma + coverage | 15 dk |
| **Toplam** | | **~2.5 saat** |

---

## 6. Hedef

| Metrik | Mevcut | Hedef |
|--------|--------|-------|
| Test sayısı | 128 | ≥160 |
| Coverage | %20 | ≥%25 |
| Edge case | ~0 | ≥35 |
| Eksik fonksiyon | 10 | 0 |
