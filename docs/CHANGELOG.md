# Changelog

Tüm önemli değişiklikler bu dosyada belgelenmiştir.

Format [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) standardına uygundur.

---

## [1.0.0] - 2025-11-17

### Added - JSON Prompt Format (v1.0)

#### Yeni Özellikler
- ✅ **JSON Prompt Format**: AI prompt'ları artık JSON formatında gönderiliyor
- ✅ **Hybrid Prompt System**: JSON veri + text instruction kombinasyonu
- ✅ **11 JSON Section**: Tüm prompt bölümleri JSON format'ta
  - COUNTER_TRADE_ANALYSIS
  - TREND_REVERSAL_DATA
  - ENHANCED_CONTEXT
  - DIRECTIONAL_BIAS
  - COOLDOWN_STATUS
  - TREND_FLIP_GUARD
  - POSITION_SLOTS
  - MARKET_DATA
  - HISTORICAL_CONTEXT
  - RISK_STATUS
  - PORTFOLIO

#### Yeni Dosyalar
- `prompt_json_builders.py` - JSON builder fonksiyonları
- `prompt_json_utils.py` - JSON utility fonksiyonları (SafeJSONEncoder, compression, etc.)
- `prompt_json_schemas.py` - JSON schema tanımları
- `test_prompt_json.py` - Unit testleri
- `test_prompt_integration.py` - Integration testleri
- `test_prompt_ab_comparison.py` - A/B karşılaştırma testleri
- `TEST_JSON_PROMPT.md` - Test rehberi
- `MIGRATION_GUIDE.md` - Migration rehberi
- `CHANGELOG.md` - Bu dosya

#### Yeni Configuration Flags
- `USE_JSON_PROMPT` - JSON format'ı aktif et/kapat
- `JSON_PROMPT_COMPACT` - Compact JSON format (token tasarrufu)
- `VALIDATE_JSON_PROMPTS` - Runtime JSON validation
- `JSON_PROMPT_VERSION` - Format versiyonu (1.0)
- `JSON_SERIES_MAX_LENGTH` - Series compression threshold (50)
- `JSON_CACHE_ENABLED` - JSON serialization cache (opsiyonel)
- `JSON_CACHE_TTL` - Cache TTL (240 saniye)

#### İyileştirmeler
- ✅ **Token Optimizasyonu**: Compact mode ile %26'ya kadar token tasarrufu
- ✅ **Series Compression**: Büyük seriler otomatik sıkıştırılır (%79 token tasarrufu)
- ✅ **Error Handling**: JSON serialization hatalarında otomatik fallback
- ✅ **NaN/None Handling**: SafeJSONEncoder ile güvenli serialization
- ✅ **Format Versioning**: JSON format versiyonlama

### Changed

#### Prompt Generation
- `generate_alpha_arena_prompt_json()` - Yeni önerilen metod
- `generate_alpha_arena_prompt()` - Deprecated (backward compatibility için korunuyor)

#### Format Functions
- Tüm `format_*()` fonksiyonları deprecated olarak işaretlendi
- JSON builder fonksiyonları kullanılması öneriliyor

### Deprecated

#### Fonksiyonlar
- ⚠️ `generate_alpha_arena_prompt()` - `generate_alpha_arena_prompt_json()` kullanın
- ⚠️ `format_position_context()` - `build_position_slot_json()` kullanın
- ⚠️ `format_market_regime_context()` - JSON builders kullanın
- ⚠️ `format_performance_insights()` - JSON builders kullanın
- ⚠️ `format_directional_feedback()` - JSON builders kullanın
- ⚠️ `format_risk_context()` - `build_risk_status_json()` kullanın
- ⚠️ `format_suggestions()` - JSON builders kullanın
- ⚠️ `format_trend_reversal_analysis()` - `build_trend_reversal_json()` kullanın
- ⚠️ `format_volume_ratio()` - JSON builders kullanın
- ⚠️ `format_list()` - JSON builders kullanın

**Not**: Deprecated fonksiyonlar hala çalışıyor (backward compatibility için), ancak warning verirler.

### Fixed

#### Market Regime Detection
- ✅ Market regime detection logic düzeltildi
- ✅ 1h + (3m OR 15m) alignment kuralı eklendi
- ✅ Daha strict regime detection

#### Total Value Calculation
- ✅ `total_value` hesaplamasına `unrealized_pnl` eklendi
- ✅ Hem `update_prices()` hem de `sync_live_account()` düzeltildi

#### Cooldown System
- ✅ Cooldown kontrolleri düzeltildi
- ✅ `getattr` yerine direkt attribute erişimi kullanılıyor
- ✅ Cooldown'lar artık doğru çalışıyor

### Documentation

- ✅ README.md güncellendi (JSON format bölümü eklendi)
- ✅ `.envexample.txt` güncellendi (JSON prompt flags eklendi)
- ✅ System prompt'ta JSON format açıklamaları eklendi
- ✅ MIGRATION_GUIDE.md oluşturuldu
- ✅ CHANGELOG.md oluşturuldu

---

## [0.x.x] - Önceki Versiyonlar

### Text Format Era
- Text-based prompt generation
- Manual formatting functions
- Basic error handling

---

## Migration Notes

### v0.x → v1.0

1. **JSON Format Aktif Et**:
   ```bash
   USE_JSON_PROMPT=true
   JSON_PROMPT_COMPACT=true
   ```

2. **Test Et**:
   - Bot'u çalıştır
   - JSON format kullanıldığını kontrol et
   - Fallback mekanizmasını test et

3. **Deprecated Warning'leri Kontrol Et**:
   - Eski fonksiyonlar warning verir (normal)
   - Yeni fonksiyonlara geçiş yapılabilir (opsiyonel)

---

*Changelog oluşturulma tarihi: 2025-11-17*

