# JSON Prompt Test Rehberi

## 🧪 1 Günlük Test İçin .env Ayarları

### Minimum Gerekli Ayarlar

```bash
# JSON Prompt'u aktif et
USE_JSON_PROMPT=true

# Token tasarrufu için compact mode (önerilir)
JSON_PROMPT_COMPACT=true
```

### Opsiyonel Ayarlar (Test için gerekli değil)

```bash
# Runtime validation (performansı biraz etkileyebilir, ilk test için false bırak)
VALIDATE_JSON_PROMPTS=false

# Series compression threshold (varsayılan 50, değiştirmeye gerek yok)
JSON_SERIES_MAX_LENGTH=50

# Cache (ilk test için gerekli değil)
JSON_CACHE_ENABLED=false
JSON_CACHE_TTL=120
```

## 📊 Test Sırasında İzlenecekler

### 1. Cycle History'de Format Kontrolü

Her cycle'da `cycle_history.json` dosyasında şu alanları kontrol et:

```json
{
  "cycle": 1,
  "metadata": {
    "prompt_format": "json",  // veya "text" veya "json_fallback"
    "json_serialization_error": null  // Hata varsa burada görünür
  }
}
```

### 2. Console Log'ları

Bot çalışırken şu mesajları göreceksin:

- ✅ `Using JSON prompt format (version 1.0)` - JSON format kullanılıyor
- ⚠️ `JSON prompt generation failed: ...` - Hata olursa text format'a döner
- ⚠️ `Falling back to text format...` - Fallback çalıştı

### 3. Token Kullanımı Karşılaştırması

Test sonrası karşılaştırma için:

```bash
# Test sonuçlarını analiz et
python test_prompt_ab_comparison.py
```

### 4. İzleme Checklist

- [ ] JSON format başarıyla kullanılıyor mu? (`prompt_format: "json"`)
- [ ] Fallback oluyor mu? (`prompt_format: "json_fallback"` varsa neden?)
- [ ] AI response kalitesi nasıl? (chain_of_thoughts ve decisions kontrol et)
- [ ] Token kullanımı azaldı mı? (compact mode ile)
- [ ] Hata var mı? (`json_serialization_error` kontrol et)

## 🔄 Test Sonrası

### Başarılı Test Sonrası

Eğer her şey yolundaysa:
- `USE_JSON_PROMPT=true` olarak bırak
- `JSON_PROMPT_COMPACT=true` olarak bırak (token tasarrufu için)

### Sorun Varsa

- `USE_JSON_PROMPT=false` yaparak eski format'a dön
- Hataları `cycle_history.json`'dan kontrol et
- `json_serialization_error` alanına bak

## 📝 Örnek .env Ayarları

```bash
# JSON Prompt Test Ayarları
USE_JSON_PROMPT=true
JSON_PROMPT_COMPACT=true
VALIDATE_JSON_PROMPTS=false
JSON_PROMPT_VERSION=1.0
JSON_SERIES_MAX_LENGTH=50
JSON_CACHE_ENABLED=false
JSON_CACHE_TTL=120
```

## 🚨 Önemli Notlar

1. **Fallback Güvenliği**: Hata olursa otomatik olarak text format'a döner, bot çalışmaya devam eder
2. **İlk Cycle'lar**: İlk birkaç cycle'da JSON format test edilir, sorun yoksa devam eder
3. **Monitoring**: `cycle_history.json`'daki `metadata.prompt_format` alanını izle
4. **Rollback**: Her zaman `USE_JSON_PROMPT=false` yaparak eski format'a dönebilirsin

