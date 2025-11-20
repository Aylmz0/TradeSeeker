# Prompt Optimizasyon Planı

**Hedef:** AI yanıt süresini ~61 saniyeden ~25-30 saniyeye düşürmek.  
**Tarih:** 2025-11-20  
**Güncel AI Süresi:** ~61 saniye (61,215 ms)

---

## 📊 Mevcut Prompt Analizi

### Token Dağılımı (Tahmini)
| Bölüm | Token (Yaklaşık) | Yüzde |
|:------|:-----------------|:------|
| MARKET_DATA (6 coin × 3 timeframe × 50 series) | ~15,000 | 60% |
| Counter-Trade + Reversal Analysis | ~2,500 | 10% |
| Enhanced Context + Directional Bias | ~2,000 | 8% |
| System Prompt (Her requestte tekrar) | ~3,000 | 12% |
| Cooldown + Position Slots + Portfolio | ~1,500 | 6% |
| Tekrarlanan Açıklama Metinleri |~1,000 | 4% |
| **TOPLAM** | **~25,000** | **100%** |

---

## ⚠️ Risk Analizi: Series Uzunluğunu Kısaltmak

### Soru: `JSON_SERIES_MAX_LENGTH=50` → `15` yaparsak sorun olur mu?

**CEVAP: HAYIR, ciddi bir sorun olmaz. İşte analizi:**

#### ✅ Neden Güvenli:

1. **AI Zaten Son Değerlere Odaklanıyor**
   - AI chain_of_thoughts'unda sadece en son RSI, MACD, price değerlerini kullanıyor
   - 50 veri noktasının 45'i hiç kullanılmıyor
   - Son 15 veri = 45 dakikalık trend (3m interval × 15)

2. **Trend Tespiti İçin Yeterli**
   - 15 veri noktası = 45 dakika (3m için)
   - 15 veri noktası = 3.75 saat (15m için)  
   - 15 veri noktası = 15 saat (1h için)
   - Bu, momentum değişimini yakalamak için YETERLİ

3. **Mevcut Kodda Zaten Compression Var**
   - `prompt_json_utils.py` zaten 50+ veriyi sıkıştırıyor
   - 15'e düşürünce compression'a bile gerek kalmaz

#### ⚠️ Potansiyel Riskler:

1. **Çok Düşük Volatilitede** trend görünmeyebilir
   - **Çözüm:** 15 yerine 20 kullan (ikisi arasında sweet spot)

2. **Long-term Pattern Recognition**
   - AI daha uzun kalıpları göremez
   - **Ama:** Zaten 3m, 15m, 1h için farklı timeframe'ler var

**SONUÇ: `JSON_SERIES_MAX_LENGTH=20` GÜVENL İ VE OPTİMAL**

---

## 🎯 Tespit Edilen Tekrar ve Ölü Promptlar

### 1. **Tekrarlayan Açıklamalar (EN BÜYÜK SORUN)**

#### ❌ Gereksiz Tekrarlar:

```markdown
# User Prompt Line ~5929:
"We pre-compute the standard 5 counter-trend conditions for every coin. 
Review these findings first; only recalc if you detect inconsistencies or need extra validation."

# User Prompt Line ~5935:
"All notes below are informational statistics about potential reversals; 
evaluate them independently before acting."

# User Prompt Line ~5941:
"Metrics and remarks in this section are informational only. 
You must weigh them yourself before making any trading decision."
```

**Analiz:** Bu açıklamalar **System Prompt'ta zaten var**. User Prompt'ta tekrar gereksiz.

**Tasarruf:** ~800-1000 token

---

### 2. **Ölü/Az Kullanılan JSON Bölümleri**

#### 📦 HISTORICAL_CONTEXT (Düşük Değer)
```json
{
  "total_cycles_analyzed": 50,
  "market_behavior": "Trending Up",
  "recent_decisions": [...],
  "performance_trend": "Improving"
}
```

**Analiz:** AI bu veriyi **hiç kullanmıyor**. Chain of thoughts'ta referans yok.

**Önerilen Aksiyon:** 
- **Kaldır** veya
- **Sadece 3 coin için** (şu anda 6 coin için gönderiliyor)

**Tasarruf:** ~1,200 token

---

#### 📦 ENHANCED_CONTEXT.suggestions (Düşük Etki)
```json
"suggestions": [
  "[INFO] Bearish regime detected with ≥3 open positions",
  "[INFO] Bullish regime detected with zero current exposure"
]
```

**Analiz:** AI bu önerileri **bazen** okuyor ama kararlarını değiştirmiyor.

**Önerilen Aksiyon:** 
- Sadece **kritik durumlar için** göster (örn: cash < $20, position count > 4)
- Normal durumlarda boş array gönder

**Tasarruf:** ~400 token

---

### 3. **Counter-Trade vs Trend Reversal (Overlap)**

**Sorun:** İkisi de benzer bilgi veriyor:
- `COUNTER_TRADE_ANALYSIS`: "1h BEARISH, 15m+3m BULLISH"
- `TREND_REVERSAL_DATA`: "1h reversal: true, 3m reversal: true"

**Fark:**
- Counter-Trade: **Yeni pozisyon** açmak için
- Trend Reversal: **Mevcut pozisyon** kapatmak için

**Önerilen Aksiyon:**
- `TREND_REVERSAL_DATA` sadece **pozisyon varsa** gönder
- Pozisyon yoksa bu bölümü **hiç gönderme**

**Tasarruf (pozisyon yokken):** ~800 token

---

### 4. **Market Data Timeframe Seçimi**

**Sorun:** Her coin için **3 timeframe × 50 veri** gönderiliyor.

**AI Kullanımı:**
- **3m:** Giriş timing (EN ÖNEMLİ)
- **15m:** Momentum konfirmasyonu (ÖNEMLİ)
- **1h:** Trend yönü (ÖNEMLİ)

**Öneri:**
- 3m: 20 veri (1 saat)
- 15m: 15 veri (3.75 saat)
- 1h: 12 veri (12 saat)

**Tasarruf:** ~60% (15,000 → 6,000 token)

---

## 📋 İmplementasyon Planı

### Faz 1: Hemen Uygulanabilir (Risk: DÜŞÜK)

#### ✅ Adım 1: Config Değişiklikleri
```bash
# .env dosyasına ekle:
JSON_SERIES_MAX_LENGTH=20  # 50'den 20'ye
JSON_PROMPT_COMPACT=true   # JSON indent kaldır
```

**Beklenen Süre:** 61s → 40s (%35 azalma)

---

#### ✅ Adım 2: Gereksiz Açıklamaları Kaldır

**Dosya:** `alpha_arena_deepseek.py` (lines 5927-5970)

**Kaldırılacak satırlar:**
```python
# Line ~5929: Kaldır
"We pre-compute the standard 5 counter-trend conditions..."

# Line ~5935: Kaldır  
"All notes below are informational statistics..."

# Line ~5941: Kaldır
"Metrics and remarks in this section are informational only..."
```

**Beklenen Süre:** 40s → 35s (%12 azalma)

---

### Faz 2: Koşullu Optimizasyon (Risk: ORTA)

#### ✅ Adım 3: TREND_REVERSAL sadece pozisyon varsa gönder

**Dosya:** `alpha_arena_deepseek.py` (line ~5937)

**Değişiklik:**
```python
# Şu anki:
{create_json_section("TREND_REVERSAL_DATA", trend_reversal_json, compact=compact)}

# Yeni:
if any(pos for pos in self.portfolio.positions.values()):
    {create_json_section("TREND_REVERSAL_DATA", trend_reversal_json, compact=compact)}
else:
    # Pozisyon yoksa atla
    pass
```

**Beklenen Süre:** 35s → 30s (pozisyon yokken)

---

#### ✅ Adım 4: HISTORICAL_CONTEXT kısalt

**Dosya:** `prompt_json_builders.py`

**Değişiklik:**
```python
# recent_decisions maksimum 5 göster (şu anda 10)
recent_decisions = trading_context.get('recent_decisions', [])[-5:]
```

**Beklenen Süre:** 30s → 28s

---

### Faz 3: Agresif Optimizasyon (Risk: YÜKSEK)

#### ⚠️ Adım 5: Timeframe-specific series length

**Dosya:** `prompt_json_builders.py` → `build_market_data_json`

**Değişiklik:**
```python
# Farklı timeframe'ler için farklı uzunluklar
series_lengths = {
    '3m': 20,   # 1 saat
    '15m': 15,  # 3.75 saat  
    '1h': 12    # 12 saat
}
```

**Beklenen Süre:** 28s → 22s (%22 azalma)

**Risk:** Orta - AI'ın farklı timeframe'lerde farklı derinlikte data olur

---

## 🎬 Önerilen İş Akışı

### Option A: Muhafazakar (ÖNERİLEN)
1. Faz 1 Adım 1-2 uygula → Test et
2. Başarılıysa Faz 2 uygula → Test et
3. Faz 3'ü sadece gerekirse uygula

**Hedef Süre:** 30-35 saniye (40% azalma)

### Option B: Agresif
1. Tüm Faz 1-2-3'ü birden uygula
2. Sorun çıkarsa geri al

**Hedef Süre:** 22-25 saniye (60% azalma)  
**Risk:** Orta

---

## 📊 Beklenen Sonuçlar

| İyileştirme | Token Tasarrufu | Süre Kazancı | Risk |
|:------------|:----------------|:-------------|:-----|
| Series 50→20 | ~9,000 | 60s → 40s | DÜŞÜK |
| Compact JSON | ~1,500 | -3s | YOK |
| Açıklama Kaldır | ~1,000 | -2s | YOK |
| Reversal Koşullu | ~800 | -2s | DÜŞÜK |
| Historical Kısalt | ~600 | -2s | DÜŞÜK |
| Timeframe-specific | ~3,000 | -6s | ORTA |
| **TOPLAM** | **~15,900** | **61s → 25s** | **DÜŞÜK-ORTA** |

---

## ✅ Sonuç ve Öneri

**En Güvenli ve Etkili Yaklaşım:**

1. ✅ `JSON_SERIES_MAX_LENGTH=20` yap (config)
2. ✅ `JSON_PROMPT_COMPACT=true` yap (config)
3. ✅ Gereksiz açıklamaları kaldır (kod)
4. ✅ TREND_REVERSAL koşullu yap (kod)

**Bu 4 adımla:**
- **Süre:** 61s → ~30s (%50 azalma)
- **Risk:** Çok düşük
- **Test Süresi:** 2-3 cycle yeterli

Daha agresif optimizasyon gerekirse Faz 3'e geçebiliriz.
