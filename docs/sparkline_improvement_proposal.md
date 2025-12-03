# Sparkline ve Veri Temsili İyileştirme Önerileri

Bu doküman, TradeSeeker projesinde Yapay Zeka'ya sunulan piyasa verilerinin (özellikle fiyat grafiklerinin) temsilini iyileştirmek için teknik öneriler içerir. Bu öneriler, projenin "Gelecek Çalışmalar" (Future Work) bölümünde sunulabilir veya hayata geçirilebilir.

## Mevcut Durum (Baseline)
Şu anki sistemde **8 Seviyeli ASCII Karakterleri** (` ▂▃▄▅▆▇█`) kullanılmaktadır.
*   **Yöntem:** Son 24 mumun fiyat verisi alınır, min-max normalizasyonu yapılır ve 0-7 arasına ölçeklenerek ilgili karaktere dönüştürülür.
*   **Kısıt:** Dikey çözünürlük sadece 8 pikseldir. Küçük volatilite değişimleri ile büyük trendler arasındaki fark kaybolabilir.

---

## 🚀 SEÇİLEN YAKLAŞIM: "AKILLI SPARKLINE" (Smart Sparkline)

Yapay Zeka modellerinin (LLM) görsel veriden ziyade **semantik (anlamsal) ve mantıksal** veriyi daha iyi işlediği gerçeğinden yola çıkarak, "İki Katmanlı Veri Temsili" modeli benimsenmiştir.

### 1. Alt Katman: Matematiksel Analiz (Python)
Python'un güçlü kütüphaneleri (`numpy`) kullanılarak fiyat serisi üzerinde deterministik analizler yapılır. AI'a "resmi yorumla" demek yerine, resmin matematiksel özellikleri çıkarılır.

*   **Trend Eğimi (Slope):** Lineer regresyon ile trendin yönü ve şiddeti hesaplanır.
*   **Tepe/Dip Analizi:** Yerel maksimum ve minimum noktalar tespit edilir.
*   **Volatilite:** Standart sapma üzerinden oynaklık durumu belirlenir.

### 2. Üst Katman: Semantik Özet (AI Prompt)
Elde edilen matematiksel veriler, AI'ın anlayacağı zenginleştirilmiş bir JSON formatına dönüştürülür.

```json
{
  "smart_sparkline": {
    "visual": "↗️↗️⏫↘️➡️↗️",  // Vektörel Hareket (Görsel İlüzyonu Önler)
    "semantic": "STRONG_UPTREND_WITH_PULLBACK", // Anlamsal Özet
    "critical_points": "PEAK_AT_155,PULLBACK_TO_142", // Kritik Seviyeler
    "trend_slope": 0.0023 // Kesin Matematiksel Eğim
  }
}
```

### Neden Bu Yöntem?
1.  **Hibrit Zeka:** Python'un hesaplama gücü ile LLM'in muhakeme gücünü birleştirir.
2.  **Hata Payı:** AI'ın ASCII karakterlerini yanlış yorumlama (halüsinasyon) riskini sıfıra indirir.
3.  **Verimlilik:** AI, karmaşık görseli çözmek yerine doğrudan "sonuca" odaklanır.

---

## Diğer Alternatifler (Değerlendirildi ve Elendi)

### Alternatif 1: Braille Desenleri (High-Res Visuals)
*   **Tanım:** Braille karakterleri ile 4 kat yüksek çözünürlük.
*   **Durum:** Görsel olarak etkileyici olsa da, LLM tokenization sorunları nedeniyle "Smart Sparkline" kadar verimli bulunmadı.

### Alternatif 2: Saf Vektörel Temsil
*   **Tanım:** Sadece ok işaretleri (`↗ ↘`) kullanmak.
*   **Durum:** "Smart Sparkline" içine entegre edildi. Tek başına kullanıldığında büyüklük (magnitude) bilgisini kaybedebilir.

---

## Uygulama Planı

1.  **Feature Extraction:** `numpy` kullanılarak fiyat serisinden eğim ve tepe noktalarının çıkarılması.
2.  **Vector Generation:** Fiyat değişimlerinin sembolik vektörlere (`↗`, `↘`) dönüştürülmesi.
3.  **Prompt Entegrasyonu:** `prompt_json_builders.py` dosyasının güncellenmesi.
