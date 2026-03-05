# 🚀 TradeSeeker: High-Performance AI Trading Maestro (v9.5)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Binance](https://img.shields.io/badge/Binance-Futures-F0B90B?style=for-the-badge&logo=binance)
![AI](https://img.shields.io/badge/AI-OpenRouter%20%2F%20DeepSeek-black?style=for-the-badge&logo=openai)
![Status](https://img.shields.io/badge/Optimization-V9_Hybrid_ML-success?style=for-the-badge)

**TradeSeeker**, Büyük Dil Modellerinin (LLM - DeepSeek/Trinity vb.) muhakeme gücünü, **XGBoost** tabanlı saf istatistiksel makine öğrenmesi ve **$O(1)$ karmaşıklıkta** optimize edilmiş vektörel indikatör motoru ile birleştiren, yüksek performanslı bir algoritmik ticaret maestrosudur. 

---

## 🧠 Mimarisi: The Hybrid Maestro
TradeSeeker, kararları tek bir beyne bırakmaz. Kantitatif Matematik (XGBoost) ile Stratejik Muhakeme (LLM) arasındaki boşluğu kapatan hibrit bir sistemdir:

1. **XGBoost (Makine Öğrenmesi):** Piyasanın Refleksleri. Geçmiş fiyat hareketlerindeki matematiksel kalıpları ezberler ve her coin için Yükseliş/Düşüş olasılığı (ml_consensus) üretir. "Global Model" mimarisiyle tüm piyasayı tek potada öğrenir.
2. **LLM (Yapay Zeka):** Stratejik Karar Verici. Model destekli (Reasoning) bir dil modeli, piyasa rejimini, indikatörleri ve XGBoost'un olasılıklarını sentezleyerek bir Fon Yöneticisi gibi işlem kararı alır. Dinamik OpenRouter entegrasyonu sayesinde saniyeler içinde Claude, Gemini veya farklı DeepSeek modellerine geçirilebilir.
3. **Core Engine:** Kesin Disiplin. ATR tabanlı stop-loss ve risk yönetim kurallarını milisaniye hızında (NumPy) işler. AI'ın zafiyet göstermesini engeller.

---

## ⚡ Teknik Üstünlükler & Optimizasyonlar
* **Global ML Training Pipeline:** Her coin için ayrı beyin yerine, `train_model.py` tüm borsa verilerini (XRP, SOL, ETH vb.) tek bir devasa "Global Dataset" olarak işler ve genelleştirilmiş bir yapay zeka eğitir.
* **Graceful Degradation (Kusursuz Düşüş):** Eğer ML (XGBoost) verisi/modeli yoksa bot çökmez. Kendini anında izole ederek klasik "AI-Only" moduna geçer ve sadece teknik analize dayanarak sorunsuz çalışmaya devam eder.
* **Hardcore Vektörizasyon:** Tüm teknik indikatörler saf NumPy kullanılarak C hızında hesaplanır. Python `for` döngüleri kullanılmaz.
* **Unified Data Engine (SQLite):** Her karar, "Confidence Skoru", "ML Probability" ve gerçekleşen "Reel PnL (Kâr/Zarar)" ile birlikte SQLite veritabanına loglanır. Bu altyapı gelecekteki **Kendi Kendine Öğrenme (Reinforcement Learning)** aşamasının omurgasıdır.
* **Console Cleansing:** Tamamen stabilize edilmiş ve Emoji/Unicode gürültüsünden arındırılmış, profesyonel terminal logları. 

---

## 🛠️ Kurulum & Kullanım

```bash
# 1. Gereksinimleri yükle
pip install -r requirements.txt

# 2. .env dosyanı yapılandır (Ayarlar ve API Keyler)
cp .env.example .env

# 3. Canlı / Simülasyon döngüsünü başlat (Veri toplamaya başlar)
python3 src/main.py

# 4. Yeterli veri toplandığında (İsteğe bağlı) XGBoost Beynini Eğit
python3 scripts/train_model.py
```

---

## ⚠️ Yasal Uyarı
Bu yazılım **yüksek riskli** bir finansal araçtır. Yalnızca eğitim amaçlıdır. Kullanımdan doğacak mali sonuçlardan kullanıcı sorumludur.

---
⭐ **Aylmz0/TradeSeeker** - *The Art of Algorithmic Mastery*
