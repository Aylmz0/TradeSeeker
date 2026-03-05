# 🚀 TradeSeeker: High-Performance AI Trading Maestro

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Binance](https://img.shields.io/badge/Binance-Futures-F0B90B?style=for-the-badge&logo=binance)
![DeepSeek](https://img.shields.io/badge/AI-DeepSeek--V3-black?style=for-the-badge&logo=deepseek)
![Status](https://img.shields.io/badge/Optimization-V9_Hardened-success?style=for-the-badge)

**TradeSeeker**, DeepSeek-V3'ün muhakeme gücünü, **$O(1)$ karmaşıklıkta** optimize edilmiş matematiksel motor ile birleştiren, yüksek performanslı bir algoritmik ticaret maestrosudur. Sadece veri okumaz; piyasa rejimini analiz eder, I/O darboğazlarını ortadan kaldırır ve milisaniyelik hassasiyetle portföy yönetir.

---

## 🧠 Maestro Mimarisi (Decoupled Services)
TradeSeeker, karmaşıklığı yönetmek için tamamen ayrıştırılmış (decoupled) bir servis mimarisi kullanır:

*   **`Orchestrator (main.py)`**: Sistemin kalbi; servisler arası veri akışını ve döngü yönetimini kontrol eder.
*   **`AIService`**: Prompt mühendisliği ve LLM entegrasyonu; bağlamı optimize ederek AI'a saf "sinyal" sunar.
*   **`AccountService`**: Canlı borsa etkileşimi; emir iletimi, senkronizasyon ve "Ghost Position" koruması.
*   **`PortfolioManager`**: $O(1)$ hızında durum takibi; risk yönetimi, PnL izleme ve in-memory state yönetimi.
*   **`StrategyAnalyzer`**: Piyasa yapısı ve rejim analizi; çoklu zaman dilimi (HTF/3m) verilerini sentezler.
*   **`MLService (XGBoost)`**: 2026 model XGBoost katmanı; 150+ indikatörle eğitilmiş model, AI kararlarına "Consensus" (Mutabakat) sağlar.
*   **`Vectorized Indicators`**: Saf NumPy ile optimize edilmiş teknik indikatör motoru.

---

## ⚡ Teknik Üstünlükler & Optimizasyonlar
Sıradan botların aksine TradeSeeker, düşük gecikme ve yüksek tutarlılık için modernize edilmiştir:

### 1. Hardcore Vektörizasyon ($O(N) \rightarrow O(C)$)
Iteratif Python döngüleri (OBV, Supertrend) tamamen NumPy vektörel işlemlerine dönüştürülmüştür. Bu sayede hesaplama süreleri milisaniyeler bazına indirilmiş ve matematiksel tutarlılık (Test Parity) %100 sağlanmıştır.

### 2. I/O Hardening (Dirty-Cache)
Dosya sistemi darboğazlarını aşmak için `portfolio_state.json` ve `bot_control.json` gibi kritik dosyalar üzerinde **filesystem mtime** tabanlı bir önbellek katmanı geliştirilmiştir. Disk okumaları sadece değişim anında yapılır, döngü içi okumalar $O(1)$ bellek hızındadır.

### 3. Unified Data Pipeline (UDP)
Veri "bir kez çekilir, bir kez işlenir, her yerde kullanılır." ML ve AI servisleri aynı OHLCV snapshot'ını paylaşır, redundant API çağrıları ve hesaplama yükü %60 oranında azaltılmıştır.

### 4. Smart Limit Order Entry
Piyasa emirleri (Market Order) yerine, milisaniyelik emir defteri (orderbook) analiziyle en uygun fiyata **Limit Order** girilir. 30 saniye içinde dolmayan emirler otomatik olarak en iyi fiyattan realize edilerek "slippage" (fiyat kayması) minimize edilir.

---

## ✨ Ana Özellikler

*   **🧠 Hybrid Intelligence (AI + ML)**: DeepSeek-V3'ün mantıksal derinliği ile XGBoost'un istatistiksel olasılıkları birleştirilerek hatalı sinyal oranı düşürülmüştür.
*   **🛡️ ATR-Based Dynamic Authority**: AI'ın stop-loss önerileri yerine sistem, ATR bazlı matematiksel stop-loss ve profit target sınırlarını katı olarak uygular.
*   **📊 Smart Sparkline v2.1**: Fiyat hareketlerini $O(1)$ pivot tespiti ile görselleştirerek AI'ın piyasa yapısını (HH/HL) anlamasını sağlar.
*   **🔄 Session Pooling**: API çağrılarında kalıcı `RetryManager` session'ları kullanılarak TCP/TLS handshaking yükü minimize edilmiştir.
*   **📉 Sharpe & Risk Metrics**: Portföy performansı akademik standartlarda (Sharpe Ratio, Peak PnL Erosion) izlenir.

---

## 🛠️ Kurulum & Kullanım

```bash
# Gereksinimleri yükle
pip install -r requirements.txt

# Botu çalıştır (Simülasyon veya Canlı Mod)
python src/main.py
```

*Ayarlar için `.env` dosyasını `config/config.py` kriterlerine göre yapılandırın.*

---

## ⚠️ Yasal Uyarı
Bu yazılım **yüksek riskli** bir finansal araçtır. Yalnızca eğitim amaçlıdır. Kullanımdan doğacak mali sonuçlardan kullanıcı sorumludur.

---
⭐ **Aylmz0/TradeSeeker** - *The Art of Algorithmic Mastery*
