# 📚 TradeSeeker: Mimari ve Dosya Rehberi

Bu dosya, TradeSeeker projesinin teknik mimarisini, dosya yapısını ve veri akışını anlamak için hazırlanmış kapsamlı bir kılavuzdur.

---

## 🏗️ Genel Mimari (Decentralized Service Architecture)

TradeSeeker, **"Decoupled Architecture"** (Ayrıştırılmış Mimari) prensibiyle tasarlanmıştır. Her servis kendi sorumluluk alanına sahiptir ve birbirleriyle `main.py` (Orchestrator) üzerinden haberleşir.

### Veri Akış Modeli: Unified Data Pipeline (UDP)
Sistemde veri "bir kez çekilir, bir kez işlenir, her yerde kullanılır." 
1. `RealMarketData` (veya `DataEngine`) 1h, 15m, ve 3m verilerini çeker, SQLite ambarına yazar ve cache'ler.
2. `MLService` ve `AIService` aynı bellek snapshot'ını kullanarak analiz yapar.
3. Bu sayede API gecikmesi ve hesaplama tutarsızlıkları önlenir.

### Zaman Dilimi Hiyerarşisi (Timeframe Hierarchy)
Sistemin beyni, karar alırken farklı veri katmanlarını şu şekilde derecelendirir:
1. **1h Katmanı (The Boss):** Hakim rejim (BULLISH/BEARISH/NEUTRAL) ve fiyatın nerede konumlandığını belirler. Trend takibi veya karşıt trend stratejisinin anahtarıdır.
2. **15m Katmanı (The Advisor):** Mikro piyasa yapısını (HH_HL, LH_LL, RANGE) ve momentum gücünü ölçer. Giriş için onay merciidir.
3. **3m Katmanı (The Sensor):** İşlem hacmi ve ani RSI tepkilerini (Liquidity & Micro-Reaction) ölçer. Bir giriş sinyalini iptal edemez, ancak risk seviyesini (`POOR/LOW` volume) işaretler.

### Yapay Zeka Entropi Kontrolü: Decoupled Analysis Mimarisi
Model çıktılarını stabil tutmak ve LLM "token limit" (truncation) hatalarını (özellikle `Unterminated string`) önlemek için ayrıştırılmış mimari kullanılır.
- **Reasoning:** Yapay Zeka teknik analizi kendi iç `<thought>` / `[AI] REASONING` bloğunda uzun uzun yapar. Bu metin, insan okuması için idealdir.
- **Decisions:** Çıktının sonunda ise yalnızca kararları barındıran katı ve minik bir JSON bloğu verilir (`DECISIONS: { ... }`).
- **Robust JSON Repair:** Gelen JSON'daki noktalama unutulmaları (eksik virgül, kapanmamış string vb.), agresif Regex yakalayıcılarla donatılmış `deepseek_api.py` tarafından anında tamir edilir.

## 📁 Dosya ve Klasör Yapısı

### 1. `src/` (Kaynak Kod)
Projenin beyni buradadır.

- **`main.py`**: Sistemin **Orchestrator**'ı. Döngü yönetimini, servis başlatmalarını ve botun hayat döngüsünü (Lifecycle) yönetir.

#### `src/core/` (Çekirdek Servisler)
- **`ai_service.py`**: Karar mekanizması. Prompt hazırlığı, LLM iletişimi ve ML konsensüs birleştirmesini yönetir.
- **`account_service.py`**: Borsa etkileşimi. Emirlerin iletilmesi (`place_smart_limit_order`), "Ghost Position" kontrolü ve cüzdan senkronizasyonu.
- **`portfolio_manager.py`**: Risk ve Durum Yönetimi. PnL hesaplama, state takibi ve $O(1)$ hızında `Dirty-Cache` yönetimi. `utils.py` üzerinden Thread-Safe (yarış-durumu korumalı) dosya yazma işlemlerini koordine eder.
- **`market_data.py`**: Veri Motoru. Mum verisi çekme, UDP (Unified Data Pipeline) cache yönetimi ve hata ayıklama.
- **`indicators.py`**: **Vektörize Matematik**. NumPy ile optimize edilmiş tüm teknik indikatörlerin (RSI, EMA, Supertrend vb.) hesaplandığı yer.
- **`data_engine.py`**: Veri Ambarı. SQLite işlemleri, eğitim verisi toplama (`features` table) ve ML özellik hazırlığı.
- **`cache_manager.py`**: Performans Katmanı. API yanıtları için akıllı TTL (Time-to-Live) cache sistemi.
- **`performance_monitor.py`**: Analitik. Geçmiş işlemlerin başarısını ölçen istatistiksel raporlama motoru.

#### `src/services/` (Dış Servisler)
- **`binance.py`**: Binance API Wrapper. Smart Limit Order mantığı ve borsa protokolleri burada tanımlıdır.
- **`ml_service.py`**: XGBoost Çıkarım (Inference) Motoru. Eğitilmiş modelin yüklenmesi ve canlı tahmin üretilmesi.

#### `src/ai/`
- **`prompt_json_builders.py`**: AI'a gönderilen JSON paketlerini yapılandıran modül. İş mantığı (Business Logic) burada enkapsüle edilmiştir.

#### `src/web/` (Dashboard)
- **`admin_server_flask.py`**: Dashboard'un backend API'sı.
- **`templates/index.html`**: Kullanıcı arayüzü (UI).
- **`static/js/app.js`**: Arayüz dinamikleri ve gerçek zamanlı grafik güncellemeleri.

### 2. `scripts/` (Yardımcı Araçlar)
- **`train_model.py`**: SQLite verilerini kullanarak XGBoost modelini eğiten script.
- **`drift_check.py`**: Modelin başarısını gerçek veriyle kıyaslayıp "sapma" (drift) tespiti yapan araç.

### 3. `tests/` (Doğrulama)
- **`test_vectorization.py`**: Matematiksel işlemlerin test parity doğrulaması.
- **`test_hybrid.py`**: AI ve ML entegrasyonunun uçtan uca testi.

---

## 🔄 Bir İşlem Döngüsü Nasıl Çalışır?

1.  **Market Fetch**: `RealMarketData` 3m, 15m ve 1h verilerini SQLite ambarına kaydeder.
2.  **UDP Cache**: Çekilen veriler işlemlerde o tur boyunca bellekten okunur.
3.  **ML Consensus**: `MLService`, 15m verisini cache'den alır ve Global XGBoost modeli üzerinden olasılıkları (Buy/Sell/Hold) üretir.
4.  **AI Analysis**: `AIService` piyasa verisi + ML tahmini ile kapsamlı bir prompt oluşturur. API İsteği gönderir (OpenRouter/Groq/Z.AI).
5.  **Reasoning & Parsing**: AI teknik detayları yazar, JSON yakalayıcı bunu Regex mekanizmalarıyla temizce çekip ayrıştırır.
6.  **Risk Check**: AI "GİR" derse, `PortfolioManager` kasa limitlerini ve hedef yöndeki riski kontrol eder.
7.  **Smart Entry**: `AccountService`, emir defterini analiz eder ve **Limit Order** girer.
8.  **Thread-Safe Logging**: İşlem SQLite ve JSON dosyalarına, `threading.Lock` korumasında çarpışma olmadan (`Errno 2` güvenliği) kaydedilir.

---

## 🛠️ Konfigürasyon ve Güvenlik

- **`.env`**: API Key'ler ve botun çalışma parametreleri (leverage, risk pct vb.).
- **`pyproject.toml`**: Ruff linter ayarları ve proje metadata'sı.
- **`.gitignore`**: Gizli anahtarların ve büyük veri dosyalarının repoya girmesini engeller.

---
⭐ **TradeSeeker** - *The Art of Algorithmic Mastery*
