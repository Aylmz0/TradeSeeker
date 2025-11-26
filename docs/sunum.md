# Alpha Arena: Yapay Zeka Destekli Sistematik Alım-Satım Sistemi

Bu doküman, Alpha Arena projesinin teknik mimarisini, çalışma mantığını ve temel bileşenlerini akademik bir sunum formatında detaylandırmaktadır.

## 1. Proje Özeti ve Amacı

**Alpha Arena**, kripto para piyasalarında otonom işlem yapabilen, hibrit bir yapay zeka mimarisine sahip sistematik bir alım-satım botudur. Sistem, klasik teknik analiz yöntemlerini Büyük Dil Modellerinin (LLM - DeepSeek) muhakeme yeteneği ile birleştirerek piyasa verilerini yorumlar ve işlem kararları alır.

**Temel Amaç:**
*   Duygusal kararları elimine etmek.
*   Çoklu zaman dilimlerini (Multi-Timeframe) aynı anda analiz etmek.
*   Karmaşık piyasa koşullarında (Trend ve Yatay piyasa) dinamik stratejiler uygulamak.

---

## 2. Sistem Mimarisi

Sistem dört ana katmandan oluşmaktadır:

### 2.1. Veri Toplama Katmanı (`binance.py`)
*   **Görevi:** Binance Borsası'ndan gerçek zamanlı fiyat (OHLCV) verilerini çeker.
*   **Zaman Dilimleri:** 3 dakikalık (kısa vade), 15 dakikalık (orta vade) ve 1 saatlik (ana trend) veriler toplanır.
*   **Ek Veriler:** Hacim, Emir Defteri Derinliği (Order Book), Fonlama Oranları (Funding Rates).

### 2.2. Analiz Motoru (`alpha_arena_deepseek.py` & `utils.py`)
*   **Görevi:** Ham veriyi işleyerek teknik indikatörleri hesaplar.
*   **Kullanılan İndikatörler:** EMA, RSI, MACD, ATR (Detayları Bölüm 3'te açıklanmıştır).

### 2.3. Karar Mekanizması (AI & Hibrit Promptlama)
Sistemin beyni burasıdır. Klasik algoritmik botlardan farklı olarak, kararlar "if-else" blokları yerine bir LLM (DeepSeek) tarafından verilir.

*   **Hibrit Prompt Yapısı:**
    *   **Sistem Promptu (Statik):** AI'ın kişiliğini (Risk Manager), kuralları ve strateji sınırlarını belirler.
    *   **Kullanıcı Promptu (Dinamik JSON):** Piyasa verileri, hesap durumu ve indikatör analizleri **JSON formatında** yapılandırılarak AI'a sunulur. Bu, AI'ın veriyi sayısal olarak kesin bir şekilde işlemesini sağlar.

### 2.4. İşlem Yönetimi (`performance_monitor.py`)
*   **Görevi:** AI'dan gelen "AL/SAT" sinyallerini borsaya iletir, pozisyon büyüklüğünü ayarlar ve risk kontrollerini yapar.

---

## 3. Temel Teknik İndikatörler ve Metrikler

Sistemin karar verirken kullandığı temel finansal araçların ve metriklerin detaylı açıklamaları şunlardır:

### 3.1. RSI (Göreceli Güç Endeksi - Relative Strength Index)
*   **Nedir?** Fiyat hareketlerinin hızını ve değişimini ölçen, 0 ile 100 arasında değer alan bir momentum osilatörüdür.
*   **Genel Kullanım:** 70 seviyesinin üzeri "Aşırı Alım" (Fiyatın düşme ihtimali yüksek), 30 seviyesinin altı "Aşırı Satım" (Fiyatın yükselme ihtimali yüksek) olarak yorumlanır.
*   **Sistemdeki Rolü:** Özellikle **Counter-Trade (Karşıt İşlem)** stratejisinde kritik rol oynar. Fiyat yükselirken RSI 70'i aşarsa (veya düşerken 30'un altına inerse), sistem bunu bir "yorulma" veya "dönüş sinyali" olarak algılar ve ters yönlü işlem fırsatı arar.

### 3.2. MACD (Hareketli Ortalama Yakınsama Iraksama)
*   **Nedir?** İki farklı hareketli ortalamanın (genelde 12 ve 26 periyotluk) ilişkisini gösteren bir trend takip ve momentum indikatörüdür.
*   **Genel Kullanım:** Sinyal çizgisi kesişimleri ve "Uyumsuzluklar" (Divergence) aranır.
*   **Sistemdeki Rolü:** **Trend Reversal (Trend Dönüşü)** tespiti için kullanılır. Örneğin, fiyat yeni bir tepe yaparken MACD yeni bir tepe yapamıyorsa (Negatif Uyumsuzluk), bu trendin zayıfladığını ve yakında dönebileceğini gösterir.

### 3.3. Hacim (Volume) ve Hacim Kalitesi
*   **Nedir?** Belirli bir zaman diliminde alınıp satılan varlık miktarıdır. Piyasaya giren paranın gücünü gösterir.
*   **Genel Kullanım:** Fiyat hareketlerini doğrulamak için kullanılır. Hacimsiz (zayıf) yükselişler genellikle güvenilmezdir ve "tuzak" olabilir.
*   **Sistemdeki Rolü:** Sistem, anlık hacmi geçmiş ortalama hacimle kıyaslar (`volume_ratio`). Eğer hacim ortalamanın 1.5-2 katına çıkmışsa, bu "Güçlü Sinyal" olarak kabul edilir ve AI'ın işleme giriş güvenini (confidence) artırır.

### 3.4. EMA (Üstel Hareketli Ortalama - Exponential Moving Average)
*   **Nedir?** Son fiyatlara daha fazla ağırlık vererek hesaplanan bir ortalama türüdür. Basit ortalamaya göre fiyat değişimlerine daha hızlı tepki verir.
*   **Sistemdeki Rolü:**
    *   **Trend Yönü Belirleme:** Fiyat EMA20'nin (20 periyotluk ortalama) üzerindeyse trend **BULLISH** (Yükseliş), altındaysa **BEARISH** (Düşüş) kabul edilir.
    *   **Dinamik Destek/Direnç:** Fiyatın EMA20'ye geri çekilmeleri (Pullback), trend yönünde işleme giriş fırsatı olarak değerlendirilir.

### 3.5. ATR (Ortalama Gerçek Aralık - Average True Range)
*   **Nedir?** Piyasdaki volatiliteyi (oynaklığı) ölçen bir metriktir. Yönü göstermez, hareketin ortalama boyutunu gösterir.
*   **Sistemdeki Rolü:** **Risk Yönetimi** için hayati önem taşır. Sistem, Stop-Loss seviyelerini sabit bir yüzde (örn. %1) yerine ATR'ye göre belirler (örn. 2xATR). Böylece volatil (hareketli) piyasada stoplar genişler, sakin piyasada daralır; bu da "gürültü" yüzünden gereksiz yere stop olmayı engeller.

---

## 4. Temel Stratejiler ve Mantık

Sistem iki ana strateji üzerine kuruludur:

### 4.1. Trend Takibi (Trend Following)
*   **Mantık:** "Trend senin dostundur."
*   **Koşul:** 1 Saatlik (Ana) trend ile 15m ve 3m (Ara) trendlerin aynı yönde olması.
*   **Örnek:** 1H BULLISH + 15m BULLISH + 3m BULLISH -> **LONG İşlem**.

### 4.2. Karşıt İşlem (Counter-Trade / Mean Reversion)
*   **Mantık:** Fiyatın ana trende ters yönde yaptığı kısa vadeli düzeltmeleri yakalamak.
*   **Tanım:** 15m ve 3m momentumunun, 1h yapısal trendine **karşı** hizalanması.
*   **Kritik Hata Düzeltmesi:** Sistemde daha önce yapılan bir düzeltme ile, sadece 3m değil, **hem 15m hem de 3m** trendinin ana trende ters olması şartı getirilmiştir.
*   **Örnek:** 1H BULLISH (Ana Trend Yukarı) ancak 15m BEARISH ve 3m BEARISH (Kısa vadeli düşüş) -> **Counter-Trend SHORT Fırsatı**.

---

## 5. Kritik Değişkenler ve Parametreler

Profesörünüze anlatırken vurgulamanız gereken en önemli sistem değişkenleri:

### `market_regime` (Piyasa Rejimi)
*   **Nedir?** Piyasaların genel durumunu özetleyen değişken.
*   **Değerler:** `BULLISH` (Yükseliş), `BEARISH` (Düşüş), `NEUTRAL` (Yatay/Kararsız).
*   **Önemi:** AI, stratejisini rejime göre değiştirir. Örneğin `NEUTRAL` rejimde daha dar Stop-Loss kullanır.

### `alignment_strength` (Trend Uyumu)
*   **Nedir?** Farklı zaman dilimlerindeki trendlerin birbirine ne kadar uyumlu olduğunu gösterir.
*   **Hesaplama:** 1H, 15m ve 3m trendlerinin yön birliğine bakılır.
*   **Önemi:** `STRONG` uyum, yüksek güvenli işlem demektir.

### `risk_level` (Risk Seviyesi)
*   **Nedir?** Bir işlemin ne kadar riskli olduğunun sayısal veya kategorik ifadesi.
*   **Faktörler:** Volatilite, RSI'ın aşırı bölgelerde olması, trende ters olma durumu.
*   **Değerler:** `LOW_RISK`, `MEDIUM_RISK`, `HIGH_RISK`.

### `confidence` (Güven Skoru)
*   **Nedir?** AI'ın verdiği karara ne kadar güvendiğini belirten 0.0 ile 1.0 arası bir sayı.
*   **Kullanımı:** Pozisyon büyüklüğünü (Position Sizing) belirler. Yüksek güven = Daha büyük pozisyon.

### `conditions_met` (Karşılanan Koşullar)
*   **Nedir?** Bir karşıt işlem (Counter-Trade) sinyali için gerekli 5 teknik şarttan kaçının sağlandığı.
*   **Şartlar:**
    1.  Trend Uyumsuzluğu (1H vs 15m+3m)
    2.  Hacim Onayı
    3.  RSI Aşırılığı
    4.  EMA'dan Uzaklık
    5.  MACD Uyumsuzluğu

---

## 6. Neden Hibrit Promptlama? (Teknik Yenilik)

Sistemimiz, **"Chain of Thought" (Düşünce Zinciri)** yöntemini yapılandırılmış **JSON verisi** ile birleştirir.

*   **Sorun:** LLM'ler düz metin içindeki sayısal verileri (örn. "RSI 75 oldu") bazen gözden kaçırabilir veya yanlış yorumlayabilir.
*   **Çözüm:** Veriler `COUNTER_TRADE_ANALYSIS` gibi JSON blokları halinde verilir.
*   **Avantaj:** AI, JSON verisini bir veritabanı gibi okur, kesin kuralları uygular ve ardından kararını metin olarak açıklar. Bu, "Halüsinasyon" (Yapay Zeka Yanılsaması) riskini minimize eder.

---

## 7. Sonuç

Alpha Arena, sadece teknik indikatörlere bakan kör bir algoritma değil, piyasa bağlamını (Context) anlayan ve risk yönetimi yapabilen akıllı bir asistandır. Çok katmanlı mimarisi ve hibrit veri işleme yeteneği ile akademik ve pratik açıdan ileri düzey bir finansal teknoloji örneğidir.
