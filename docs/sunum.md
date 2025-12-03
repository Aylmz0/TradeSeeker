# TradeSeeker: Hibrit Yapay Zeka Destekli Sistematik Alım-Satım Sistemi
**Teknik Detaylı Proje Sunum Dokümanı**

## 1. Projenin Amacı ve Kapsamı
Bu proje, finansal piyasalarda (Kripto Para) duygusal kararları elimine eden, veri odaklı ve otonom bir alım-satım sistemi (Trading Bot) geliştirmeyi amaçlar. Sistem, klasik algoritmik yaklaşımları (Teknik Analiz) modern **Büyük Dil Modelleri (LLM - DeepSeek)** ile birleştirerek "Hibrit Bir Zeka" oluşturur.

---

## 2. Sistem Mimarisi ve Çalışma Mantığı (Workflow)
Sistem, doğrusal olmayan ve çok katmanlı bir karar mekanizmasına sahiptir. Akış şöyledir:

### Adım 1: Veri Toplama ve İşleme (The Sensory Layer)
Sistem her döngüde (Cycle) piyasadan ham veriyi (Fiyat, Hacim) çeker ve işler.
*   **Zaman Dilimleri:** 1 Saatlik (Trend), 15 Dakikalık (Momentum), 3 Dakikalık (Giriş) veriler paralel işlenir.
*   **Özellik Mühendisliği (Feature Engineering):** Ham veriden türetilmiş veriler (RSI, EMA, ATR) hesaplanır.
*   **Gelişmiş Bağlam:** Fiyat hareketleri ASCII karakterlerine (Sparklines) dönüştürülerek AI'a "Görsel" bir özet sunulur.

### Adım 2: Bağlam Oluşturma (The Context Layer)
Toplanan tüm veriler, AI'ın anlayabileceği yapılandırılmış bir **JSON Prompt** formatına dönüştürülür.
*   *Örnek:* "RSI: 75" verisi, "RSI_Overbought" etiketiyle zenginleştirilir.
*   *Örnek:* Hacim verisi, son 20 mumun ortalamasıyla kıyaslanıp "Volume Ratio" (Hacim Oranı) olarak sunulur.

### Adım 3: Yapay Zeka Analizi (The Reasoning Layer)
Hazırlanan veri paketi **DeepSeek AI** modeline gönderilir. AI, kendisine verilen "Fon Yöneticisi" kişiliğiyle veriyi yorumlar:
1.  **Trend Analizi:** Fiyat EMA'nın üzerinde mi? (Yön Tayini)
2.  **Risk Analizi:** RSI çok mu şişik? Piyasa testere (Choppy) modunda mı?
3.  **Karar:** Al (Buy), Sat (Sell) veya Bekle (Hold).

### Adım 4: Güvenlik ve Filtreleme (The Safety Layer)
AI "Al" dese bile, sistemin Python tarafındaki **"Katı Kurallar" (Hard Rules)** devreye girer. Bu, AI'ın halüsinasyon görmesini veya riskli işlem yapmasını engeller.
*   **Hacim Filtresi:** Hacim ortalamanın altındaysa işlem reddedilir.
*   **Slot Kontrolü:** Maksimum 5 işlem limiti doluysa yeni işlem açılmaz.
*   **Soğuma (Cooldown):** Bir coin üzerinde işlem yapıldıysa, belirli bir süre o coin'e tekrar girilmez.

### Adım 5: İcra ve Yönetim (The Execution Layer)
Tüm filtreleri geçen kararlar uygulanır.
*   **Dinamik TP/SL:** Kar Al (Take Profit) ve Zarar Durdur (Stop Loss) seviyeleri sabit değil, piyasanın o anki oynaklığına (ATR) göre dinamik hesaplanır.

---

## 3. Teknik İndikatörler ve Matematiksel Hesaplamalar
Sistem, karar verirken aşağıdaki matematiksel modelleri kullanır. Tüm hesaplamalar `pandas` kütüphanesi ile vektörel olarak yapılır.

### 3.1. Exponential Moving Average (EMA) - Üstel Hareketli Ortalama
Fiyatın yönünü (Trend) belirlemek için kullanılır. Son verilere daha fazla ağırlık verir.
*   **Formül:** `EMA_t = (P_t * K) + (EMA_{t-1} * (1 - K))`
    *   `P_t`: Bugünkü Fiyat
    *   `N`: Periyot (Sistemde 20 ve 50 kullanılır)
    *   `K`: Ağırlık Faktörü = `2 / (N + 1)`
*   **Kullanım:** Fiyat > EMA20 ise "Yükseliş Trendi", Fiyat < EMA20 ise "Düşüş Trendi".

### 3.2. Relative Strength Index (RSI) - Göreceli Güç Endeksi
Fiyatın değişim hızını ölçerek aşırı alım/satım bölgelerini tespit eder.
*   **Formül:** `RSI = 100 - (100 / (1 + RS))`
    *   `RS = Ortalama Kazanç / Ortalama Kayıp`
*   **Hesaplama:** Son 14 periyottaki pozitif kapanışların ortalaması ile negatif kapanışların ortalaması oranlanır.
*   **Kullanım:** RSI > 70 (Aşırı Alım - Satış ihtimali), RSI < 30 (Aşırı Satım - Alış ihtimali).

### 3.3. Average True Range (ATR) - Ortalama Gerçek Aralık
Piyasanın volatilitesini (oynaklığını) ölçer. Yön belirtmez, sadece hareketin büyüklüğünü gösterir.
*   **Formül:**
    1.  `TR = Max(|High - Low|, |High - Close_prev|, |Low - Close_prev|)`
    2.  `ATR = SMA(TR, 14)` (TR değerlerinin 14 günlük ortalaması)
*   **Kullanım:** Stop Loss seviyesini belirlerken kullanılır.
    *   *Örnek:* `Stop Loss = Giriş Fiyatı - (2 * ATR)`
    *   Bu sayede oynak piyasada stop mesafesi genişler, sakin piyasada daralır.

### 3.4. Kaufman Efficiency Ratio (ER) - Etkinlik Oranı
Piyasanın "Trend" mi yoksa "Testere" (Yatay/Choppy) mi olduğunu anlamak için kullanılır.
*   **Formül:** `ER = Net Değişim / Toplam Oynaklık`
    *   `ER = |Fiyat_t - Fiyat_{t-n}| / Toplam(|Fiyat_i - Fiyat_{i-1}|)`
*   **Mantık:** Fiyat dümdüz bir çizgide giderse ER=1 olur. Sürekli zikzak çizip aynı yere gelirse ER=0'a yaklaşır.
*   **Kullanım:** ER < 0.40 ise "Choppy Market" kabul edilir ve işlem yapılması engellenir.

### 3.5. Sparklines (ASCII Grafikleri)
Fiyat serisini metin tabanlı bir grafiğe dönüştürür.
*   **Algoritma:**
    1.  Son 24 mumun fiyatları alınır.
    2.  Min ve Max değerler bulunur.
    3.  Veri 0-7 aralığına normalize edilir (8 seviyeli karakter seti: ` ▂▃▄▅▆▇█`).
    4.  Her fiyat, karşılık gelen karaktere dönüştürülür.
*   **Örnek:** ` ▂▃▄▅▆▇█` (Güçlü Yükseliş), `█▇▆▅▄▃▂ ` (Güçlü Düşüş).

---

## 4. Kod Mimarisi ve Dosya Yapısı
Proje, modüler ve sürdürülebilir bir yazılım mimarisine sahiptir:

### `src/main.py` (Orkestra Şefi)
*   Sistemin ana döngüsünü (Infinite Loop) yönetir.
*   Tüm alt modülleri sırasıyla çağırır ve senkronize eder.

### `src/core/market_data.py` (Duyu Organları)
*   Borsa API'sinden ham veriyi çeker.
*   Tüm teknik indikatörleri (RSI, EMA, Sparklines) hesaplar.
*   Veriyi temizler ve işler.

### `src/core/portfolio_manager.py` (Cüzdan ve Risk Müdürü)
*   Mevcut bakiyeyi ve açık pozisyonları takip eder.
*   Risk hesaplamalarını (Pozisyon büyüklüğü, TP/SL) yapar.
*   İşlemlerin kaydını tutar.

### `src/ai/deepseek_api.py` (Beyin)
*   DeepSeek LLM ile iletişimi sağlar.
*   "System Prompt" (AI'ın kişiliği ve kuralları) burada tanımlıdır.

### `src/ai/prompt_json_builders.py` (Tercüman)
*   Python objelerini (DataFrame, Dict), AI'ın anlayacağı optimize edilmiş JSON formatına çevirir.
*   Veri tasarrufu (Token optimization) burada yapılır.

---

## 5. Projenin Yenilikçi Yönleri (Innovation)
1.  **Hibrit Zeka:** Klasik botlar sadece sayıya bakar (RSI < 30). Bu sistem ise AI sayesinde bağlamı (Context) anlar. *"RSI düşük ama trend çok güçlü düşüyor, alma"* diyebilir.
2.  **Görsel Patern Tanıma (Metin Tabanlı):** Fiyat grafiklerini ASCII karakterlerine (` ▂▃▄▅`) dönüştürerek, metin tabanlı bir modele "Görsel" analiz yeteneği kazandırılmıştır.
3.  **Dinamik Adaptasyon:** Piyasa durgunken hedefleri küçültür, hareketliyken büyütür (ATR Adaptasyonu).