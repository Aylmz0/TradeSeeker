# 🌟 TradeSeeker: Master Architecture & "North Star" Plan

Bu belge, TradeSeeker projesinin "Kutup Yıldızı"dır. Proje ne kadar büyürse büyüsün, mimari bütünlüğü korumak ve stratejik rotadan sapmamak için bu plana sadık kalınacaktır.

---

## 🏗️ 1. Temel Felsefe: "The Hybrid Maestro"
TradeSeeker, **Kantitatif Matematik (ML)** ile **Stratejik Muhakeme (LLM)** arasındaki boşluğu kapatan hibrit bir sistemdir.
*   **XGBoost**: Refleksler ve Olasılıklar (Sayılardan Duygu Çıkarma).
*   **DeepSeek (AI)**: Analiz ve Karar (Duygulardan Strateji Üretme).
*   **Core Engine**: Disiplin ve Uygulama (Stratejiyi Emire Dönüştürme).

---

## 🗺️ 2. Mimarinin Katmanları (The Blueprint)

### A. Data Layer (Veri Katmanı) - *Foundation*
*   **Current**: REST API (Polling), `JSON` state files.
*   **North Star**: 
    - **Websocket Stream**: Anlık fiyat ve emir defteri (orderbook) verisi.
    - **Persistence DB (SQLite/PostgreSQL)**: İşlem geçmişi ve KLine verilerinin SQL tabanlı saklanması (İstendiği an eğitim verisine dönüşebilen yapı).
    - **Dirty-Cache v2**: Disk I/O'yu sıfıra indiren ve sadece event-driven tetiklenen bellek yönetimi.

### B. Logic Layer (Hesaplama Katmanı) - *The Muscle*
*   **Vector Engine**: NumPy tabanlı $O(1)$ indikatör motoru.
*   **ML Predictor (XGBoost)**: 
    - **Feature Engineering**: İndikatörleri ham veri olarak değil, normalize edilmiş (StandardScaled) girdi olarak kullanır.
    - **Probability Inference**: Her coin için "Bias" olasılığı üretir. (Örn: %85 Long).
*   **Feedback Loop (Self-Learning)**:
    - **Decision Tagging**: AI'ın verdiği kararlar ve ML'in tahminleri veritabanına işlenir.
    - **Outcome Mapping**: Pozisyon kapandığında elde edilen gerçek PnL, geçmiş tahminlerle eşleştirilir. 
    - **Retraining**: Model, sadece piyasayı değil, kendi "isabet oranını" da öğrenerek hatalarını minimize eder.

### C. Intelligence Layer (Zeka Katmanı) - *The Brain*
*   **Contextual AI (DeepSeek-V3)**: 
    - AI artık ham mumları görmez. ML'in ürettiği olasılıkları, volatility metriklerini ve piyasa rejimini (Bull/Bear) yorumlayan bir "Fon Yöneticisi" gibi davranır.
    - **Token Stewardship**: Promptlar yapılandırılmış JSON özetlerine (Summary) dayandırılarak maliyet minimize edilir.

### D. Execution Layer (Uygulama Katmanı) - *The Hand*
*   **Order Executor**: 
    - **API Consistency Buffer**: Ghost Position engelleme sisteminin standartlaşması.
    - **North Star Transition**: Piyasa emrinden (Market Order), fiyat kaymasını (Slippage) engelleyen akıllı Limit Order yapılarına geçiş.
*   **ATR-Authority**: Risk yönetimi AI'nın inisiyatifinde değildir; sistemin matematiksel "Katı Kuralıdır".

---

## 🛤️ 3. Stratejik Yol Haritası (Milestones)

### Phase 1: Mimari Temizlik (TAMAMLANDI)
- Modüler yapıya geçiş ([indicators.py](file:///home/yilmaz/projects/TradeSeeker/src/core/indicators.py), [ai_service.py](file:///home/yilmaz/projects/TradeSeeker/src/core/ai_service.py), [account_service.py](file:///home/yilmaz/projects/TradeSeeker/src/core/account_service.py)).
- Devasa sınıfların ayrıştırılması.

### Phase 2: Teknik Sağlamlaştırma (TAMAMLANDI)
- Vektörizasyon ($O(1)$).
- I/O Darboğazlarının giderilmesi (Dirty-Cache).
- Binance API stabilizasyonu.

### Phase 3: ML Hybridization & Global Training (AKTİF) - *The Current Focus*
1.  **SQLite Ingestion**: Geçmiş verinin eğitim için toplanması ve karlı/zararlı işlemlerin etiketlenmesi.
2.  **Global XGBoost Training**: Sistemdeki *tüm* aktif coinlerin birleştirilmiş verileri üzerinden "Global" bir piyasa modeli eğitilmesi (Tek model, çoklu coin).
3.  **Hybrid Orchestration**: AI ve ML kararlarının birleştirilmesi. ML verisi yoksa sistemin güvenli bir şekilde (Fallback) sadece teknik analize (AI-Only) dönmesi.

### Phase 4: Self-Learning (Kendi Kendine Öğrenme) & Scalability (GELECEK)
- **Outcome Mapping (Self-Learning)**: Botun `decisions` tablosundaki kendi geçmiş kararlarını ve reel PnL (Kâr/Zarar) sonuçlarını analiz ederek XGBoost/AI kararlarını cezalandırıp/ödüllendirmesi (Reinforcement Learning).
- **Websocket Entegrasyonu**: Anlık veri akışı ile tick-level analiz.
- **Smart Limit Order Entry**: Slippage (Kayma) optimizasyonu için derin orderbook analizi.
- **DashBoard**: Performansın görsel olarak izlenebileceği bir Web UI.

---

## 🛡️ 4. Değişmez Kurallar (Guardrails)
1.  **Kural 1**: Matematik AI'ı veto edebilir, ancak AI matematiğin stop-loss kurallarını asla veto edemez.
2.  **Kural 2**: Sisteme hiçbir "iterative loop" (yavaş döngü) matematiksel hesaplama eklenemez. Her şey vektörel olmalıdır.
3.  **Kural 3**: Hiçbir API çağrısı Retry-Mechanism (Oturum Havuzu) olmadan yapılamaz.
4.  **Kural 4**: Kod tabanında asla Türkçe yorum veya ASCII dışı karakter barındırılamaz (Temizlik Standardı).

---

## 🖥️ 5. Kaynak Yönetimi (Resource Stewardship)
Sistem, kısıtlı donanımlarda (VDS 3GB RAM / Home Server) 7/24 çalışacak şekilde optimize edilecektir:
- **Lightweight Inference**: Canlı döngüde sadece `.predict()` çalışır (Gerekli bellek < 50MB).
- **Off-Peak Training**: Model eğitimi sadece manuel tetiklendiğinde veya hafta sonu düşük piyasa hacminde çalışır.
- **SQLite Optimization**: Veriler RAM'de değil, diskte (SQLite) tutulur; sadece ihtiyaç anında (batch) çekilir.
- **CPU Throttling**: Eğitim işlemi sırasında sistemin ana döngüsünü (Trading Loop) kesmemesi için düşük "process priority" kullanılır.

---
> [!NOTE]
> Bu plan, TradeSeeker'ın bir hobiden bir finansal enstrümana dönüşüm dökümanıdır. "North Star" budur, rota şaşmaz.
