# 🗺️ [EXECUTION ROADMAP] XGBoost + AI Hybrid Integration

Bu plan, TradeSeeker'ı "Rule-Based AI" botundan "Quantitative Machine Learning" platformuna taşıyacak olan hardcore teknik detayları içerir.

---

## 🛠️ Teknoloji Stack & Bağımlılıklar

*   **Model Core**: `xgboost` (Hızlı, gradient boosting kütüphanesi).
*   **Data Engineering**: `pandas`, `numpy` (Mevcut vektörize motorumuzla uyumlu).
*   **Preprocessing**: `scikit-learn` (StandardScaler, LabelEncoder).
*   **Storage (Database)**: `sqlite3` (Başlangıç aşaması için hafif ve hızlı).
*   **Persistence**: `joblib` (Model serileştirme).
*   **Logging/Observability**: `MLflow` (Deney takibi) veya yapılandırılmış `JSON logging` (Inference sonuçları için).

---

## 📐 Mimari Harita (Phased Roadmap)

### Faz 1: Data Ingestion & Storage (Temel)
XGBoost'un "yemek yemesi" lazım. Ham veriyi eğitim verisine dönüştürecek katmanı kuruyoruz.
*   **[NEW] `src/core/data_engine.py`**: Binance geçmiş verilerini (`klines`) çekip SQLite veritabanına kaydeden script.
*   **Labeling**: Geçmiş veriler üzerinde `t+N` zaman dilimi sonrasındaki getiriyi (Return) hesaplayıp "BUY/SELL/HOLD" etiketlerini oluşturma.
*   **Decision Feedback**: AI'ın geçmiş `long/short/wait` kararlarını ve sonucundaki kâr/zararı SQL'e işleme (Öz-eğitim verisi).

### Faz 2: Feature Matrix Construction
Mevcut indikatörlerimizi XGBoost'un anlayacağı bir matrix'e (DMatrix) çeviriyoruz.
*   **[MODIFY] [src/core/indicators.py](file:///home/yilmaz/projects/TradeSeeker/src/core/indicators.py)**: İndikatör çıktılarını tek bir `pd.DataFrame` (Feature Matrix) olarak döndürecek wrapper eklenmesi.
*   **Feature Selection**: `RSI`, `MACD_Hist`, `ATR_Ratio`, `OBV_Trend`, `Volatility_Index` gibi temel feature'ların yanı sıra, zamansal feature'lar (Hour of day, Day of week) eklenmesi.

### Faz 3: Training Pipeline (The Factory)
Modelin eğitildiği ve optimize edildiği laboratuvar katmanı.
*   **[NEW] `scripts/train_model.py`**: 
    1.  Veriyi SQL'den çek.
    2.  `StandardScaler` ile normalize et.
    3.  Hyperparameter tuning (GridSearch/Optuna).
    4.  Modeli `models/seeker_v1.xgb` olarak kaydet.

### Faz 4: ML Prediction Service (The Reflex)
Orchestrator'a bağlanacak olan çıkarım motoru.
*   **[NEW] `src/services/ml_service.py`**:
    *   **`predict(coin, timeframe)`**: İlgili coin için anlık veriyi alır, scalar'dan geçirir ve XGBoost olasılığını döner.
    *   **Failsafe**: Eğer model güven skoru (Conf. Score) %60'ın altındaysa AI'a hiç sormadan döngüyü `skip` eder.

### Faz 5: AI Hybridization (The Brain)
*   **[MODIFY] `src/services/ai_service.py`**: Prompt builder'a yeni bir section eklenir:
    ```json
    "ml_analysis": {
        "prediction": "STRONG_LONG",
        "probability": 0.84,
        "key_features": ["RSI_Oversold", "Volume_Spike"]
    }
    ```
*   **Decision Matrix**: AI artık ham mumları değil, ML'in ürettiği bu yapılandırılmış analizi yorumlar.

---

## 📈 Loglama & İzlenebilirlik

1.  **Prediction Logs (`data/ml_predictions.csv`)**:
    *   [timestamp](file:///home/yilmaz/projects/TradeSeeker/src/services/binance.py#78-80), [coin](file:///home/yilmaz/projects/TradeSeeker/src/main.py#316-319), `predicted_label`, `probability`, `actual_outcome` (İşlem kapandıktan sonra güncellenir).
2.  **Model Drift Detection**: Eğitim sırasındaki hata payı (RMSE/LogLoss) ile canlıdaki başarı oranının karşılaştırılması.

---

## ⚠️ Kritik Kararlar
> [!IMPORTANT]
> **Neden XGBoost?** Deep Learning (LSTM/RNN) kripto için çok gürültülüdür (noisy). XGBoost ise tabüler veride (indikatörler) hala dünyanın en iyisidir ve CPU üzerinde milisaniyeler içinde sonuç verir.
