# Alpha Arena DeepSeek Trading Bot

Profesyonel kripto para ticaret botu - DeepSeek AI entegrasyonu ile otomatik trading sistemi.

## 🚀 Özellikler

- **AI-Powered Trading**: DeepSeek API ile akıllı ticaret kararları
- **Multi-Asset Support**: XRP, DOGE, ASTR, ADA, LINK, SOL
- **Advanced Risk Management**: Dinamik risk yönetimi ve pozisyon boyutlandırma
- **Configurable Trend Detection**: Varsayılan 1h EMA20 trend analizi (HTF_INTERVAL ile değiştirilebilir)
- **Auto TP/SL**: Otomatik kar al ve stop-loss yönetimi
- **Real-time Data**: Binance API ile gerçek zamanlı piyasa verileri
- **Web Dashboard**: Gerçek zamanlı izleme ve kontrol paneli
- **Flexible Risk Levels**: Low, Medium, High risk seviyeleri
- **JSON Prompt Format** (v1.0): Yapılandırılmış JSON format ile daha iyi AI parsing ve token optimizasyonu

## 📋 Sistem Gereksinimleri

- Python 3.8+
- DeepSeek API Key
- Binance API Keys (Opsiyonel - gelişmiş özellikler için)
- codeserver veya benzeri geliştirme ortamı

## ⚙️ Hızlı Kurulum

### 1. Gereksinimleri Yükleme

```bash
# Python ve gerekli paketleri yükle
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv tmux

# Sanal ortam oluştur
python3 -m venv .venv
source .venv/bin/activate

# Gerekli kütüphaneleri yükle
pip install -r requirements.txt
```

### 2. API Anahtarlarını Ayarlama

`.env` dosyasını düzenleyin:

```bash
# DeepSeek API Configuration
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx...

# Binance API Configuration (Opsiyonel)
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key
TRADING_MODE=simulation   # simulation | live
BINANCE_TESTNET=true      # live modunda testnet için true
BINANCE_MARGIN_TYPE=ISOLATED
BINANCE_DEFAULT_LEVERAGE=10
BINANCE_RECV_WINDOW=5000

# Trading Configuration
INITIAL_BALANCE=200.0
MAX_TRADE_NOTIONAL_USD=150.0
CYCLE_INTERVAL_MINUTES=2
MAX_LEVERAGE=15
MIN_CONFIDENCE=0.4
MAX_POSITIONS=4

# Exit Plan Defaults
DEFAULT_STOP_LOSS_PCT=0.01
DEFAULT_PROFIT_TARGET_PCT=0.015
MIN_EXIT_PLAN_OFFSET=0.0001

# Trailing Stop Controls
TRAILING_PROGRESS_TRIGGER=80.0
TRAILING_TIME_PROGRESS_FLOOR=50.0
TRAILING_TIME_MINUTES=20
TRAILING_ATR_MULTIPLIER=1.2
TRAILING_FALLBACK_BUFFER_PCT=0.004
TRAILING_VOLUME_ABSOLUTE_THRESHOLD=0.2
TRAILING_VOLUME_DROP_RATIO=0.5
TRAILING_MIN_IMPROVEMENT_PCT=0.0005

# Trend Detection
HTF_INTERVAL=1h  # Options: 30m, 1h, 2h, 4h

# Risk Level Configuration
RISK_LEVEL=medium  # Options: low, medium, high

# JSON Prompt Feature Flags (New in v1.0)
USE_JSON_PROMPT=false  # Enable JSON format prompts (true/false)
JSON_PROMPT_COMPACT=true  # Use compact JSON format to save tokens (true/false) - Recommended: true
VALIDATE_JSON_PROMPTS=false  # Enable runtime JSON validation (true/false)
JSON_PROMPT_VERSION=1.0  # JSON prompt format version
JSON_SERIES_MAX_LENGTH=50  # Maximum series length before compression (default: 50)
JSON_CACHE_ENABLED=false  # Enable JSON serialization cache (true/false)
JSON_CACHE_TTL=120  # Cache TTL in seconds (2 minutes = 1 cycle)
```

### 2.1 Canlı Kullanım Konfigürasyon İpuçları

- **HISTORY_RESET_INTERVAL**: (Varsayılan `35`) Her bu kadar cycle'da geçmiş logları temizler, sistemin uzun süreli bias geliştirmesini engeller. Canlıda 30-50 arası değer önerilir.  
- **HTF_INTERVAL**: (Varsayılan `1h`) Trend tespiti için kullanılan üst zaman dilimi. Kısa vadeli stratejiler için `30m`, daha geniş perspektif için `2h` veya `4h` deneyebilirsiniz; bot otomatik olarak EMA20 referanslarını yeni interval üzerinden hesaplar.  
- **SAME_DIRECTION_LIMIT**: Maksimum aynı yönde (long/short) pozisyon slotu. Futures cüzdan boyutunuza göre azaltıp artırabilirsiniz; borsanın kaldıraç limitini aşmamasına dikkat edin.  
- **CYCLE_INTERVAL_MINUTES** & **calculate_optimal_cycle_frequency**: Varsayılan 2 dakika. Spot/USDT perpetual tarafında API sınırlarını zorlamamak için minimum 2 dk önerilir; volatilite yüksekse bot otomatik olarak 2-4 dakika aralığına geçer.  
- **MIN_CONFIDENCE**: AI karar filtre eşiği. Gerçek bakiyede çok düşük ayarlanması gereksiz işlem sayısını artırabilir; 0.4-0.5 aralığı sağlıklı.  
- **INITIAL_BALANCE / MIN_POSITION_MARGIN_USD**: Gerçek bakiyeniz farklıysa `.env` ve `config.py` değerlerini güncelleyip botu yeniden başlatın; margin limitleri yeni bakiyeye göre otomatik ölçeklenir.  
- **API Limits & Failover**: Binance tarafında saniyede 10 istek limitini aşmamak için `MAX_RETRY_ATTEMPTS`, `REQUEST_TIMEOUT` gibi parametreleri aşırı düşürmeyin.  
- **USE_JSON_PROMPT**: (Varsayılan `false`) JSON format prompt'ları aktif eder. Token tasarrufu için `JSON_PROMPT_COMPACT=true` ile birlikte kullanılması önerilir. Hata durumunda otomatik olarak text format'a fallback yapar. Compact mode ile text format'tan %26'ya kadar token tasarrufu sağlanabilir.  

### 2.2 Yeni Cihazda Kurulum Kontrol Listesi

Projeyi farklı bir makinede ilk kez ayağa kaldırırken aşağıdaki adımlar en yaygın kurulum hatalarını engeller:

1. **Sistem paketlerini hazırlayın**
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3 python3-venv python3-dev build-essential libssl-dev libffi-dev
   ```
   > macOS için `xcode-select --install` komutu ile geliştirici araçlarını kurun; Windows için [python.org](https://www.python.org/downloads/) üzerinden Python 3.10+ kurulumunda "Add Python to PATH" seçeneğini işaretleyin.

2. **Depoyu klonlayın ve dizine girin**
   ```bash
   git clone https://github.com/<kullanici>/AlphaArena.git
   cd AlphaArena
   ```

3. **Sanal ortam oluşturup etkinleştirin**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

4. **pip araçlarını güncelleyin ve bağımlılıkları yükleyin**
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```
   Eğer `pandas` veya `numpy` derleme hatası alırsanız, 1. adımda listelenen geliştirici paketlerinin yüklü olduğundan emin olun.

5. **.env dosyasını hazırlayın**
   - `.env` henüz yoksa örnek olarak aşağıdaki içeriği kullanın ve kendi API anahtarlarınızı ekleyin:
     ```
     DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
     BINANCE_API_KEY=
     BINANCE_SECRET_KEY=
     INITIAL_BALANCE=200.0
     MAX_TRADE_NOTIONAL_USD=150.0
     CYCLE_INTERVAL_MINUTES=2
     MIN_CONFIDENCE=0.4
     HISTORY_RESET_INTERVAL=35
     ```
   - `.env` dosyasını oluşturduktan sonra `source .venv/bin/activate` komutu ile tekrar sanal ortamı aktifleştirip `python -m dotenv list` (opsiyonel) komutuyla değişkenlerin okunduğunu doğrulayabilirsiniz.

6. **Kurulumu doğrulayın**
   ```bash
   python -m py_compile alpha_arena_deepseek.py
   python short_scenario_tests.py  # opsiyonel doğrulama
   ```

7. **Servisleri başlatın** — README'deki "Sistem Başlatma" adımlarını izleyin. tmux kullanmıyorsanız tek terminalde botu, ikinci terminalde web arayüzünü çalıştırabilirsiniz.

### 2.3 Gerçek Hesap / Simülasyon Modu

- `TRADING_MODE=simulation` (varsayılan) → Bot tüm işlemleri portföy yöneticisi üzerinden simüle eder, gerçek emir göndermez.
- `TRADING_MODE=live` → `binance.py` aracılığıyla Binance USDT-M Futures’a piyasa (`MARKET`) emri gönderilir.  
  - **Gerekenler:** Futures API anahtarları (`BINANCE_API_KEY`, `BINANCE_SECRET_KEY`), margin tipi (`BINANCE_MARGIN_TYPE=ISOLATED` önerilir) ve varsayılan kaldıraç (`BINANCE_DEFAULT_LEVERAGE`).  
  - **Testnet:** Gerçek fon kullanmadan önce `BINANCE_TESTNET=true` ayarıyla [testnet](https://testnet.binancefuture.com/) üzerinde dry-run yapın.  
  - **Güvenlik:** API anahtarına sadece gerekli izinleri verin (Futures/Trade + Read), mümkünse IP kısıtlaması tanımlayın.
- Canlı moda geçtikten sonra bot, her cycle’da ve her emirden sonra hesabı senkronize eder; açık pozisyonlar ve bakiyeler `portfolio_state.json` ile uyumlu tutulur.
- Geçmiş log temizleme çalıştırıldığında `history_backups/` klasöründe zaman damgalı yedekler oluşur; olası hata durumunda buradan geri yükleme yapabilirsiniz.
- `.env` güncellemelerinden sonra botu yeniden başlatın ve `python -m py_compile alpha_arena_deepseek.py` ile hızlı bir sözdizimi kontrolü yapın.

### 3. Sistem Başlatma

```bash
# tmux oturumu oluştur
tmux new -s arena_bot

# Botu başlat (Pencere 0)
source .venv/bin/activate
python3 alpha_arena_deepseek.py

# Yeni pencere aç (Ctrl+B, C)
# Web sunucusunu başlat (Pencere 1) - Flask tabanlı
source .venv/bin/activate
python3 admin_server_flask.py

# Oturumdan ayrıl (Ctrl+B, D)
```

### 4. Web Arayüzüne Erişim

- codeserver'da "PORTS" sekmesini açın
- 8000 portunu bulun ve linke tıklayın
- Arayüz http://localhost:8000 adresinde açılacak

## 🔧 Risk Seviyesi Yönetimi

Sistem 3 farklı risk seviyesi sunar:

### Risk Seviyelerini Değiştirme

```bash
# Low risk
sed -i 's/RISK_LEVEL=.*/RISK_LEVEL=low/' .env

# Medium risk
sed -i 's/RISK_LEVEL=.*/RISK_LEVEL=medium/' .env

# High risk
sed -i 's/RISK_LEVEL=.*/RISK_LEVEL=high/' .env
```

Değişiklikten sonra botu yeniden başlatın.

## 📊 Risk Parametreleri Detayları

### Temel Trading Parametreleri

- **INITIAL_BALANCE**: Başlangıç bakiyesi ($200)
- **MAX_TRADE_NOTIONAL_USD**: Maksimum işlem büyüklüğü
- **CYCLE_INTERVAL_MINUTES**: Analiz aralığı (dakika)
- **MAX_LEVERAGE**: Maksimum kaldıraç oranı
- **MIN_CONFIDENCE**: Minimum güven seviyesi (0-1 arası)
- **MAX_POSITIONS**: Aynı anda açık maksimum pozisyon sayısı

### Risk Yönetimi Parametreleri

- **max_portfolio_risk**: Toplam portföy risk limiti (%)
- **max_position_risk**: Tek pozisyon risk limiti (%)
- **risk_per_trade**: İşlem başına risk miktarı
- **confidence_adjustment**: Güven seviyesine göre pozisyon ayarlama

## 🎯 AI Trading Stratejisi

### Teknik Analiz
- **EMA (20, 50)**: Trend analizi
- **RSI (7, 14)**: Momentum göstergesi
- **MACD**: Trend dönüş sinyalleri
- **ATR**: Volatilite ve stop-loss mesafesi
- **Volume Analysis**: İşlem hacmi analizi

### Piyasa Verileri
- **Open Interest**: Açık pozisyonlar
- **Funding Rate**: Finansman oranı
- **Price Action**: Fiyat hareketleri
- **Market Sentiment**: Piyasa duyarlılığı

## 📈 Performans Metrikleri

- **Total Return**: Toplam getiri (%)
- **Sharpe Ratio**: Risk-ajuste getiri oranı
- **Win Rate**: Kazanç oranı
- **Max Drawdown**: Maksimum düşüş
- **Trade Frequency**: İşlem sıklığı

## 🔍 Dosya Yapısı

```
.
├── alpha_arena_deepseek.py    # Ana trading botu
├── binance.py                 # Binance Futures emir yürütücüsü (live mode)
├── admin_server_flask.py      # Web sunucusu (Flask)
├── config.py                  # Konfigürasyon yönetimi
├── backtest.py                # Backtesting modülü
├── utils.py                   # Yardımcı fonksiyonlar
├── enhanced_context_provider.py  # Enhanced context sağlayıcı
├── performance_monitor.py     # Performans izleme ve trend reversal
├── cache_manager.py           # Cache yönetimi
├── prompt_json_builders.py    # JSON prompt builder fonksiyonları (v1.0)
├── prompt_json_utils.py       # JSON utility fonksiyonları (v1.0)
├── prompt_json_schemas.py     # JSON schema tanımları (v1.0)
├── test_prompt_json.py        # JSON prompt unit testleri
├── test_prompt_integration.py # Integration testleri
├── test_prompt_ab_comparison.py  # A/B karşılaştırma testleri
├── index.html                 # Web arayüzü
├── .env                       # Çevre değişkenleri
├── .envexample.txt            # Örnek .env dosyası
├── portfolio_state.json       # Portföy durumu
├── trade_history.json         # İşlem geçmişi
├── cycle_history.json         # Cycle geçmişi
├── history_backups/           # Periyodik log yedekleri
└── manual_override.json       # Manuel müdahale dosyası
```

## 🛠️ Gelişmiş Özellikler

### JSON Prompt Format (v1.0)

Sistem artık AI'a gönderilen prompt'ları JSON formatında sunabilir. Bu özellik:

- **Daha İyi Parsing**: AI modeli yapılandırılmış veriyi daha kolay parse eder
- **Token Optimizasyonu**: Compact mode ile %26'ya kadar token tasarrufu
- **Tip Güvenliği**: Veri tipleri açıkça belirtilir (number, string, array, object)
- **Kolay Validation**: JSON schema validation ile veri doğrulama
- **Series Compression**: Büyük seriler otomatik olarak sıkıştırılır (%79 token tasarrufu)

#### JSON Prompt Kullanımı

```bash
# .env dosyasında
USE_JSON_PROMPT=true
JSON_PROMPT_COMPACT=true  # Token tasarrufu için önerilir
```

**Özellikler:**
- `USE_JSON_PROMPT`: JSON format'ı aktif eder (varsayılan: false)
- `JSON_PROMPT_COMPACT`: Compact JSON kullanır, token kullanımını azaltır (önerilir: true)
- `VALIDATE_JSON_PROMPTS`: Runtime'da JSON validation yapar (opsiyonel)
- `JSON_SERIES_MAX_LENGTH`: Seri uzunluğu bu değeri aşarsa otomatik sıkıştırma yapılır (varsayılan: 50)

**Avantajlar:**
- Compact mode ile text format'tan %26 daha az token kullanımı
- Büyük serilerde %79 token tasarrufu (compression ile)
- Hata durumunda otomatik text format'a fallback
- NaN/None değerler güvenli şekilde handle edilir

**Test:**
```bash
# Unit testler
python test_prompt_json.py

# Integration testler
python test_prompt_integration.py

# A/B karşılaştırma
python test_prompt_ab_comparison.py
```

### Manuel Müdahale
`manual_override.json` dosyası oluşturarak manuel işlem yapabilirsiniz:

```json
{
  "decisions": {
    "BTC": {
      "signal": "buy_to_enter",
      "leverage": 10,
      "quantity_usd": 50,
      "confidence": 0.75,
      "profit_target": 55000,
      "stop_loss": 48000
    }
  }
}
```

### Backtesting
Geçmiş verilerle strateji testi:

```python
from backtest import BacktestEngine

engine = BacktestEngine(initial_balance=1000.0)
result = engine.run_backtest(strategy_func, symbols, start_date, end_date)
```

## 🚨 Risk Uyarıları

1. **Yüksek Risk**: Kripto ticareti yüksek risk içerir
2. **API Limitleri**: API kullanım limitlerine dikkat edin
3. **Backup**: Düzenli yedekleme yapın
4. **Monitoring**: Sistem performansını sürekli izleyin
5. **Stop-Loss**: Her zaman stop-loss kullanın

## 📞 Destek ve Sorun Giderme

### Sık Karşılaşılan Sorunlar

**API Bağlantı Hatası**
- API anahtarlarını kontrol edin
- İnternet bağlantısını doğrulayın
- API limitlerini kontrol edin

**ModuleNotFoundError**
- Sanal ortamı aktifleştirin: `source .venv/bin/activate`
- Gerekli kütüphaneleri yükleyin: `pip install -r requirements.txt`
- `pip: command not found` hatası alırsanız Python kurulumunun PATH üzerinde olduğundan emin olun veya `python -m ensurepip --upgrade` komutunu çalıştırın.

**Port Çakışması**
- `admin_server.py` dosyasında port numarasını değiştirin
- Farklı bir port kullanın

### Log Analizi
Log dosyalarını kontrol ederek sistem durumunu izleyin:
- `portfolio_state.json` - Mevcut portföy durumu
- `trade_history.json` - İşlem geçmişi
- `cycle_history.json` - AI karar geçmişi

## 📄 Lisans

Bu proje MIT lisansı altında dağıtılmaktadır.

## 🤝 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Commit edin (`git commit -m 'Add some AmazingFeature'`)
4. Push edin (`git push origin feature/AmazingFeature`)
5. Pull Request oluşturun

---

**Önemli Not**: Bu yazılım eğitim amaçlıdır. Gerçek para ile ticaret yapmadan önce kapsamlı testler yapın ve riskleri anlayın.
