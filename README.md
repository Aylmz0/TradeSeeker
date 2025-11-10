# Alpha Arena DeepSeek Trading Bot

Profesyonel kripto para ticaret botu - DeepSeek AI entegrasyonu ile otomatik trading sistemi.

## 🚀 Özellikler

- **AI-Powered Trading**: DeepSeek API ile akıllı ticaret kararları
- **Multi-Asset Support**: XRP, DOGE, ASTR, ADA, LINK, SOL
- **Advanced Risk Management**: Dinamik risk yönetimi ve pozisyon boyutlandırma
- **Auto TP/SL**: Otomatik kar al ve stop-loss yönetimi
- **Real-time Data**: Binance API ile gerçek zamanlı piyasa verileri
- **Web Dashboard**: Gerçek zamanlı izleme ve kontrol paneli
- **Flexible Risk Levels**: Low, Medium, High risk seviyeleri

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

# Trading Configuration
INITIAL_BALANCE=200.0
MAX_TRADE_NOTIONAL_USD=150.0
CYCLE_INTERVAL_MINUTES=2
MAX_LEVERAGE=15
MIN_CONFIDENCE=0.4
MAX_POSITIONS=4

# Risk Level Configuration
RISK_LEVEL=medium  # Options: low, medium, high
```

### 2.1 Canlı Kullanım Konfigürasyon İpuçları

- **HISTORY_RESET_INTERVAL**: (Varsayılan `35`) Her bu kadar cycle'da geçmiş logları temizler, sistemin uzun süreli bias geliştirmesini engeller. Canlıda 30-50 arası değer önerilir.  
- **SAME_DIRECTION_LIMIT**: Maksimum aynı yönde (long/short) pozisyon slotu. Futures cüzdan boyutunuza göre azaltıp artırabilirsiniz; borsanın kaldıraç limitini aşmamasına dikkat edin.  
- **CYCLE_INTERVAL_MINUTES** & **calculate_optimal_cycle_frequency**: Varsayılan 2 dakika. Spot/USDT perpetual tarafında API sınırlarını zorlamamak için minimum 2 dk önerilir; volatilite yüksekse bot otomatik olarak 2-4 dakika aralığına geçer.  
- **MIN_CONFIDENCE**: AI karar filtre eşiği. Gerçek bakiyede çok düşük ayarlanması gereksiz işlem sayısını artırabilir; 0.4-0.5 aralığı sağlıklı.  
- **INITIAL_BALANCE / MIN_POSITION_MARGIN_USD**: Gerçek bakiyeniz farklıysa `.env` ve `config.py` değerlerini güncelleyip botu yeniden başlatın; margin limitleri yeni bakiyeye göre otomatik ölçeklenir.  
- **API Limits & Failover**: Binance tarafında saniyede 10 istek limitini aşmamak için `MAX_RETRY_ATTEMPTS`, `REQUEST_TIMEOUT` gibi parametreleri aşırı düşürmeyin.  

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
├── admin_server.py            # Web sunucusu
├── config.py                  # Konfigürasyon yönetimi
├── backtest.py                # Backtesting modülü
├── utils.py                   # Yardımcı fonksiyonlar
├── index.html                 # Web arayüzü
├── .env                       # Çevre değişkenleri
├── portfolio_state.json       # Portföy durumu
├── trade_history.json         # İşlem geçmişi
├── cycle_history.json         # Cycle geçmişi
└── manual_override.json       # Manuel müdahale dosyası
```

## 🛠️ Gelişmiş Özellikler

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
