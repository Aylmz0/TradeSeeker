# 🚀 TradeSeeker: AI-Powered Crypto Trading Bot

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

**TradeSeeker**, DeepSeek LLM'in muhakeme yeteneğini klasik teknik analiz ile birleştiren yeni nesil bir algoritmik ticaret botudur. Sadece indikatörlere bakmaz; piyasa bağlamını (Context) okur, yorumlar ve bir fon yöneticisi gibi karar verir.

---

## ✨ Özellikler

*   **🧠 Hibrit Zeka:** RSI, EMA gibi matematiksel verileri AI'ın yorumlama gücüyle harmanlar.
*   **👁️ Görsel Analiz (Sparklines):** Fiyat grafiklerini metin tabanlı (` ▂▃▄▅`) görselleştirmelere çevirerek AI'a sunar.
*   **🛡️ Akıllı Risk Yönetimi:**
    *   **Dinamik Stop-Loss:** Piyasa oynaklığına (ATR) göre stop seviyesini otomatik ayarlar.
    *   **Anti-Choppy:** Testere piyasasını (Yatay) algılar ve işlem yapmayı durdurur.
    *   **Fake-Pump Koruması:** Hacim onayı olmayan yükselişlere kanmaz.
*   **⚡ Tam Otonom:** 7/24 piyasayı izler, fırsatları yakalar ve yönetir.

---

## 🛠️ Kurulum

Projeyi yerel makinenizde çalıştırmak için adımları takip edin.

### 1. Gereksinimler
*   Python 3.10 veya üzeri
*   DeepSeek API Anahtarı
*   Binance API Anahtarı (Canlı işlem yapılacaksa)

### 2. İndirme ve Hazırlık
```bash
# Depoyu klonlayın
git clone https://github.com/kullaniciadi/TradeSeeker.git
cd TradeSeeker

# Sanal ortam oluşturun (Önerilen)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Kütüphaneleri yükleyin
pip install -r requirements.txt
```

### 3. Ayarlar (.env)
Ana dizinde `.env` dosyası oluşturun ve gerekli ayarları girin:

```env
# AI Ayarları
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx

# Borsa Ayarları (Opsiyonel - Sadece Live Mod için)
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret_key

# Bot Ayarları
TRADING_MODE=simulation  # 'live' veya 'simulation'
INITIAL_BALANCE=1000     # Başlangıç bakiyesi (USD)
RISK_LEVEL=medium        # low, medium, high
```

---

## � Kullanım

Botu başlatmak için tek komut yeterlidir:

```bash
python src/main.py
```

Bot çalışmaya başladığında:
1.  Binance'den verileri çeker.
2.  Teknik analizi yapar.
3.  AI'a durumu sorar.
4.  Kararı terminale ve log dosyalarına (`data/`) yazar.

---

## � Proje Yapısı

*   `src/main.py`: Sistemin ana giriş noktası.
*   `src/ai/`: Yapay zeka ile iletişim ve prompt yönetimi.
*   `src/core/`: Piyasa verisi işleme ve portföy yönetimi.
*   `data/`: İşlem geçmişi ve performans raporlarının tutulduğu klasör.

---

## ⚠️ Yasal Uyarı

Bu proje **eğitim ve araştırma amaçlıdır**. Kripto para piyasaları yüksek risk içerir. Bu yazılımın kullanımından doğacak finansal kayıplardan geliştirici sorumlu değildir. Yatırım tavsiyesi değildir.

---

⭐ **Projeyi beğendiyseniz yıldız vermeyi unutmayın!**
