# Risk Seviyesi Switch Sistemi - Detaylı Kılavuz

## 📋 Risk Seviyeleri ve Parametreler

Sistem artık 3 farklı risk seviyesi arasında kolayca geçiş yapabilir. Her risk seviyesi, trading stratejinizi ve risk maruziyetinizi belirleyen bir dizi parametre içerir.

## 🔵 LOW RISK (Düşük Risk) - Muhafazakar

### Parametre Değerleri
- **Maksimum İşlem Büyüklüğü:** $100
- **Maksimum Kaldıraç:** 10x
- **Minimum Güven Seviyesi:** 0.5
- **Maksimum Pozisyon Sayısı:** 3
- **Risk/Reward Oranı:** 1:1.5
- **Portföy Risk Limiti:** %2
- **Pozisyon Risk Limiti:** %1

### Kullanım Senaryoları
- Yeni başlayan kullanıcılar
- Yüksek volatilite dönemleri
- Korunmacı trading stratejileri
- Uzun vadeli pozisyonlar

## 🟡 MEDIUM RISK (Orta Risk) - Nof1ai Mantığı

### Parametre Değerleri
- **Maksimum İşlem Büyüklüğü:** $200
- **Maksimum Kaldıraç:** 20x
- **Minimum Güven Seviyesi:** 0.4
- **Maksimum Pozisyon Sayısı:** 4
- **Risk/Reward Oranı:** 1:1.3
- **Portföy Risk Limiti:** %10
- **Pozisyon Risk Limiti:** %6

### Kullanım Senaryoları
- Deneyimli kullanıcılar
- Normal piyasa koşulları
- Dengeli risk/getiri profili
- Orta vadeli stratejiler

## 🔴 HIGH RISK (Yüksek Risk) - Agresif

### Parametre Değerleri
- **Maksimum İşlem Büyüklüğü:** $200
- **Maksimum Kaldıraç:** 20x
- **Minimum Güven Seviyesi:** 0.3
- **Maksimum Pozisyon Sayısı:** 5
- **Risk/Reward Oranı:** 1:1.1
- **Portföy Risk Limiti:** %5
- **Pozisyon Risk Limiti:** %3

### Kullanım Senaryoları
- Çok deneyimli trader'lar
- Güçlü trend dönemleri
- Yüksek risk toleransı
- Kısa vadeli spekülasyon

## 🔧 Parametre Detaylı Açıklamaları

### 1. Maksimum İşlem Büyüklüğü (MAX_TRADE_NOTIONAL_USD)
**Ne İşe Yarar:** Tek bir işlemde kullanabileceğiniz maksimum USD miktarı
**Etkisi:** 
- Düşük: Küçük pozisyonlar, düşük risk
- Yüksek: Büyük pozisyonlar, yüksek risk
**Örnek:** $150 = Her işlemde maksimum $150 kullanabilirsiniz

### 2. Maksimum Kaldıraç (MAX_LEVERAGE)
**Ne İşe Yarar:** Pozisyon büyütme oranı
**Etkisi:**
- Düşük: Düşük getiri, düşük risk
- Yüksek: Yüksek getiri, yüksek risk
**Örnek:** 15x = $10 margin ile $150 pozisyon açabilirsiniz

### 3. Minimum Güven Seviyesi (MIN_CONFIDENCE)
**Ne İşe Yarar:** AI'nın işlem yapmak için gereken minimum güven seviyesi
**Etkisi:**
- Yüksek: Sadece yüksek güvenilirlikte işlemler
- Düşük: Daha fazla işlem fırsatı
**Örnek:** 0.4 = AI %40 güvenle işlem yapabilir

### 4. Maksimum Pozisyon Sayısı (MAX_POSITIONS)
**Ne İşe Yarar:** Aynı anda açık olabilecek maksimum pozisyon sayısı
**Etkisi:**
- Düşük: Daha iyi odaklanma, daha az çeşitlendirme
- Yüksek: Daha fazla çeşitlendirme, daha fazla yönetim
**Örnek:** 4 = 4 farklı coinde aynı anda pozisyon açabilirsiniz

### 5. Risk/Reward Oranı
**Ne İşe Yarar:** Risk alınan her $1 için beklenen getiri
**Etkisi:**
- Yüksek (1:1.5): Daha yüksek getiri beklentisi
- Düşük (1:1.1): Daha düşük getiri beklentisi
**Örnek:** 1:1.3 = $1 risk için $1.3 getiri hedefi

### 6. Portföy Risk Limiti (max_portfolio_risk)
**Ne İşe Yarar:** Toplam portföyünüzün maksimum risk maruziyeti
**Etkisi:**
- Düşük: Daha güvenli, daha düşük potansiyel kayıp
- Yüksek: Daha riskli, daha yüksek potansiyel kayıp
**Örnek:** %3 = $200 portföyde maksimum $6 risk

### 7. Pozisyon Risk Limiti (max_position_risk)
**Ne İşe Yarar:** Tek bir pozisyonda alabileceğiniz maksimum risk
**Etkisi:**
- Düşük: Daha küçük pozisyonlar
- Yüksek: Daha büyük pozisyonlar
**Örnek:** %2 = $200 bakiyede pozisyon başına maksimum $4 risk

## ⚙️ Risk Seviyesi Değiştirme Yöntemleri

### 1. .env Dosyasını Manuel Düzenleme
```bash
# Düşük risk için
RISK_LEVEL=low

# Orta risk için (varsayılan)
RISK_LEVEL=medium

# Yüksek risk için
RISK_LEVEL=high
```

### 2. Hızlı Komutlar
```bash
# Low risk'e geç
sed -i 's/RISK_LEVEL=.*/RISK_LEVEL=low/' .env

# Medium risk'e geç  
sed -i 's/RISK_LEVEL=.*/RISK_LEVEL=medium/' .env

# High risk'e geç
sed -i 's/RISK_LEVEL=.*/RISK_LEVEL=high/' .env
```

### 3. Manuel Override ile Geçici Değişiklik
`manual_override.json` dosyası oluşturun:
```json
{
  "risk_level": "high",
  "max_trade_notional": 200,
  "max_leverage": 20
}
```

## 🔍 Sistemin Çalıştığını Doğrulama

Risk seviyesi değişikliğinden sonra sistemi yeniden başlatın:
```bash
python alpha_arena_deepseek.py
```

Log çıktısında şu satırı görmelisiniz:
```
RISK_LEVEL: MEDIUM
```

## 📈 Performans Beklentileri

| Risk Seviyesi | Beklenen Getiri | Maksimum Drawdown | Trade Frekansı | Pozisyon Süresi |
|---------------|-----------------|-------------------|----------------|-----------------|
| Low           | %5-15           | %2-5              | Düşük          | Uzun            |
| Medium        | %15-30          | %5-10             | Orta           | Orta            |
| High          | %30-50+         | %10-20+           | Yüksek         | Kısa            |

**Not:** Bu rakamlar tahmini olup gerçek piyasa koşullarına göre değişiklik gösterebilir.

## 🎯 Önerilen Kullanım Stratejileri

### Yeni Başlayanlar İçin
1. **LOW RISK** ile başlayın
2. Sistemi 1-2 hafta gözlemleyin
3. Performansı değerlendirin
4. **MEDIUM RISK**'e kademeli geçiş yapın

### Deneyimli Kullanıcılar İçin
1. Piyasa koşullarına göre risk seviyesi seçin
2. Yüksek volatilite → **LOW RISK**
3. Normal koşullar → **MEDIUM RISK** 
4. Güçlü trend → **HIGH RISK**

### Profesyonel Stratejiler
1. **Portföy Çeşitlendirmesi:** Farklı risk seviyelerinde paralel botlar
2. **Dinamik Ayarlama:** Piyasa koşullarına göre otomatik risk ayarı
3. **Hedef Bazlı:** Getiri hedeflerine göre risk seviyesi seçimi

## 🛡️ Risk Yönetimi İpuçları

### 1. Pozisyon Boyutlandırma
- Her zaman stop-loss kullanın
- Pozisyon büyüklüğünü risk seviyesine göre ayarlayın
- Maksimum %2-5 risk kuralını uygulayın

### 2. Portföy Çeşitlendirmesi
- Aynı anda maksimum 3-5 pozisyon
- Farklı asset sınıflarına yatırım
- Korelasyon düşük varlıklar seçin

### 3. Sürekli İzleme
- Düzenli performans değerlendirmesi
- Risk metriklerini takip edin
- Strateji optimizasyonu yapın

### 4. Emniyet Önlemleri
- Manuel override dosyası hazır bulundurun
- Düzenli yedekleme yapın
- Acil durum planları oluşturun

## 🔄 Risk Seviyesi Geçiş Stratejileri

### 1. Kademeli Geçiş
- Low → Medium: 2 hafta test
- Medium → High: 1 hafta test
- Her geçişte performansı değerlendirin

### 2. Koşula Bağlı Geçiş
```python
# Örnek koşullu risk ayarı
if volatility > 0.05:  # Yüksek volatilite
    risk_level = "low"
elif strong_trend:     # Güçlü trend
    risk_level = "high"  
else:                  # Normal koşullar
    risk_level = "medium"
```

### 3. Zaman Bazlı Geçiş
- Sabah seansı: Medium risk
- Öğle volatilitesi: Low risk  
- Akşam trendi: High risk

## 📊 Risk Metrikleri ve Analiz

### Temel Metrikler
- **Sharpe Ratio:** Risk-ajuste getiri
- **Max Drawdown:** Maksimum düşüş
- **Win Rate:** Kazanç oranı
- **Profit Factor:** Kazanç/kayıp oranı

### Gelişmiş Metrikler
- **VaR (Value at Risk):** Belirli güven seviyesinde maksimum kayıp
- **Expected Shortfall:** VaR'ı aşan kayıpların ortalaması
- **Calmar Ratio:** Getiri/maksimum düşüş oranı

---

**Önemli Uyarı:** Bu kılavuzda belirtilen tüm parametreler ve stratejiler eğitim amaçlıdır. Gerçek para ile ticaret yapmadan önce kapsamlı backtesting ve risk değerlendirmesi yapın. Kripto para ticareti yüksek risk içerir ve sermaye kaybına yol açabilir.
