# Trailing Stop Loss Sistemi - Detaylı Açıklama

## 🎯 Amaç
Kar elde edildiğinde, stop loss'u yukarı çekerek (long pozisyonlar için) veya aşağı çekerek (short pozisyonlar için) kazanılan karı korumak.

---

## 🔄 Nasıl Çalışıyor?

### 1. **Aktivasyon Koşulları**

Trailing stop loss sistemi **sadece kar durumunda** aktif olur:
- `unrealized_pnl_percent > 0` (pozisyon karda olmalı)
- Minimum kar seviyesi: `level1 * 0.5` (level1'in yarısı kadar kar olmalı)

**İki farklı aktivasyon yöntemi var:**

#### A) **Progress-Based (İlerleme Bazlı)**
- **Koşul**: Profit target'a doğru ilerleme **%80 veya daha fazla** olmalı
- **Hesaplama**: 
  - Long: `((current_price - entry_price) / (profit_target - entry_price)) * 100`
  - Short: `((entry_price - current_price) / (entry_price - profit_target)) * 100`
- **Örnek**: Entry $100, TP $110, Current $108 → Progress = 80% ✅

#### B) **Time-Based (Zaman Bazlı)**
- **Koşul 1**: Pozisyonda **20 dakika veya daha fazla** kalınmış olmalı
- **Koşul 2**: Profit target'a doğru ilerleme **%50 veya daha fazla** olmalı
- **Örnek**: 25 dakika pozisyondasın, progress %60 → ✅

**Not**: İki koşuldan biri sağlanırsa trailing stop aktif olur.

---

### 2. **Stop Loss Hesaplama**

Trailing stop loss, **ATR (Average True Range)** ve **Volume** bazlı dinamik bir buffer kullanır:

#### A) **ATR Buffer Hesaplama**
```python
atr_buffer = max(
    atr_value * 1.2,  # ATR'nin 1.2 katı
    current_price * 0.0005  # Minimum %0.05 improvement
)
```

- **ATR**: Volatiliteyi ölçer, daha volatil coinlerde daha geniş buffer
- **Multiplier**: 1.2x (Config'de `TRAILING_ATR_MULTIPLIER`)
- **Fallback**: ATR yoksa, fiyatın %0.4'ü kullanılır

#### B) **Yeni Stop Loss Hesaplama**

**Long Pozisyonlar İçin:**
```python
baseline_stop = current_price - atr_buffer
# Minimum entry_price'ın üstünde olmalı (kar korunmalı)
baseline_stop = max(baseline_stop, entry_price + min_improvement)
# Mevcut stop loss'tan daha yukarıda olmalı (sadece yukarı çekilebilir)
if existing_stop:
    baseline_stop = max(baseline_stop, existing_stop + min_improvement)
```

**Short Pozisyonlar İçin:**
```python
baseline_stop = current_price + atr_buffer
# Maximum entry_price'ın altında olmalı (kar korunmalı)
baseline_stop = min(baseline_stop, entry_price - min_improvement)
# Mevcut stop loss'tan daha aşağıda olmalı (sadece aşağı çekilebilir)
if existing_stop:
    baseline_stop = min(baseline_stop, existing_stop - min_improvement)
```

---

### 3. **Minimum Improvement Kontrolü**

Stop loss **sadece yukarı çekilebilir** (long) veya **sadece aşağı çekilebilir** (short):
- **Minimum değişim**: Fiyatın %0.05'i veya `MIN_EXIT_PLAN_OFFSET` (hangisi büyükse)
- **Amaç**: Gereksiz güncellemeleri önlemek, sadece anlamlı değişikliklerde güncellemek

**Örnek:**
- Mevcut stop: $100.50
- Yeni hesaplanan stop: $100.48
- **Sonuç**: Güncelleme yapılmaz (aşağı inemez)

---

### 4. **Volume Kontrolü**

Volume düşüşü tespit edilirse, trailing stop daha agresif olabilir:
- **Absolute Threshold**: Volume ratio ≤ 0.2x → Volume düşüşü var
- **Relative Drop**: Mevcut volume, entry volume'un %50'sinden az → Volume düşüşü var

**Not**: Volume düşüşü trailing stop'un aktivasyonunu etkilemez, sadece log'a eklenir.

---

## 📊 Örnek Senaryo

### Senaryo: Long SOL Pozisyonu

**Başlangıç:**
- Entry Price: $140.00
- Stop Loss: $138.00 (initial)
- Profit Target: $145.00
- Quantity: 1.0 SOL
- Notional: $1400

**Adım 1: Kar Elde Edildi**
- Current Price: $143.00
- Unrealized PnL: $3.00 (%2.14)
- Progress: ((143-140)/(145-140)) * 100 = **60%**

**Durum**: Progress %80 değil, time-based kontrol edilir:
- 15 dakika pozisyondasın → ❌ (20 dakika gerekli)
- **Sonuç**: Trailing stop aktif değil

**Adım 2: Progress %80'e Ulaştı**
- Current Price: $144.00
- Progress: ((144-140)/(145-140)) * 100 = **80%** ✅
- ATR: $0.50
- ATR Buffer: $0.50 * 1.2 = $0.60

**Yeni Stop Loss Hesaplama:**
```
baseline_stop = $144.00 - $0.60 = $143.40
baseline_stop = max($143.40, $140.00 + $0.07) = $143.40
baseline_stop = max($143.40, $138.00 + $0.07) = $143.40
```

**Sonuç**: Stop loss $138.00 → **$143.40**'a çekildi! 🎯
- Artık en az $3.40 kar garantili
- Eğer fiyat $143.40'ın altına düşerse, pozisyon kapanır

**Adım 3: Fiyat Daha da Yükseldi**
- Current Price: $144.50
- ATR Buffer: $0.60

**Yeni Stop Loss Hesaplama:**
```
baseline_stop = $144.50 - $0.60 = $143.90
baseline_stop = max($143.90, $143.40 + $0.07) = $143.90
```

**Sonuç**: Stop loss $143.40 → **$143.90**'a çekildi! 🎯
- Artık en az $3.90 kar garantili

**Adım 4: Fiyat Geri Düştü**
- Current Price: $143.50
- Mevcut Stop Loss: $143.90

**Durum**: Fiyat stop loss'un altına düştü → **Pozisyon kapatılır!**
- Kapanış fiyatı: ~$143.90
- Kar: $3.90 ✅

---

## ⚙️ Konfigürasyon Parametreleri

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `TRAILING_PROGRESS_TRIGGER` | 80.0% | Progress-based aktivasyon için minimum ilerleme |
| `TRAILING_TIME_MINUTES` | 20 dakika | Time-based aktivasyon için minimum süre |
| `TRAILING_TIME_PROGRESS_FLOOR` | 50.0% | Time-based aktivasyon için minimum ilerleme |
| `TRAILING_ATR_MULTIPLIER` | 1.2x | ATR buffer çarpanı |
| `TRAILING_FALLBACK_BUFFER_PCT` | 0.4% | ATR yoksa kullanılan fallback buffer |
| `TRAILING_VOLUME_ABSOLUTE_THRESHOLD` | 0.2x | Volume düşüşü için absolute threshold |
| `TRAILING_VOLUME_DROP_RATIO` | 0.5x | Volume düşüşü için relative threshold |
| `TRAILING_MIN_IMPROVEMENT_PCT` | 0.05% | Minimum stop loss güncelleme miktarı |

---

## 🎯 Özet

1. **Aktivasyon**: Kar durumunda + (Progress %80+ VEYA 20dk+%50 progress)
2. **Hesaplama**: ATR bazlı dinamik buffer ile yeni stop loss hesaplanır
3. **Koruma**: Stop loss sadece yukarı çekilebilir (long) veya aşağı çekilebilir (short)
4. **Minimum Değişim**: Anlamlı değişikliklerde güncelleme yapılır
5. **Monitoring**: 30 saniyede bir kontrol edilir ve güncellenir

**Sonuç**: Kar elde edildiğinde, stop loss otomatik olarak yukarı çekilerek kazanılan kar korunur! 🛡️

