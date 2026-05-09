# TradeSeeker LLM Benchmark — Cevap Anahtarı

Her senaryo farklı bir karar kuralını test eder. Her karar **20 puan** üzerinden değerlendirilir.
Maksimum puan: **120**.

---

## Senaryo 1 — XRP: Trend Takipçisi LONG
**Doğru Karar:** `buy_to_enter` | `trend_following` | confidence ≥ 0.78

### Neden?
- 1h: TF_STRONG_BULLISH, ADX 38 STRONG
- 15m: HH_HL yapısı, STRENGTHENING momentum
- ML: BUY 46% → tekniklerle hizalı, conviction artırır
- Volume: GOOD (1.85x) — kısıt yok

### Puanlama (20p)
| Kriter | Puan |
|:---|:---:|
| `buy_to_enter` sinyali | 8 |
| `trend_following` stratejisi | 5 |
| Confidence ≥ 0.75 | 4 |
| CHAIN_OF_THOUGHTS'ta 1h + 15m hizalamasını doğru açıkladı | 3 |

---

## Senaryo 2 — DOGE: CT_HIGH_RISK — 3 Koşul Karşılandı
**Doğru Karar:** `sell_to_enter` | `counter_trend` | confidence 0.62–0.72

### Neden?
CT_HIGH_RISK için 3 koşulun TAMAMI karşılandı:
1. ✅ 15m yapı: LH_LL (1h BULLISH'e karşı döndü)
2. ✅ ML SELL: 0.41 ≥ 0.40
3. ✅ UPPER_10 + WEAKENING

Ek destekler: BEARISH_DIVERGENCE, OVEREXTENDED_UP, RSI 72 aşırı alım, Funding +0.04% (long bias aşırı).

### Puanlama (20p)
| Kriter | Puan |
|:---|:---:|
| `sell_to_enter` sinyali | 8 |
| `counter_trend` stratejisi | 4 |
| 3 koşulun tamamını CHAIN_OF_THOUGHTS'ta sayıp doğruladı | 5 |
| Confidence aralığında (0.62-0.72, düşük tutması doğru) | 3 |

> ⚠️ **Tuzak:** Agresif LLM burada "CT_HIGH_RISK = HOLD" der — bu YANLIŞ.
> Kuralı bilen LLM 3 koşul karşılandığında girer.

---

## Senaryo 3 — ASTER: CT_HIGH_RISK — ML Koşulu Eksik
**Doğru Karar:** `hold`

### Neden?
CT_HIGH_RISK için 3 koşulun TAMAMI gerekiyor:
1. ✅ 15m yapı: LH_LL ✓
2. ❌ ML SELL: **0.36 < 0.40** — BAŞARISIZ
3. ✅ UPPER_10 + WEAKENING ✓

Tek koşul eksik → HOLD. Koşulsuz girmeyi kabul etmemeli.

### Puanlama (20p)
| Kriter | Puan |
|:---|:---:|
| `hold` sinyali | 10 |
| Neden hold? ML koşulunun karşılanmadığını (0.36 < 0.40) açıkça belirtti | 7 |
| DOGE ile arasındaki farkı (0.41 vs 0.36) fark etti | 3 |

> ⚠️ **Tuzak:** Bu senaryo kasıtlı olarak Senaryo 2'ye çok benziyor.
> Farkı görmek LLM'in kurala sadakatini ve dikkatini test eder.

---

## Senaryo 4 — TRX: Açık LONG — Zorunlu Çıkış
**Doğru Karar:** `close_position` | `risk_management` | confidence ≥ 0.88

### Neden?
- `erosion_status: CRITICAL` (120.5% — peak_pnl $2.20 tamamen silindi, şimdi zarar)
- `reversal_threat: RT_STRONG`
- 15m yapı: LH_LL (LONG'a karşı döndü)
- ML SELL: 52%

Kural: *"CRITICAL erosion → Close immediately"* + RT_STRONG → çıkış zorunlu.

### Puanlama (20p)
| Kriter | Puan |
|:---|:---:|
| `close_position` sinyali | 10 |
| `risk_management` stratejisi | 4 |
| CHAIN_OF_THOUGHTS'ta CRITICAL erosion + RT_STRONG gerekçesi | 4 |
| Confidence ≥ 0.85 (kesin çıkış → yüksek confidence beklenir) | 2 |

---

## Senaryo 5 — ETH: POOR Volume, Confluence Yok
**Doğru Karar:** `hold`

### Neden?
- Volume: POOR (0.48x, 0.30-0.70 aralığında)
- Kural: *"POOR sadece güçlü yapısal onay ile kabul edilir: divergence + 15m yapı tersine dönüşü VEYA zone exhaustion"*
- ETH'te: rsi_divergence = NONE, price_location = MIDDLE (zone yok), momentum = STABLE
- Bullish setup var ama POOR volume'u gerekçelendirecek hiçbir confluence yok

### Puanlama (20p)
| Kriter | Puan |
|:---|:---:|
| `hold` sinyali | 10 |
| Volume POOR kuralını doğru uyguladı | 6 |
| Neden yetmediğini açıkladı (divergence yok + zone yok) | 4 |

> ⚠️ **Tuzak:** 1h bullish + 15m HH_HL görünce agresif LLM burada BUY verir.
> Kuralı bilen LLM volume confluence gerekliliğini hatırlar.

---

## Senaryo 6 — SOL: Kârlı Açık SHORT
**Doğru Karar:** `hold`

### Neden?
- 1h: TF_STABLE_BEARISH → SHORT'u destekliyor
- 15m: LH_LL → SHORT'u destekliyor
- RT_NONE, erosion_status = NONE
- Unrealized PnL: +$2.10
- ML SELL: 48% → pozisyonu destekliyor

Hiç kapama gerekçesi yok. Bekle.

### Puanlama (20p)
| Kriter | Puan |
|:---|:---:|
| `hold` sinyali | 10 |
| 1h rejim + 15m yapı hizalamasını gerekçe gösterdi | 6 |
| RT_NONE + erosion NONE'ı açıkça belirtti | 4 |

---

## Toplam Puan Tablosu

| | LLM-A | LLM-B |
|:---|:---:|:---:|
| XRP (Trend Following) | /20 | /20 |
| DOGE (CT_HIGH_RISK 3-koşul) | /20 | /20 |
| ASTER (CT_HIGH_RISK eksik koşul) | /20 | /20 |
| TRX (CRITICAL erosion exit) | /20 | /20 |
| ETH (POOR volume + no confluence) | /20 | /20 |
| SOL (Winning position, hold) | /20 | /20 |
| **TOPLAM** | **/120** | **/120** |

---

## Karar Kılavuzu

| Puan | Yorum |
|:---|:---|
| 100-120 | Mükemmel — Tüm kuralları eksiksiz uyguladı |
| 80-99 | İyi — 1-2 ince kurala takıldı |
| 60-79 | Orta — Temel kuralları biliyor ama detaylarda kayıyor |
| < 60 | Yetersiz — Kurallara yüzeysel uyum |

**Kritik Senaryo:** ASTER (S3) en önemli testtir. DOGE ile aynı görünüp farklı sonuç veren bu senaryo, LLM'in kural hassasiyetini ve dikkatini doğrudan ölçer.
