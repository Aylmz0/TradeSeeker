# Sistem Düzeltme Planı — 744 Cycle Gerçek Sorun Analizi

**Tarih:** 2026-07-18
**Durum:** 📋 PLAN — Onay bekliyor

---

## Gerçek Sorun Özeti

Mevcut korumalar **hepsi soft** — confidence cezası uyguluyor ama trade yine de açılıyor/kapatılıyor.

| Sorun | Mevcut Koruma | Neden Yetersiz |
|-------|---------------|----------------|
| CHOPPY'de işlem | confidence ×0.70, sizing ×0.70 | AI %85 confidence üretiyor, ceza bile %59'a düşürüyor — hâlâ MIN_CONFIDENCE üzeri |
| Counter-trend | 5 aşama pipeline, cooldown | Compound ceza bile yetersiz — trade yine açılıyor |
| AI panik satışı | 12dk hold, erosion, reversal | Erosion CRITICAL'e ulaşmadan AI kapatıyor |

---

## Kök Neden: Soft Ceza → Hard Block eksikliği

### K1: CHOPPY — Hard Block Gerekli

**Mevcut:** `confidence *= 0.70` + `regime_multiplier = 0.70`
**Sorun:** AI %85 confidence üretiyor → cezadan sonra %59 → hâlâ MIN_CONFIDENCE (genelde %0.40-0.50) üzeri

**Çözüm:** CHOPPY'de yeni pozisyon açmayı **hard block** et

**Dosya:** `src/core/portfolio_manager.py` — `_check_runtime_blocks()` fonksiyonuna ekle

```python
# CHOPPY hard block — yeni pozisyon açmayı engelle
if market_regime == "CHOPPY" and not has_existing_position:
    execution_report["blocked"].append({
        "coin": coin, "reason": "choppy_market_block",
        "regime": market_regime, "er": er_value,
    })
    trade["runtime_decision"] = "blocked_choppy_market"
    return False, confidence
```

**Not:** Mevcut pozisyonları kapatmayı engellemez — sadece yeni girişleri bloklar.

---

### K2: Counter-Trend — Hard Block Gerekli

**Mevcut:** 5 aşama pipeline (CT multiplier ×0.90, clash ×0.85, directional bias, cooldown, flip guard)
**Sorun:** Tüm cezalar compound olsa bile trade yine açılıyor. 20 counter-trend trade, %15 win rate, -$16.29

**Çözüm:** Counter-trend trades'i **hard block** et

**Dosya:** `src/core/portfolio_manager.py` — `_apply_counter_trend_flip_logic()` fonksiyonunda

```python
# Counter-trend hard block — tüm counter-trend girişlerini engelle
if classification == "counter_trend" and not has_existing_position:
    execution_report["blocked"].append({
        "coin": coin, "reason": "counter_trend_hard_block",
        "classification": classification,
    })
    trade["runtime_decision"] = "blocked_counter_trend_hard_block"
    return False, confidence, partial_margin_factor
```

**Not:** Mevcut pozisyonlar kapatılabilir — sadece yeni counter-trend girişlerini bloklar.

---

### K3: AI Panik Satışı — Erosion Wiring eksik

**Mevcut:**
- 12 dakika minimum hold shield var (`main.py:581-606`)
- Erosion tracking var (20/50/100% eşikleri)
- Reversal strength definitions var (WEAK→don't exit, STRONG→conditional, CRITICAL→mandatory)
- `validate_exit_signal()` fonksiyonu var ama **AI close path'ine bağlı değil** (dead code)

**Sorun:** Erosion status AI'a soft guidance olarak gönderiliyor ama AI yine de küçük zararla kapatıyor. `validate_exit_signal()` fonksiyonu (-1.5% ile +2% arası kapatmayı engelliyor) hiç çağrılmıyor.

**Çözüm 1:** `validate_exit_signal()`'i AI close path'ine bağla

**Dosya:** `src/main.py` — `close_position` işleme noktasında (line ~581)

```python
# Mevcut 12-minute hold kontrolüne ekle:
# validate_exit_signal() çağrısı — zayıf çıkışları engelle
if hasattr(self.portfolio, 'validate_exit_signal'):
    exit_valid = self.portfolio.validate_exit_signal(
        coin, position, current_price, "ai_close"
    )
    if not exit_valid:
        logger.info("AI close blocked — exit_signal validation failed for {}", coin)
        close_execution_report["blocked"].append({
            "coin": coin, "reason": "exit_signal_validation_failed",
        })
        continue
```

**Çözüm 2:** Erosion durumuna göre zorunlu hold

```python
# Erosion CRITICAL değilse kapatmayı engelle
erosion_status = getattr(position, "erosion_status", "NONE")
if erosion_status in ("NONE", "MINOR") and unrealized_pnl < 0:
    logger.info("AI close blocked — erosion {} (not critical)", erosion_status)
    close_execution_report["blocked"].append({
        "coin": coin, "reason": "erosion_not_critical",
        "erosion_status": erosion_status,
    })
    continue
```

---

### K4: directional_bias Rebuild — Gerekmez

**Mevcut:** Reset sonrası directional_bias sıfırlanıyor
**Neden rebuild zararlı olabilir:**
- `full_trade_history` eski piyasa koşullarını içeriyor
- Piyasa değiştiyse (BEARISH → BULLISH) eski verilerle rebuild etmek yanıltıcı olur
- directional_bias zaten **son 35 cycle** penceresinde çalışmak için tasarlanmış

**Doğru yaklaşım:** Rebuild etme — mevcut tasarımı koru. Sorun directional_bias'da değil, soft cezaların yetersizliğinde.

---

## Uygulama Önceliği

| Sıra | Çözüm | Efor | Beklenen Etki |
|:-----|:------|:-----|:-------------|
| 1 | K1: CHOPPY hard block | 10 dk | 47 trade engellenir → +$6.80 |
| 2 | K2: Counter-trend hard block | 10 dk | 20 trade engellenir → +$16.29 |
| 3 | K3: validate_exit_signal wiring | 15 dk | AI panik satışı azalır |
| 4 | K3: Erosion status gate | 10 dk | AI panik satışı azalır |

**Toplam efor:** ~45 dakika

---

## Beklenen Sonuç

| Senaryo | Mevcut | Düzeltilmiş |
|---------|--------|-------------|
| CHOPPY trade'leri | -$6.80 (47 trade) | **$0** (0 trade) |
| Counter-trend | -$16.29 (20 trade) | **$0** (0 trade) |
| AI panik satışı | -$26.27 (29 close) | **~-$10** (sadece kritik close) |
| **Toplam** | **-$26.57** | **~-$10** |

---

## Doğrulama

- [ ] CHOPPY piyasada yeni pozisyon açılmıyor (block log'u görünmeli)
- [ ] Counter-trend yeni giriş engelleniyor (block log'u görünmeli)
- [ ] AI close_position validate_exit_signal kontrolünden geçiyor
- [ ] Erosion NONE/MINOR iken AI kapatma engelleniyor
- [ ] Mevcut pozisyonlar kapatılabilir (hard block sadece girişler için)
- [ ] `ty check src/` + `ruff check src/` temiz
- [ ] 10 cycle test: CHOPPY ve counter-trend'de trade açılmamalı
