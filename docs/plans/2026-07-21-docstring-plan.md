# Docstring Plan — TradeSeeker

**Tarih:** 2026-07-21
**Sıra:** **Docstring** → Test → Refactoring
**Hedef:** 449 fonksiyonun **tamamını** baştan yaz (mevcut olanlar da dahil — güvenilmiyor)

---

## 1. Neden Tamamını Yeniden Yazıyoruz?

Mevcut 232 docstring sorunlu:

| Sorun | Açıklama | Sayı |
|-------|----------|------|
| `no_structure` | Sadece one-liner, Args/Returns yok | 232 |
| `too_short` | 4 kelimeden kısa, anlamsız | ~40 |
| `todo_marker` | TODO/FIXME ile bırakılmış | ~10 |
| `returns_no_desc` | Returns var ama açıklama yok | ~15 |
| `args_no_desc` | Args var ama parametre açıklaması yok | ~20 |

**Örnekler:**
```python
# ❌ Mevcut (kötü)
def get_stats() -> dict:
    """Get cache statistics."""

def format_performance_insights(data):
    """Formats performance insights."""

def safe_file_read(path):
    """Safely read JSON file with error handling"""
```

**Bunların hepsi yeniden yazılacak.**

---

## 2. Mevcut Durum Özeti

| Kategori | Fonksiyon | Yüzde |
|----------|-----------|-------|
| Docstring yok | 117 | %26 |
| Mevcut ama hatalı/kötü | 232 | %52 |
| Kabul edilebilir ( Args+Returns) | 100 | %22 |
| **TOPLAM YENİDEN YAZILACAK** | **349** | **%78** |

---

## 3. Docstring Standardı

### Zorunlu Format: Google-style
```python
def function_name(param1: str, param2: int = 10) -> bool:
    """One-line summary (zorunlu, kısa ve açıklayıcı).

    Extended description (isteğe bağlı, karmaşık fonksiyonlarda gerekli).

    Args:
        param1: Parametre açıklaması.
        param2: Parametre açıklaması (default: 10).

    Returns:
        True if successful, False otherwise.

    Raises:
        ValueError: If param2 is negative.

    Note:
        Ek notlar (isteğe bağlı).
    """
```

### Mutlak Kurallar
1. **Her fonksiyona** docstring yaz (class ve module level da dahil)
2. **Args:** bölümü zorunlu — her parametre tanımlanmalı
3. **Returns:** bölümü zorunlu — dönüş değeri açıklanmalı
4. **Raises:** bölümü isteğe bağlı — sadece raise eden fonksiyonlarda
5. **Note:** bölümü isteğe bağlı — ek bilgi için
6. **One-line summary** zorunlu — kısa ve açıklayıcı
7. Type hint'ler docstring'de tekrar edilmez (zaten imzada var)
8. Kodu docstring'de yeniden yazma — ne yaptığını açıkla, nasıl yaptığını değil
9. "This function does X" gibi anlamsız summary yazma
10. Fonksiyonun gerçek davranışını yaz — ne yapması gerektiğine değil, ne yaptığına

---

## 4. Dosya Bazlı Detaylı Plan

Her satır: `dosya` | `toplam fonksiyon` | `yeni yazılacak` = (yok + mevcut ama kötü)

### Faz 1 — Kritik Core Logic (~5 saat)

| # | Dosya | Toplam | Yok | Kötü | Tam | Yeni Yazılacak |
|---|-------|--------|-----|------|-----|-----------------|
| 1 | `core/market_data.py` | 19 | 18 | 0 | 1 | **18** |
| 2 | `core/data_engine.py` | 15 | 15 | 0 | 0 | **15** |
| 3 | `core/strategy_analyzer.py` | 3 | 1 | 2 | 0 | **3** |
| 4 | `core/performance_monitor.py` | 12 | 2 | 10 | 0 | **12** |
| 5 | `core/regime_detector.py` | 3 | 0 | 2 | 1 | **2** |
| | **Faz 1 Toplam** | **52** | **36** | **14** | **2** | **50** |

### Faz 2 — AI Modülleri (~5 saat)

| # | Dosya | Toplam | Yok | Kötü | Tam | Yeni Yazılacak |
|---|-------|--------|-----|------|-----|-----------------|
| 6 | `ai/deepseek_api.py` | 7 | 1 | 3 | 3 | **4** |
| 7 | `ai/enhanced_context_provider.py` | 11 | 1 | 10 | 0 | **11** |
| 8 | `ai/prompt_json_builders.py` | 17 | 0 | 7 | 10 | **7** |
| 9 | `ai/prompt_json_schemas.py` | 10 | 0 | 9 | 1 | **9** |
| 10 | `ai/prompt_json_utils.py` | 8 | 0 | 2 | 6 | **2** |
| 11 | `core/ai_service.py` | 23 | 2 | 18 | 3 | **20** |
| | **Faz 2 Toplam** | **76** | **4** | **49** | **23** | **53** |

### Faz 3 — Servisler (~3 saat)

| # | Dosya | Toplam | Yok | Kötü | Tam | Yeni Yazılacak |
|---|-------|--------|-----|------|-----|-----------------|
| 12 | `services/binance.py` | 35 | 26 | 7 | 2 | **33** |
| 13 | `services/alert_system.py` | 17 | 3 | 14 | 0 | **17** |
| 14 | `services/ml_service.py` | 7 | 2 | 5 | 0 | **7** |
| | **Faz 3 Toplam** | **59** | **31** | **26** | **2** | **57** |

### Faz 4 — Büyük Dosyalar (~5 saat)

| # | Dosya | Toplam | Yok | Kötü | Tam | Yeni Yazılacak |
|---|-------|--------|-----|------|-----|-----------------|
| 15 | `core/portfolio_manager.py` | 98 | 4 | 15 | 79 | **19** |
| 16 | `core/account_service.py` | 16 | 3 | 12 | 1 | **15** |
| 17 | `core/backtest.py` | 22 | 11 | 10 | 1 | **21** |
| 18 | `core/cache_manager.py` | 37 | 8 | 26 | 3 | **34** |
| 19 | `core/indicators.py` | 21 | 6 | 11 | 4 | **17** |
| | **Faz 4 Toplam** | **194** | **32** | **74** | **88** | **106** |

### Faz 5 — Main & Yardımcılar (~3 saat)

| # | Dosya | Toplam | Yok | Kötü | Tam | Yeni Yazılacak |
|---|-------|--------|-----|------|-----|-----------------|
| 20 | `main.py` | 23 | 6 | 17 | 0 | **23** |
| 21 | `web/admin_server_flask.py` | 25 | 2 | 23 | 0 | **25** |
| 22 | `utils.py` | 10 | 4 | 5 | 1 | **9** |
| 23 | `core/log_config.py` | 3 | 1 | 1 | 1 | **2** |
| 24 | `schemas/config.py` | 6 | 0 | 6 | 0 | **6** |
| 25 | `core/schemas/alignment.py` | 1 | 1 | 0 | 0 | **1** |
| | **Faz 5 Toplam** | **68** | **14** | **52** | **2** | **66** |

---

## 5. Genel Toplam

| Faz | Dosya Sayısı | Toplam Fonksiyon | Yeni Yazılacak | Süre |
|-----|-------------|------------------|----------------|------|
| Faz 1 — Kritik Core | 5 | 52 | 50 | ~5s |
| Faz 2 — AI | 6 | 76 | 53 | ~5s |
| Faz 3 — Servisler | 3 | 59 | 57 | ~3s |
| Faz 4 — Büyük Dosyalar | 5 | 194 | 106 | ~5s |
| Faz 5 — Main & Yardımcılar | 6 | 68 | 66 | ~3s |
| **TOPLAM** | **25 dosya** | **449** | **332** | **~21s** |

---

## 6. Büyük Dosya Batch Kuralları

**20+ fonksiyon içeren dosyalar** asla tek seferde overwrite edilmez.

| Dosya | Toplam | Batch Sayısı |
|-------|--------|--------------|
| `core/portfolio_manager.py` | 98 | 10 batch (10+10+...+8) |
| `services/binance.py` | 35 | 4 batch (10+10+10+5) |
| `core/cache_manager.py` | 37 | 4 batch (10+10+10+7) |
| `core/backtest.py` | 22 | 3 batch (10+10+2) |
| `core/indicators.py` | 21 | 3 batch (10+10+1) |
| `web/admin_server_flask.py` | 25 | 3 batch (10+10+5) |
| `core/ai_service.py` | 23 | 3 batch (10+10+3) |
| `main.py` | 23 | 3 batch (10+10+3) |

**Kural:** `read` ile batch'i oku → sadece o batch'i `edit` ile güncelle → sonraki batch.

---

## 7. Uygulama Kuralları

### Her Dosya İçin Adımlar
1. Dosyayı oku — tüm fonksiyonların imzasını ve gövdesini anla
2. Mevcut docstring'i sil (eğer varsa)
3. Her fonksiyon için sıfırdan Google-style docstring yaz
4. `ruff check` ile doğrula
5. Sonraki dosyaya geç

### Kontrol Listesi (her fonksiyon için)
- [ ] One-line summary var mı? (Kısa, açıklayıcı, "This function..." içermiyor)
- [ ] Args bölümü var mı? (Her parametre tanımlı, açıklama var)
- [ ] Returns bölümü var mı? (Düşük value açıklanmış)
- [ ] Raises bölümü var mı? (raise varsa zorunlu)
- [ ] Type hint'ler tekrar edilmemiş mi?
- [ ] Kodu yeniden yazmamış mı?
- [ ] Fonksiyonun GERÇEK davranışını açıklıyor mu?

### Hatalar
- ❌ "This function does X" gibi anlamsız summary
- ❌ Args'da sadece parametre adı, açıklama yok
- ❌ Returns'de sadece "The result" gibi muğlak ifade
- ❌ Kodu docstring'de tekrar yazma
- ❌ Type hint'leri docstring'de tekrar etme
- ❌ TODO/FIXME ile bırakma
- ❌ One-liner ile geçiştirme (Args/Returns olmadan)

---

## 7. Sonrası

Docstring tamamlandıktan sonra:
1. `ruff check src/` — temiz olmalı
2. `.venv/bin/python -m pdoc src -o docs/api -d google --no-include-undocumented --no-show-source` — API docs'ları yenile
3. `graphify update ./src` — graph'i güncelle
4. Test fazına geç
