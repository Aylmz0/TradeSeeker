# Pandas/NumPy → Polars Migration Plan

## Mimari Bağımlılık Haritası

```
indicators.py  ←─ her şey buna bağlı
      │
      ├── market_data.py  ←─ indicators'ı çağırır
      ├── data_engine.py  ←─ indicators ile aynı pattern'leri kullanır
      ├── ml_service.py   ←─ indicators.get_features_for_ml() çağırır
      └── backtest.py     ←─ indicators kullanmaz ama pandas pattern paylaşıyor
```

---

## PARÇA 0: Altyapı
**Kapsam**: `requirements.txt` + polars kurulumu
**Süre**: 5 dakika
**Bağımlılık**: Yok
**Doğrulama**: `python -c "import polars; print(polars.__version__)"` çalışmalı

**Yapılacaklar**:
1. `requirements.txt`'den `pandas>=2.0.0` ve `numpy>=1.24.0` sil
2. `polars>=1.0.0` ekle
3. `pip install polars` çalıştır
4. Import doğrulaması yap

---

## PARÇA 1: Teknik İndikatörler (EN KRİTİK)
**Kapsam**: `src/core/indicators.py` (~670 satır)
**Süre**: 60-90 dakika
**Bağımlılık**: Parça 0
**Doğrulama**: `python -c "from src.core.indicators import calculate_ema_series"` çalışmalı

### Dönüşüm Tablosu

| Fonksiyon | Kullanılan Pattern | Polars Karşılığı |
|---|---|---|
| `calculate_ema_series` | `prices.ewm(span=N, adjust=False).mean()` | `prices.ewm_mean(span=N, adjust=False)` |
| `calculate_rsi_series` | `delta.where(delta > 0, 0)`, `.ewm().mean()` | `pl.when().then().otherwise()`, `.ewm_mean()` |
| `calculate_macd_series` | EMA zinciri × 3 | EMA zinciri polars'ta aynı |
| `calculate_atr_series` | `pd.concat([s1,s2,s3], axis=1).max(axis=1)` | `pl.max_horizontal(s1, s2, s3)` |
| `calculate_adx` | Karmaşık `.where()`, `.fillna()`, `.ewm()` | `pl.when().then().otherwise()` + `.fill_null()` |
| `calculate_vwap` | `.rolling().sum()`, `.replace()` | `.rolling_sum()` + `.replace()` |
| `calculate_bollinger_bands` | `.rolling().mean()`, `.rolling().std()` | `.rolling_mean()`, `.rolling_std()` |
| `calculate_obv` | `np.sign()`, `.cumsum()` | `_sign()` + `.cum_sum()` |
| `calculate_supertrend` | NumPy array loop | Saf Python list + loop |
| `extract_semantic_features` | `np.polyfit()`, `np.std()`, `np.mean()` | `_linear_slope()` + polars |
| `calculate_slope_label` | `np.polyfit()` | `_linear_slope()` |
| `calculate_rsi_divergence_label` | `np.polyfit()` × 2 | `_linear_slope()` × 2 |
| `generate_smart_sparkline` | `np.ptp()`, `np.max()`, `np.min()` | `max()-min()`, `max()`, `min()` |
| `calculate_pivots` | `df.iloc[-periods:]` | `df.tail(periods)` |
| `get_features_for_ml` | Tüm pipeline | Tüm polars karşılıkları |

### Yeni Yardımcı Fonksiyonlar

```python
def _linear_slope(x, y):
    """Saf Python lineer regresyon — numpy polyfit yerine"""
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_x2 = sum(xi * xi for xi in x)
    denom = n * sum_x2 - sum_x * sum_x
    return (n * sum_xy - sum_x * sum_y) / denom if denom != 0 else 0.0

def _sign(x):
    """np.sign yerine"""
    return 1 if x > 0 else (-1 if x < 0 else 0)
```

---

## PARÇA 2: Veri Katmanı
**Kapsam**: `src/core/data_engine.py` + `src/core/market_data.py`
**Süre**: 45-60 dakika
**Bağımlılık**: Parça 1
**Doğrulama**: Import'lar çalışmalı

### data_engine.py
- `pd.read_sql_query()` → `sqlite3 cursor.execute().fetchall()` + `pl.DataFrame()`
- `df.iterrows()` → `df.iter_rows(named=True)`
- `pd.DataFrame(data, columns=[...])` → `pl.DataFrame(data)`
- `df["col"].astype(float).round(8)` → `df["col"].cast(pl.Float64).round(8)`
- `pd.to_datetime(ts, unit="ms")` → `pl.from_epoch("timestamp", time_unit="ms")`
- `np.select()` → `pl.when().then().otherwise()` zinciri
- `np.maximum()` → `pl.max_horizontal()`
- `df.dropna()` → `df.drop_nulls()`

### market_data.py
- `.where(pd.notna, None).tolist()` → `.fill_null(None).to_list()`
- `.iloc[-1]` → `series[-1]`
- `.nunique()` → `.n_unique()`
- `np.isnan()`, `np.isinf()` → `math.isnan()`, `math.isinf()`
- `pd.notna()` → `.is_not_null()`

---

## PARÇA 3: Servisler
**Kapsam**: `ml_service.py`, `ai_service.py`, `performance_monitor.py`, `portfolio_manager.py`, `backtest.py`
**Süre**: 30-45 dakika
**Bağımlılık**: Parça 1 ve 2

| Dosya | Kritik Dönüşüm |
|---|---|
| `ml_service.py` | `df.iloc[[-1]][cols]` → `df.select(cols).tail(1)` |
| `ai_service.py` | `pd.notna()` → `is not None and not math.isnan()` |
| `performance_monitor.py` | `pd.Series.pct_change()` → `pl.Series.pct_change()` |
| `portfolio_manager.py` | Sadece numpy import'u var → kaldır |
| `backtest.py` | `np.random` → `random`, `np.mean/std` → `statistics` |

---

## PARÇA 4: Script & Scratch
**Kapsam**: 7 scripts + 4 scratch dosyası
**Süre**: 30-45 dakika
**Bağımlılık**: Parça 1-3

Genel polars dönüşüm tablosu:
| Pandas | Polars |
|---|---|
| `pd.DataFrame()` | `pl.DataFrame()` |
| `pd.read_csv()` | `pl.read_csv()` |
| `.groupby().agg()` | `.group_by().agg()` |
| `.sort_values()` | `.sort()` |
| `.value_counts()` | `.group_by("col").len()` |
| `.apply(lambda)` | `.map_elements()` |
| `.iterrows()` | `.iter_rows(named=True)` |

---

## PARÇA 5: Temizlik & Doğrulama
**Kapsam**: Tüm proje
**Süre**: 15-20 dakika

1. `grep -r "import pandas" src/` → sonuç boş olmalı
2. `grep -r "import numpy" src/` → sonuç boş olmalı
3. `ruff check src/` → temiz olmalı
4. `ty check src/` → temiz olmalı
5. Tüm ana modüller import edilebilmeli

---

## İkinci İnceleme Turu (Doğrulama) — Bulunan ve Düzeltilen Çalışma-Zamanı Hataları

Planın Parça 1-5'i tamamlandıktan sonra, kod SADECE import edilerek (static) değil,
çalıştırılarak (runtime) test edildi. Import testi geçse bile çalışma zamanında çöken
polars uyumsuzlukları bulundu ve düzeltildi:

| # | Dosya | Satır | Pandas API (bozuk) | Polars Karşılığı | Durum |
|---|-------|------|--------------------|------------------|-------|
| 1 | `indicators.py` | 825 | `pct_change(periods=N)` | `pct_change(n=N)` | DÜZELTİLDİ |
| 2 | `indicators.py` | 846 | `df.forward_fill()` | `df.fill_null(strategy="forward")` | DÜZELTİLDİ |
| 3 | `indicators.py` | 847 | `map_batches(lambda x: x.fill_nan(None))` | `df.fill_nan(None)` | DÜZELTİLDİ |
| 4 | `ai_service.py` | 923,927 | `df.empty` | `df.is_empty()` | DÜZELTİLDİ |
| 5 | `main.py` | 406 | `df.empty` | `df.is_empty()` | DÜZELTİLDİ |
| 6 | `backtest.py` | 296,301,308 | `df.iloc[i]["col"]` / `df.iloc[:i+1]` | `df["col"][i]` / `df.head(i+1)` | DÜZELTİLDİ |
| 7 | `train_model.py` | 57,64 | `df.empty` | `df.is_empty()` | DÜZELTİLDİ |
| 8 | `data_engine.py` | 209 | `pl.DataFrame(rows, schema=cols)` (warning) | `pl.DataFrame(rows, schema=cols, orient="row")` | DÜZELTİLDİ |

**Not (ölü kod):** `src/core/cache_manager.py` içindeki `optimize_dataframe_operations`
metodu hâlâ pandas API'si (`.query()`, `.apply()`, `.copy()`) kullanıyor ancak proje
içinde HİÇ çağrılmıyor (dead code) ve pandas import etmiyor. Çalışma zamanı etkisi yok.

**Fonksiyonel doğrulama (gerçek veriyle):** `get_features_for_ml` (99 satır/47 kolon),
`calculate_adx`, `data_engine` yaz-oku roundtrip, `backtest` iloc düzeltmesi, `ml_service.predict`
hepsi çalışıyor. Tüm modüller (src + scripts + scratch + main.py) import başarılı.
