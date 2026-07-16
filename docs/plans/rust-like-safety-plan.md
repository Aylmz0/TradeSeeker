# Rust-Like Safety Plan — Python'da Güvenlik Katmanları

## 1. 🎯 Objective
Python kod tabanını Rust kadar güvenli hale getirmek: statik tip kontrolü, immutable veri yapıları, pattern matching.

## 2. 🏗️ Tech Strategy
- **Pattern:** Kademeli adoption — mevcut kodu bozmadan yeni katmanlar ekle
- **State:** ty (Rust-based type checker) + Pydantic frozen models + match/case + tuple-based Result
- **Constraints:** Pre-commit hooks ile zorunlu kıl, CI/CD'de error-on-warning

## 3. 📂 File Changes

| Action | File Path | Brief Purpose |
|:-------|:----------|:--------------|
| [MOD]  | `pyproject.toml` | ty config güncelle (şu an tüm kurallar ignore) |
| [MOD]  | `.pre-commit-config.yaml` | ty hook ekle |
| [MOD]  | `src/schemas/position.py` | Frozen Pydantic modeller |
| [MOD]  | `src/schemas/trade.py` | Frozen Pydantic modeller |
| [MOD]  | `src/schemas/ai.py` | Frozen Pydantic modeller |
| [MOD]  | `src/core/ai_service.py` | match/case + tuple Result pattern |
| [MOD]  | `src/core/portfolio_manager.py` | match/case + frozen model kullanımı |
| [MOD]  | `requirements.txt` | ty versiyon güncelle |

## 4. 👣 Execution Sequence

### Aşama 1: ty Yapılandırması (En Yüksek Etki)
1. `pyproject.toml` `[tool.ty]` bölümünü temizle
2. Kritik kuralları etkinleştir (unresolved-import, invalid-argument-type, vb.)
3. `error-on-warning` aktif et
4. Pre-commit hook ekle

### Aşama 2: Frozen Pydantic Modelleri
1. `Position`, `ExitPlan`, `TradeHistoryEntry` modellerini `frozen=True` yap
2. Mutation'lar için `.model_copy(update={...})` kullan
3. Mevcut `pos["key"] = value` syntax'ını `.model_copy()` ile değiştir

### Aşama 3: Tuple-Based Result Pattern (Özel Sınıf Yerine)
1. `tuple[dict, None] | tuple[None, str]` pattern kullan
2. Try/except yerine tuple return
3. Match ile işleme

```python
# Basit, hafif, Pythonic — guerrilla felsefesine uygun
def parse_ai_response(response: str) -> tuple[dict, None] | tuple[None, str]:
    try:
        data = json.loads(response)
        return data, None
    except Exception as e:
        return None, str(e)

# Match ile işleme
match parse_ai_response(response):
    case (data, None):
        process(data)
    case (None, error):
        logger.warning(f"Error: {error}")
```

### Aşama 4: match/case (Python 3.10+)
1. `signal` kontrolündeki if/elif chain'lerini match/case'e çevir
2. `direction` ve `regime` kontrolünde uygula

### Aşama 5: Doğrulama
1. `ty check src/` — zero errors
2. `ruff check src/` — zero errors
3. `python3 src/main.py` — runtime test

## 5. Blast Radius

| Değişiklik | Etkilenen Modüller | Risk |
|------------|-------------------|------|
| ty config | Tüm modüller (statik analiz) | Düşük — sadece konfigürasyon |
| Frozen models | portfolio_manager, account_service | Yüksek — mutation pattern değişir |
| Tuple Result | ai_service, deepseek_api | Düşük — sadece return type değişir |
| match/case | main.py, portfolio_manager | Düşük — sadece syntax değişikliği |

## 6. ✅ Verification Standards

- [ ] `ty check src/` — zero errors
- [ ] `ruff check src/` — zero errors
- [ ] `python3 src/main.py` — 1 cycle sorunsuz çalışır
- [ ] Frozen model test: `Position(...).model_copy(update={...})` çalışır
- [ ] Tuple Result test: `parse_ai_response()` doğru tuple döndürür

## 7. ty Yapılandırma Detayı

### Mevcut Sorun
```toml
# Şu an — tüm kurallar ignore, hiçbir hata yakalanmıyor
[tool.ty]
overrides = [
    { rules = { "invalid-parameter-default" = "ignore", "member-access" = "ignore", ... } }
]
```

### Önerilen Yapılandırma
```toml
[tool.ty]
# Hedef: Python 3.10+
[tool.ty.environment]
python-version = "3.10"
python-platform = "linux"

# Kaynak dizinler
[tool.ty.src]
include = ["src", "config"]

# Kritik kurallar (error seviyesinde)
[tool.ty.rules]
# Import hataları
unresolved-import = "error"
unresolved-reference = "error"

# Tip hataları
invalid-argument-type = "error"
invalid-assignment = "error"
invalid-return-type = "error"
missing-argument = "error"
call-non-callable = "error"

# Operatör hataları
unsupported-operator = "error"
not-iterable = "error"
not-subscriptable = "error"

# Sınıf/Protocol hataları
invalid-method-override = "error"

# Olası hatalar (warn seviyesinde)
possibly-missing-attribute = "warn"
possibly-unresolved-reference = "warn"

# Trading-kritik
division-by-zero = "warn"

# Terminal ayarları
[tool.ty.terminal]
error-on-warning = true
output-format = "full"

# Test dosyaları için daha gevşek kurallar
[[tool.ty.overrides]]
include = ["tests/**"]
[tool.ty.overrides.rules]
possibly-unresolved-reference = "ignore"
```

### Pre-commit Hook
```yaml
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: ty
      name: ty type check
      entry: ty check
      language: system
      types: [python]
      pass_filenames: false
```

## 8. Frozen Pydantic Models

### Neden Frozen?
Rust'ta değişkenler varsayılan olarak immutable'dır. Pydantic'te `frozen=True` ile aynı davranışı elde ederiz:

```python
# Şu an — mutable (güvenli değil)
position["margin_usd"] = 10.0

# Önerilen — immutable (güvenli)
position = position.model_copy(update={"margin_usd": 10.0})
```

### Uygulama
```python
# src/schemas/position.py
class Position(BaseModel):
    model_config = ConfigDict(frozen=True)  # Immutable

    symbol: str
    direction: str
    # ... diğer alanlar

# Mutation örneği
new_position = old_position.model_copy(update={
    "current_price": 75.50,
    "unrealized_pnl": 0.25,
})
```

## 9. match/case Örnekleri

### Şu an (if/elif chain)
```python
if signal == "buy_to_enter":
    # ...
elif signal == "sell_to_enter":
    # ...
elif signal == "close_position":
    # ...
elif signal == "hold":
    # ...
```

### Önerilen (match/case)
```python
match signal:
    case "buy_to_enter":
        execute_long_entry(coin, trade)
    case "sell_to_enter":
        execute_short_entry(coin, trade)
    case "close_position":
        close_position(coin, trade)
    case "hold":
        pass
    case _:
        logger.warning(f"Unknown signal: {signal}")
```

## 10. Öncelik Sırası

| Öncelik | Aşama | Efor | Impact |
|---------|-------|------|--------|
| P0 | Aşama 1 (ty config) | 30 dk | En yüksek — statik tip kontrolü |
| P1 | Aşama 2 (Frozen models) | 1 saat | Yüksek — immutability |
| P2 | Aşama 3 (Tuple Result) | 30 dk | Orta — hata yönetimi |
| P3 | Aşama 4 (match/case) | 20 dk | Düşük — sadece syntax |
