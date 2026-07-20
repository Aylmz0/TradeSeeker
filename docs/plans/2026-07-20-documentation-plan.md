# TradeSeeker Dokümantasyon Planı

**Tarih:** 2026-07-20
**Araçlar:** pdoc (API referansı) + graphify (bilgi grafı + mimari dokümanlar)
**Dil:** İngilizce
**Kapsam:** Tüm codebase (src/, scripts/, docs/, config dosyaları)

---

## 1. Mevcut Durum Analizi

### Codebase Yapısı (35 Python dosyası, 89K kelime)
```
src/
  ai/           deepseek_api.py, enhanced_context_provider.py, prompt_json_builders.py,
                prompt_json_schemas.py, prompt_json_utils.py
  core/         account_service.py, ai_service.py, backtest.py, cache_manager.py,
                constants.py, data_engine.py, indicators.py, log_config.py,
                market_data.py, performance_monitor.py, portfolio_manager.py,
                regime_detector.py, strategy_analyzer.py
  core/schemas/ alignment.py
  schemas/      ai.py, config.py, position.py, trade.py
  services/     alert_system.py, binance.py, ml_service.py
  web/          admin_server_flask.py
  main.py, utils.py
scripts/        backtest_runner.py, generate_performance_report.py, forensic_audit.py,
                llm_benchmark.py, replay_trade.py (tahmini)
docs/           AI_PROMPTS_REFERENCE.md, guide.md, plans/, PDF'ler
config/         .env, .env.example
```

### Graphify Mevcut Durumu
- `graphify-out/` mevcut: graph.json (2.4MB), graph.html (2MB), GRAPH_REPORT.md
- **2150 node, 4801 edge, 114 community** (37'si isimlendirilmiş, 77'si "Community N")
- 69 dosya taranmış, ~89,241 kelime
- 97% EXTRACTED, 3% INFERRED, 0% AMBIGUOUS
- Import cycle: yok
- 18 izole node (belge boşluğu)

### God Nodes (en çok bağlantılı)
1. `PortfolioManager` — 113 edge (merkezi hub)
2. `Config` — 61 edge (tüm modüllere bağlanıyor)
3. Anahtar fonksiyonlar: `an()`, `sn()`, `f()`, `vn()`, `ds()`, `yi()`, `te()`

### Graphify-Out Topluluk Haritası (37 isimlendirilmiş)
| Topluluk | Modül | Cohesion |
|----------|-------|----------|
| Forensic & Debug Scripts | scripts/ | 0.06 |
| Charts Library (internal) | charts/ | 0.07 |
| Binance Client & Account | services/binance.py | 0.08 |
| Connection & Account Service | core/account_service.py | 0.07 |
| Cache Manager | core/cache_manager.py | 0.05 |
| Flask Admin Dashboard | web/admin_server_flask.py | 0.05 |
| Alert System | services/alert_system.py | 0.07 |
| Config & DeepSeek API | schemas/config.py + ai/deepseek_api.py | 0.10 |
| Portfolio Manager Core | core/portfolio_manager.py | 0.05 |
| AI Service (Decision Engine) | core/ai_service.py | 0.07 |
| Main Orchestrator Loop | main.py | 0.08 |
| Portfolio Entry Preconditions | core/portfolio_manager.py (entry) | 0.07 |
| Backtest Engine & Replay | core/backtest.py | 0.08 |
| Dashboard Frontend | static/app.js | 0.15 |
| AI Decision Schemas | schemas/ai.py | 0.09 |
| ML Service & Training | services/ml_service.py | 0.09 |
| Market Data Engine | core/market_data.py | 0.09 |
| Technical Indicators | core/indicators.py | 0.08 |
| Regime Detector | core/regime_detector.py | 0.18 |
| Performance Monitor | core/performance_monitor.py | 0.08 |
| Strategy Analyzer | core/strategy_analyzer.py | — |
| Data Engine (SQLite) | core/data_engine.py | 0.13 |
| Enhanced Context Provider | ai/enhanced_context_provider.py | — |
| Prompt JSON Builders | ai/prompt_json_builders.py | 0.13 |
| Prompt Schemas & Utils | ai/prompt_json_schemas.py | 0.11 |
| Trade/Position Schemas | schemas/trade.py, position.py | — |
| Docs & Guides | docs/ | 0.15 |
| LLM Benchmark Rules | scripts/llm_benchmark.py | 0.13 |

---

## 2. Docstring Kapsam Analizi

**Toplam fonksiyon:** 452
**Tam Google docstring (Args: + Returns:):** 87 (%19)
**Kısmi docstring:** ~100
**Docstring yok:** ~265

pdoc mevcut docstring'lerden otomatik API sayfası üretir. Eksik docstring'ler → pdoc'ta boş görünür. **Docstring iyileştirmesi bu planın parçası değil** (ayrı bir çalışma konusu), ama plan NOT olarak eklendi.

---

## 3. Kullanılacak Graphify Komutları (Detaylı)

### Tier 1 — Zorunlu (Dokümantasyon Temeli)

#### 3.1 `graphify query "..."`
**Amaç:** Graph üzerinden soru-cevap ile mimari ilişki keşfi.
**Kullanım Alanları:**
- Modüller arası bağımlılıkları bulma
- "What calls what?" soruları
- Veri akışı haritası çıkarma
- Dead code tespit (bağlantısız node'lar)

**Örnek sorgular:**
```bash
graphify query "what are the main modules and how do they connect to each other?"
graphify query "how does data flow from main.py through PortfolioManager to AccountService?"
graphify query "what modules depend on Config and what do they use from it?"
graphify query "which functions are called by the main orchestrator loop?"
graphify query "how does the AI decision flow work from DeepSeekAPI to trade execution?"
graphify query "what are all the entry points to PortfolioManager?"
graphify query "which modules use MarketData and what indicators do they fetch?"
graphify query "how does regime detection affect trading decisions?"
graphify query "what is the relationship between StrategyAnalyzer and the AI prompt builders?"
graphify query "which modules have the most incoming edges (most depended upon)?"
```

**Çıktı kullanımı:** `docs/architecture.md` yazımı için ham veri.

#### 3.2 `graphify path "ComponentA" "ComponentB"`
**Amaç:** İki node arasındaki ilişki yolunu gösterir.
**Kullanım Alanları:**
- Veri akışı diyagramları (main → PM → SA → AS → Binance)
- Çağrı zincirleri
- Bağımlılık zincirleri

**Örnek sorgular:**
```bash
graphify path "main" "AccountService"
graphify path "main" "BinanceOrderExecutor"
graphify path "Config" "PortfolioManager"
graphify path "MarketData" "DeepSeekAPI"
graphify path "StrategyAnalyzer" "execute_order"
graphify path "DataEngine" "PerformanceMonitor"
graphify path "RegimeDetector" "portfolio_manager"
graphify path "EnhancedContextProvider" "deepseek_api"
```

**Çıktı kullanımı:** Mimari akış diyagramları, veri akışı haritaları.

#### 3.3 `graphify explain "ModuleName"`
**Amaç:** Belirli bir modül/componenet'in ne yaptığını açıklar.
**Kullanım Alanları:**
- Modül dokümanları (her modül için "ne yapar" açıklaması)
- Yeni başlayanlar için rehber
- Kod değişikliğinde referans

**Örnek sorgular:**
```bash
graphify explain "PortfolioManager"
graphify explain "StrategyAnalyzer"
graphify explain "MarketData"
graphify explain "AccountService"
graphify explain "AIService"
graphify explain "DeepSeekAPI"
graphify explain "RegimeDetector"
graphify explain "Indicators"
graphify explain "PerformanceMonitor"
graphify explain "CacheManager"
graphify explain "DataEngine"
graphify explain "AlertSystem"
graphify explain "BinanceFuturesClient"
graphify explain "Config"
graphify explain "BacktestEngine"
graphify explain "MLService"
graphify explain "EnhancedContextProvider"
graphify explain "PromptJSONBuilders"
```

**Çıktı kullanımı:** `docs/architecture.md` modül açıklamaları, `docs/modules/` alt sayfaları.

#### 3.4 `graphify export callflow-html`
**Amaç:** İnteraktif çağrı akışı HTML'i üretir.
**Kullanım Alanları:**
- Mimari görselleştirme (modüller arası akış)
- Yeni başlayanlar için görsel rehber
- Kod değişikliğinde referans

**Komut:**
```bash
graphify export callflow-html --output docs/arch.html --max-sections 12
```

**Çıktı kullanımı:** `docs/arch.html` olarak yerleştirilir, `docs/index.md`'den link verilir.

#### 3.5 `graphify wiki`
**Amaç:** Graph'tan markdown wiki üretir (agent-crawlable).
**Kullanım Alanları:**
- Modül sayfaları (otomatik üretilir)
- İlişki haritaları (otomatik üretilir)
- Topluluk sayfaları (her topluluk için)
- AI'ın codebase'i taraması için

**Komut:**
```bash
graphify wiki --output docs/wiki
```

**Çıktı kullanımı:** `docs/wiki/` altına yerleştirilir. pdoc API referansını tamamlar.

### Tier 2 — Yararlı (Bilgi Bankası + Güncelleme)

#### 3.6 `graphify save-result`
**Amaç:** Q&A sonuçlarını kaydeder (work memory).
**Kullanım Alanları:**
- Mimari keşif sırasında bulguları kaydetme
- "Bu sorgu faydalıydı/değildi/düzeltilmeli" geri bildirimi
- reflect komutu için ham veri

**Komut:**
```bash
graphify save-result --question "How does entry validation work?" \
  --answer "PortfolioManager._validate_entry_signal checks position limits, cooldown, directional bias, and regime before allowing entry." \
  --nodes "PortfolioManager" "_validate_entry_signal" "Config" \
  --outcome useful
```

**Çıktı kullanımı:** `graphify-out/memory/` altına kaydedilir, `graphify reflect` tarafından derlenir.

#### 3.7 `graphify reflect`
**Amaç:** Kaydedilmiş Q&A sonuçlarını derler, lesson'ları üretir.
**Kullanım Alanları:**
- Bilgi bankası oluşturma
- Tekrar eden soruları tespit etme
- `docs/LESSONS.md` üretimi
- Topluluk bazlı lesson gruplandırma

**Komut:**
```bash
graphify reflect --out docs/LESSONS.md
graphify reflect --graph graphify-out/graph.json  # topluluk bazlı lesson
```

**Çıktı kullanımı:** `docs/LESSONS.md` — bilgi bankası, tekrar eden sorular, düzeltilmesi gereken bulgular.

#### 3.8 `graphify update ./src`
**Amaç:** Kod değişince graph'i günceller.
**Kullanım Alanları:**
- Yeni modül eklendiğinde
- Fonksiyon imzası değiştiğinde
- Import bağımlılıkları değiştiğinde

**Komut:**
```bash
graphify update ./src
graphify update ./src --no-cluster  # sadece raw AST grafiği, yeniden cluster yok
graphify update ./src --force       # yeni graph eskisini overwrite eder
```

**Not:** graphify hook ile otomatik çalıştırılabilir (post-commit).

#### 3.9 `graphify cluster-only`
**Amaç:** Mevcut graph'i yeniden cluster'lar.
**Kullanım Alanları:**
- Topluluk isimlerini yenileme
- Yeni modüller eklendikten sonra topluluk yapısını tazeleme
- Resolution ayarı ile daha ince/büyük topluluklar

**Komut:**
```bash
graphify cluster-only .                              # varsayılan
graphify cluster-only . --resolution 1.5             # daha küçük topluluklar
graphify cluster-only . --backend gemini             # AI ile isimlendirme
graphify cluster-only . --max-concurrency 16         # paralel
```

**Not:** `.graphify_labels.json` güncellenir → GRAPH_REPORT.md yeniden üretilir.

#### 3.10 `graphify label`
**Amaç:** Toplulukları AI ile yeniden isimlendirir.
**Kullanım Alanları:**
- "Community 38" gibi isimsiz topluluklara anlam verme
- Graph raporunu okunabilir hale getirme

**Komut:**
```bash
graphify label . --backend gemini
graphify label . --backend openai --model gpt-4o
```

### Tier 3 — Opsiyonel (İleride Kullanılabilir)

#### 3.11 `graphify --wiki` (variant)
`graphify wiki` ile aynı, wiki markdown üretir. pdoc ile birlikte tam doküman seti oluşturur.

#### 3.12 `graphify export callflow-html --output`
Farklı output konumu. docs/ altına yerleştirme için kullanılır.

#### 3.13 `graphify hook install`
Post-commit + post-checkout hook'u kurar → graph otomatik güncellenir.
```bash
graphify hook install  # her commit'te graph tazelenir
```

#### 3.14 `graphify extract` (headless LLM extraction)
CI/CD ortamında LLM kullanımı (API key gerektirir). Şu an için gerekli değil ama ileride automated doc generation için kullanılabilir.

---

## 4. pdoc Kurulumu ve Kullanımı

### 4.1 Kurulum
```bash
pip install pdoc3
```

### 4.2 Temel Kullanım
```bash
# Tüm src/ dizininden API docs üret
pdoc src/ --output-dir docs/api --force

# Tek modül
pdoc src/core/portfolio_manager.py --output-dir docs/api --force

# HTML (varsayılan)
pdoc src/ --output-dir docs/api --html --force

# Markdown
pdoc src/ --output-dir docs/api --force  # varsayılan markdown
```

### 4.3 pdoc Konfigürasyonu
pdoc Google-style docstring'leri otomatik parse eder:
- `Args:` → parametre açıklamaları
- `Returns:` → dönüş değeri
- `Raises:` → istisnalar
- `Example:` → kullanım örnekleri

Mevcut 87 tam docstring → pdoc'ta otomatik API sayfası üretir.
Eksik docstring'ler → pdoc'ta boş alan olarak görünür (docstring iyileştirmesi ayrı konu).

### 4.4 pdoc Çıktı Yapısı
```
docs/api/
  index.html              # ana sayfa (tüm modüller)
  core/
    portfolio_manager.html
    strategy_analyzer.html
    market_data.html
    account_service.html
    ai_service.html
    indicators.html
    regime_detector.html
    performance_monitor.html
    cache_manager.html
    data_engine.html
    backtest.html
    log_config.html
    constants.html
  ai/
    deepseek_api.html
    enhanced_context_provider.html
    prompt_json_builders.html
    prompt_json_schemas.html
    prompt_json_utils.html
  schemas/
    config.html
    ai.html
    position.html
    trade.html
  services/
    alert_system.html
    binance.html
    ml_service.html
  web/
    admin_server_flask.html
```

---

## 5. Doküman Yapısı (Oluşturulacak Dosyalar)

```
docs/
  index.md                          # ana sayfa (giriş, hızlı rehber)
  architecture.md                   # mimari tasarım (graphify query/path/explain ile)
  arch.html                         # interaktif çağrı akışı (graphify export callflow-html)
  configuration.md                  # Config + .env referansı ( elle yazılır, basit)
  operations.md                     # runbook: 35-cycle reset, live mod, troubleshooting
  development.md                    # katkı rehberi, docstring standartı
  LESSONS.md                        # graphify reflect ile üretilen bilgi bankası
  modules/                          # modül bazlı dokümanlar (graphify explain ile)
    portfolio_manager.md
    strategy_analyzer.md
    market_data.md
    account_service.md
    ai_service.md
    indicators.md
    regime_detector.md
    performance_monitor.md
    cache_manager.md
    data_engine.md
    backtest.md
    alert_system.md
    binance.md
    ml_service.md
    deepseek_api.md
    prompt_json_builders.md
    config.md
    schemas.md
  wiki/                             # graphify wiki çıktısı (otomatik)
    ...                             # graphify tarafından üretilir
  api/                              # pdoc API referansı (otomatik)
    index.html
    core/...
    ai/...
    schemas/...
    services/...
    web/...
  decisions/                        # ADR'ler (finalize edilen kararlar)
    2026-07-18-744-cycle-analysis.md  # mevcut → taşı
    2026-07-17-system-fixes.md        # mevcut → taşı
    ...                              # yeni ADR'ler
  plans/                            # aktif çalışma planları (mevcut, korunur)
    2026-07-20-documentation-plan.md  # bu dosya
    ...
  reference/                        # mevcut referanslar
    AI_PROMPTS_REFERENCE.md
    technical_audit.md
    guide.md
```

---

## 6. Uygulama Adımları

### Faz A — Graph Verilerini Toplama (graphify)

**Adım 1.1:** Mevcut graph'i sorgula — mimari ilişki keşfi
```bash
# Ana modül haritası
graphify query "what are the main modules and how do they connect?"
graphify query "how does data flow from main.py to trade execution?"
graphify query "what modules depend on Config?"

# Veri akışı
graphify path "main" "AccountService"
graphify path "main" "BinanceOrderExecutor"
graphify path "Config" "PortfolioManager"
graphify path "MarketData" "DeepSeekAPI"
graphify path "StrategyAnalyzer" "execute_order"

# Modül açıklamaları
graphify explain "PortfolioManager"
graphify explain "StrategyAnalyzer"
graphify explain "MarketData"
graphify explain "AccountService"
graphify explain "AIService"
graphify explain "RegimeDetector"
graphify explain "Indicators"
graphify explain "PerformanceMonitor"
graphify explain "DeepSeekAPI"
graphify explain "Config"
graphify explain "CacheManager"
graphify explain "DataEngine"
graphify explain "BacktestEngine"
graphify explain "MLService"
graphify explain "EnhancedContextProvider"
graphify explain "AlertSystem"
graphify explain "BinanceFuturesClient"
```

**Adım 1.2:** Çağrı akışı üret
```bash
graphify export callflow-html --output docs/arch.html --max-sections 12
```

**Adım 1.3:** Wiki üret
```bash
graphify wiki --output docs/wiki
```

**Adım 1.4:** Toplulukları yeniden isimlendir (isimsiz topluluklar için)
```bash
graphify label . --backend gemini
```

**Adım 1.5:** Bulguları kaydet
```bash
# Mimari keşif sonuçlarını kaydet
graphify save-result --question "How does entry validation work?" \
  --answer "PortfolioManager._validate_entry_signal checks position limits, cooldown, directional bias, and regime." \
  --nodes "PortfolioManager" "_validate_entry_signal" "Config" \
  --outcome useful

# vs. diğer bulgular...
```

**Adım 1.6:** Lessons derle
```bash
graphify reflect --out docs/LESSONS.md
```

### Faz B — API Referansı (pdoc)

**Adım 2.1:** pdoc kur
```bash
pip install pdoc3
```

**Adım 2.2:** API docs üret
```bash
pdoc src/ --output-dir docs/api --force
```

### Faz C — Dokümanları Yaz (graphify çıktılarından beslenerek)

**Adım 3.1:** `docs/index.md` — ana sayfa
- Proje nedir, nasıl çalıştırılır
- Hızlı başlangıç
- Linkler: API docs, mimari, config, wiki

**Adım 3.2:** `docs/architecture.md` — mimari tasarım
- graphify query/path/explain çıktılarından
- Modül haritası (topluluk bazlı)
- Veri akışı diyagramları (path çıktıları)
- God nodes: PortfolioManager (113 edge), Config (61 edge)
- Topluluk haritası (37 isimlendirilmiş topluluk)

**Adım 3.3:** `docs/configuration.md` — Config referansı
- Config class tablosu (schemas/config.py'den)
- .env değişkenleri
- Varsayılan değerler
- Doğrulama kuralları

**Adım 3.4:** `docs/operations.md` — operasyon rehberi
- 35-cycle reset mekanizması
- Live vs Simulation modu
- portfolio_state.json okuma
- Troubleshooting (常见 sorunlar)

**Adım 3.5:** `docs/development.md` — katkı rehberi
- Docstring standartı (Google-style)
- Ty/ruff workflow
- Pre-commit hooks
- Graph güncelleme (graphify update)

**Adım 3.6:** `docs/modules/*.md` — modül bazlı dokümanlar
- graphify explain çıktılarından
- Her modül: ne yapar, bağımlılıklar, ana fonksiyonlar

**Adım 3.7:** `docs/decisions/` — ADR'ler
- Mevcut plans/*.md'den finalize edilenler → decisions/ taşınır

**Adım 3.8:** `docs/LESSONS.md` — bilgi bankası
- graphify reflect ile üretilir
- Tekrar eden sorular, düzeltilmesi gereken bulgular

### Faz D — README Güncelleme

**Adım 4.1:** README.md'ye ekle:
- API docs linki (`docs/api/index.html`)
- Mimari doküman linki (`docs/architecture.md`)
- Wiki linki (`docs/wiki/`)
- Callflow linki (`docs/arch.html`)
- LESSONS linki (`docs/LESSONS.md`)

---

## 7. Graph Güncelleme Stratejisi

### Otomatik Güncelleme (graphify hook)
```bash
graphify hook install  # post-commit + post-checkout
```
Her commit'te graph otomatik güncellenir.

### Manuel Güncelleme
```bash
# Kod değişiminden sonra
graphify update ./src

# Sadece cluster yenile
graphify cluster-only .

# Topluluk isimlerini yenile
graphify label . --backend gemini

# Lessons güncelle
graphify reflect --out docs/LESSONS.md
```

### Güncelleme Zamanlaması
- **Her büyük değişiklikten sonra:** `graphify update ./src`
- **Her sprint sonunda:** `graphify label .` + `graphify reflect`
- **Yeni modül eklendiğinde:** `graphify update ./src` + modül explain + modules/*.md güncelleme

---

## 8. Dosya Kapsam Haritası

| Dosya | Kapsama | Graphify Topluluğu | pdoc |
|-------|---------|-------------------|------|
| src/core/portfolio_manager.py | ★★★★★ | Portfolio Manager Core + Entry Preconditions | ✓ |
| src/core/strategy_analyzer.py | ★★★★★ | Strategy Analyzer | ✓ |
| src/core/market_data.py | ★★★★★ | Market Data Engine | ✓ |
| src/core/account_service.py | ★★★★☆ | Connection & Account Service | ✓ |
| src/core/ai_service.py | ★★★★☆ | AI Service (Decision Engine) | ✓ |
| src/core/indicators.py | ★★★★☆ | Technical Indicators | ✓ |
| src/core/regime_detector.py | ★★★☆☆ | Regime Detector | ✓ |
| src/core/performance_monitor.py | ★★★☆☆ | Performance Monitor | ✓ |
| src/core/cache_manager.py | ★★★☆☆ | Cache Manager | ✓ |
| src/core/data_engine.py | ★★★☆☆ | Data Engine (SQLite) | ✓ |
| src/core/backtest.py | ★★☆☆☆ | Backtest Engine & Replay | ✓ |
| src/core/constants.py | ★★☆☆☆ | Config & DeepSeek API | ✓ |
| src/core/log_config.py | ★★☆☆☆ | (topluluk 68) | ✓ |
| src/ai/deepseek_api.py | ★★★★☆ | Config & DeepSeek API | ✓ |
| src/ai/enhanced_context_provider.py | ★★★☆☆ | Enhanced Context Provider | ✓ |
| src/ai/prompt_json_builders.py | ★★★★☆ | Prompt JSON Builders | ✓ |
| src/ai/prompt_json_schemas.py | ★★★☆☆ | Prompt Schemas & Utils | ✓ |
| src/ai/prompt_json_utils.py | ★★☆☆☆ | Prompt Schemas & Utils | ✓ |
| src/schemas/config.py | ★★★★★ | Config & DeepSeek API | ✓ |
| src/schemas/ai.py | ★★★★☆ | AI Decision Schemas | ✓ |
| src/schemas/position.py | ★★★☆☆ | Trade/Position Schemas | ✓ |
| src/schemas/trade.py | ★★★☆☆ | Trade/Position Schemas | ✓ |
| src/services/alert_system.py | ★★☆☆☆ | Alert System | ✓ |
| src/services/binance.py | ★★★☆☆ | Binance Client & Account | ✓ |
| src/services/ml_service.py | ★★☆☆☆ | ML Service & Training | ✓ |
| src/web/admin_server_flask.py | ★★★☆☆ | Flask Admin Dashboard | ✓ |
| src/main.py | ★★★★★ | Main Orchestrator Loop | ✓ |
| src/utils.py | ★★☆☆☆ | (çeşitli) | ✓ |

---

## 9. Graphify Komut Hızlı Referansı (Dokümantasyon İçin)

```bash
# === TIER 1: Zorunlu ===

# Graph'i sorgula (ilişki keşfi)
graphify query "what are the main modules and how do they connect?"
graphify query "how does data flow from main.py to trade execution?"
graphify query "what modules depend on Config?"
graphify query "what are all the entry points to PortfolioManager?"

# İki modül arası yol (akış diyagramları)
graphify path "main" "AccountService"
graphify path "Config" "PortfolioManager"
graphify path "MarketData" "DeepSeekAPI"

# Modül açıklaması (modül dokümanları)
graphify explain "PortfolioManager"
graphify explain "StrategyAnalyzer"

# Çağrı akışı HTML (mimari görselleştirme)
graphify export callflow-html --output docs/arch.html --max-sections 12

# Wiki üret (markdown, agent-crawlable)
graphify wiki --output docs/wiki

# === TIER 2: Yararlı ===

# Q&A kaydı (bilgi bankası için ham veri)
graphify save-result --question "Q" --answer "A" --nodes Foo Bar --outcome useful

# Lessons derle (bilgi bankası)
graphify reflect --out docs/LESSONS.md
graphify reflect --graph graphify-out/graph.json  # topluluk bazlı

# Graph güncelle (kod değişikliği sonrası)
graphify update ./src
graphify update ./src --force

# Cluster yenile
graphify cluster-only .
graphify cluster-only . --resolution 1.5  # daha küçük topluluklar

# Toplulukları yeniden isimlendir
graphify label . --backend gemini

# === TIER 3: Opsiyonel ===

# Post-commit hook (otomatik graph güncelleme)
graphify hook install
```

---

## 10. Sonraki Adımlar (Onay Bekliyor)

1. **Onayla** → Faz A ile başla (graph query/path/explain + callflow-html + wiki)
2. **Onayla** → Faz B ile devam et (pdoc kurulumu + API docs üretimi)
3. **Onayla** → Faz C ile tamamla (doküman yazımı)
4. **Onayla** → Faz D ile bitir (README güncelleme)

---

## 11. NOT — Docstring İyileştirmesi (Bu Planda Değil)

452 fonksiyonun 287'sinde docstring yok veya eksik. pdoc bunları boş bırakır. Docstring iyileştirmesi ayrı bir çalışma konusudur:
- Öncelik: core/ modülleri (portfolio_manager, strategy_analyzer, market_data, account_service)
- Sonra: ai/ modülleri
- En son: services/, schemas/, web/

Bu plan docstring iyileştirmesini KAPSAMAZ — sadece mevcut docstring'lerden en iyi şekilde yararlanmayı hedefler.
