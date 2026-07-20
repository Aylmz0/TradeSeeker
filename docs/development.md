# Development

## Contributing

### Code Style
- **Formatter**: ruff (configured in pyproject.toml)
- **Linter**: ruff + ty (Astral, Rust-based)
- **Docstrings**: Google-style (`Args:`, `Returns:`, `Raises:`)
- **Type hints**: Required for all public functions

### Pre-commit Hooks
```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Running Checks
```bash
# Lint
ruff check src/

# Type check (ty beta — error-on-warning = false)
ty check src/

# Format
ruff format src/
```

## Docstring Convention

Google-style docstrings are used throughout:

```python
def calculate_rsi(
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Calculate RSI using Wilder's smoothing method.

    Args:
        close: Array of closing prices.
        period: RSI period (default: 14).

    Returns:
        Array of RSI values (0-100).

    Raises:
        ValueError: If period < 2.
    """
```

### Docstring Coverage
- **452 total functions**
- **87 with full Google docstrings** (Args: + Returns:)
- **~100 with partial docstrings**
- **~265 without docstrings**

pdoc generates API docs from existing docstrings. Improve coverage incrementally as you touch files.

## How to Document New Functions

When you add or modify a function, follow this checklist:

### Step 1: Write the Docstring
```python
def my_new_function(param1: str, param2: int) -> bool:
    """One-line summary of what this function does.

    Args:
        param1: What param1 is for.
        param2: What param2 is for.

    Returns:
        True if successful, False otherwise.

    Raises:
        ValueError: If param2 is negative.
    """
```

### Step 2: Run Checks
```bash
ruff check src/your_module.py
ty check src/your_module.py
```

### Step 3: Regenerate API Docs
```bash
.venv/bin/python -m pdoc src -o docs/api -d google --no-include-undocumented --no-show-source
```

### Step 4: Update Knowledge Graph
```bash
graphify update ./src
```

### Step 5: Update Callflow (if architecture changed)
```bash
graphify export callflow-html --output docs/arch.html --max-sections 12
```

## How to Update Knowledge Graph (graphify)

### After Every Significant Code Change
```bash
# 1. Update the graph (AST extraction, no LLM needed)
graphify update ./src

# 2. Re-cluster communities (if modules added/removed)
graphify cluster-only .

# 3. Update lessons learned
graphify reflect --out docs/LESSONS.md
```

### If You Want Better Community Names
```bash
# Requires LLM API key (GEMINI_API_KEY, OPENAI_API_KEY, etc.)
graphify label . --backend gemini
```

### Query the Graph (for architecture questions)
```bash
# Find relationships
graphify query "how does X connect to Y?"

# Find path between modules
graphify path "ModuleA" "ModuleB"

# Explain a module
graphify explain "ModuleName"
```

### Save Important Discoveries
```bash
graphify save-result \
  --question "How does the new feature work?" \
  --answer "Explanation..." \
  --nodes "ModuleA" "ModuleB" \
  --outcome useful
```

## How to Keep Documentation Updated

### When You Add a New Module
1. Write docstrings (Google-style)
2. Run `graphify update ./src`
3. Run `graphify cluster-only .`
4. Run `.venv/bin/python -m pdoc src -o docs/api -d google --no-include-undocumented --no-show-source`
5. Update `docs/architecture.md` — add module to the table
6. Update `docs/index.md` — add to Core Modules section

### When You Change Architecture
1. Run `graphify update ./src`
2. Run `graphify export callflow-html --output docs/arch.html --max-sections 12`
3. Update `docs/architecture.md` — update diagrams and descriptions
4. Update `docs/configuration.md` — if new Config options added

### When You Change Config Options
1. Update `src/schemas/config.py`
2. Update `docs/configuration.md` — add new row to the table
3. Update `.env.example` — add new variable

### Every Sprint / Major Release
```bash
graphify update ./src
graphify cluster-only .
graphify reflect --out docs/LESSONS.md
.venv/bin/python -m pdoc src -o docs/api -d google --no-include-undocumented --no-show-source
graphify export callflow-html --output docs/arch.html --max-sections 12
```

## Project Structure

```
TradeSeeker/
├── src/                    # Source code
│   ├── main.py            # Entry point
│   ├── utils.py           # Utilities
│   ├── core/              # Core trading logic
│   ├── ai/                # AI/LLM integration
│   ├── schemas/           # Pydantic models
│   ├── services/          # External services
│   └── web/               # Admin dashboard
├── scripts/               # Utility scripts
├── data/                  # Runtime data (git-ignored)
├── docs/                  # Documentation
│   ├── index.md           # Entry point
│   ├── architecture.md    # System design
│   ├── configuration.md   # Config reference
│   ├── operations.md      # Runbooks
│   ├── development.md     # This file
│   ├── arch.html          # Callflow diagram
│   ├── tree.html          # Module hierarchy
│   ├── LESSONS.md         # Knowledge base
│   ├── api/               # pdoc API docs (HTML)
│   └── plans/             # Working plans
├── graphify-out/          # Knowledge graph
│   ├── wiki/              # 123 community articles
│   ├── memory/            # Q&A history
│   ├── graph.json         # The graph
│   └── GRAPH_REPORT.md    # Community map
├── .env                   # Configuration (git-ignored)
├── .env.example           # Configuration template
├── pyproject.toml         # Project config
└── requirements.txt       # Dependencies
```

## Documentation Map

| Document | Purpose | Updated When |
|----------|---------|--------------|
| `docs/index.md` | Entry point, quick start | New modules added |
| `docs/architecture.md` | System design, data flow | Architecture changes |
| `docs/configuration.md` | All Config options | Config options change |
| `docs/operations.md` | Runbooks, troubleshooting | New procedures |
| `docs/development.md` | Contributing guide | Process changes |
| `docs/api/` | API reference | Code changes |
| `docs/arch.html` | Callflow visualization | Architecture changes |
| `docs/tree.html` | Module hierarchy | Modules added/removed |
| `docs/LESSONS.md` | Knowledge base | Q&A sessions |
| `graphify-out/wiki/` | Community articles | Graph updates |

## Git Workflow

### Commit Messages
- Use imperative mood: "Add feature" not "Added feature"
- Keep under 72 characters
- Reference issues when applicable

### Branch Naming
- `feature/description` — new features
- `fix/description` — bug fixes
- `refactor/description` — code refactoring

## Architecture Decision Records (ADRs)

When making significant architectural decisions, document them:

```bash
# Create ADR
mkdir -p docs/decisions
cat > docs/decisions/YYYY-MM-DD-decision-title.md << 'EOF'
# Decision: Title

## Status
Accepted

## Context
What is the issue?

## Decision
What did we decide?

## Consequences
What are the results?
EOF
```

Move finalized plans from `docs/plans/` to `docs/decisions/` when they represent permanent decisions.
