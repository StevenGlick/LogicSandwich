# Resume Point

## Current State: Fully Wired, Ready for Optimization

The system is architecturally complete. All components talk to each other. Two LLM backends work (Ollama local + Claude API). A Tkinter launcher makes it usable without memorizing CLI flags.

### What Works Right Now
- **Reset DB** button in launcher to start fresh from seed templates
- **Pulse engine** with full quality pipeline: TransitiveFilter + DomainClassifier + Bridge awareness
- **Verification** with per-entry progress output and T/F/M verdicts (Mistral confirmed working)
- **Query engine** with bridge expansion and bridge-enriched LLM synthesis
- **Two LLM backends**: Ollama (local, free) and Claude API (fast, high quality)
- **Retry logic** on Claude API (429/529 overloaded → backoff + retry)
- **Reference test**: `test_water.py` passes (140 seed → 472 entries, 39 blocked, 14 bridge types)

### Environment
- Python 3.14 on Windows 11
- Ollama running locally (Mistral recommended for 12GB VRAM, also have llama3.2)
- Claude API key set via `ANTHROPIC_API_KEY` env var (System → Advanced → Environment Variables)
- All scripts require `py -X utf8` for Unicode box-drawing characters

---

## File Inventory

### Source Code (`src/`)

| File | Lines | Role |
|------|-------|------|
| `launcher.py` | ~830 | Tkinter GUI — 7 tabs, backend switching, Reset DB, live console |
| `curriculum_pulse.py` | ~920 | Main brain — curriculum engine + pulse + Flask dashboard |
| `pulse_production.py` | ~470 | Standalone pulse engine (no curriculum) |
| `ingest.py` | ~890 | Knowledge ingestion (textbooks, Wikipedia, Wikidata, ConceptNet) |
| `query.py` | ~485 | Query engine — NL questions, graph walk, path finding, bridges |
| `db.py` | ~310 | Shared KnowledgeDB — canonical 14-column schema, on-demand tables |
| `llm.py` | ~460 | Shared LLM — LLMBase/OllamaLLM/ClaudeLLM + factory |
| `bridges.py` | ~850 | Cross-domain bridge system — 24 types, 96 seed bridges |
| `predicate_rules.py` | ~277 | Predicate transparency — blocks invalid transitive chains |
| `domain_scope.py` | ~900 | Domain awareness — blocks cross-domain nonsense |
| `optimize.py` | ~850 | DB optimization — normalize, hierarchy, centrality, FTS5 |
| `test_water.py` | ~450 | Integration test — full pipeline on water domain |

### Seed Data (`ClaudeInput/three_layer_pulse_system/`)

| File | Contents |
|------|----------|
| `ai_kb_big.json` | 150 AI/ML seed entries |
| `water_seed.json` | 140 water domain seed entries |

### Key Project Files

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Full project reference (architecture, conventions, integration points) |
| `RULES.md` | Project rules skeleton (user to fill out) |
| `UpdateList.md` | Tracked issues and improvements (many resolved, some remaining) |

---

## Architecture Summary

```
┌─────────────────────────────────────────────┐
│              launcher.py (GUI)              │
│  Backend: [ollama|claude]  Model: [dropdown] │
│  Reset DB | Tabs: Brain/Pulse/Ingest/Query   │
└──────────────┬──────────────────────────────┘
               │
    ┌──────────┼──────────────────┐
    │          │                  │
    ▼          ▼                  ▼
┌────────┐ ┌──────────┐ ┌──────────────┐
│ Pulse  │ │  Ingest  │ │    Query     │
│ Engine │ │  Engine  │ │    Engine    │
└───┬────┘ └────┬─────┘ └──────┬───────┘
    │           │               │
    │  ┌────────┴────────┐      │
    │  │  Quality Filters │      │
    │  │  1. DomainScope  │      │
    │  │  2. TransFilter  │      │
    │  │  3. BridgeRescue │      │
    │  └─────────────────┘      │
    │                           │
    ▼                           ▼
┌────────────────────────────────────┐
│         db.py (KnowledgeDB)        │
│     curriculum.db (SQLite)         │
├────────────────────────────────────┤
│  llm.py: OllamaLLM | ClaudeLLM    │
│  bridges.py: BridgeSystem          │
│  predicate_rules.py: TransFilter   │
│  domain_scope.py: DomainClassifier │
└────────────────────────────────────┘
```

---

## What Was Completed (Sessions 1-3)

### Session 1: Setup
- Fresh project setup, CLAUDE.md written, folders created
- Python + VS Code installed, dependencies installed

### Session 2: Import + Refactor
- All 12 source files imported to `src/`
- Extracted shared `db.py` (KnowledgeDB) and `llm.py` (OllamaLLM) from 5 duplicate copies
- All 4 scripts updated to use shared modules (~1050 lines removed)
- Single canonical 14-column schema with on-demand supporting tables
- Smoke tests passing, reference test confirmed (140→472)

### Session 3: Features + Wiring
- **Launcher** (`launcher.py`): 7-tab Tkinter GUI with live console, file browsers, process management
- **Claude API backend**: LLMBase/OllamaLLM/ClaudeLLM hierarchy, `create_llm()` factory
- **Backend switching in launcher**: dropdown, API key field, model list swapping
- **Reset DB button**: delete + re-seed from template dialog
- **Verify_entry prompt rewrite**: robust parsing for T/F/M, handles markdown formatting
- **Retry logic**: 429/529 overloaded → backoff 5/10/15/20s, 4 attempts max
- **Per-entry progress output**: `verifying [1/5] term→pred(obj)... T (2.1s)`
- **Verifier M-entry recovery**: picks up stranded M entries from simulation mode
- **TransitiveFilter → pulse_production.py**: blocks 5.3x garbage cascade
- **Bridges → query.py**: search expansion + synthesis enrichment
- **Bridge-aware pulse engines**: cross-domain checks existing bridges, creates emergent candidates

---

## What's Remaining

### Optimization (from CLAUDE.md)
1. **Hash indexing** — FNV-1a 64-bit columns for fast lookups (spec in `ClaudeInput/hash_indexing_spec.md`)
2. **Cross-domain specificity scoring** — Filter generic shared properties (>50% frequency)
3. **Predicate rule learning** — Auto-learn transparency rules from LLM verification patterns

### Known Issues (from UpdateList.md)
- **Bare `except:` blocks** — 15+ instances silently swallowing real bugs
- **Substring matching false positives** — "ion" matches "animation" in domain classifiers
- **Phantom domains** — 12 domains referenced but never classified into
- **No FTS5 usage in query.py** — uses LIKE instead of the index optimize.py builds
- **No synonym expansion** — optimize.py builds synonyms, query doesn't use them

### Nice to Have
- GitHub publish (user wants this, `gh` CLI available)
- Flask dashboard in curriculum_pulse.py (exists but untested since refactor)
- RULES.md content (user to fill out)

---

## Quick Start

```bash
# Launch the GUI
py -X utf8 src/launcher.py

# Or run directly:
py -X utf8 src/pulse_production.py --model mistral --seed-json ClaudeInput/three_layer_pulse_system/water_seed.json --cycles 5

# With Claude API:
set ANTHROPIC_API_KEY=sk-ant-...
py -X utf8 src/pulse_production.py --backend claude --seed-json ClaudeInput/three_layer_pulse_system/ai_kb_big.json --cycles 10

# Query the knowledge base:
py -X utf8 src/query.py --interactive --model mistral

# Run reference test:
py -X utf8 src/test_water.py
```
