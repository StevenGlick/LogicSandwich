# LogicSandwich — Three-Layer Pulse System

## Rules
See [RULES.md](./RULES.md) for project rules and guidelines.

---

## What This Is

A knowledge engine inspired by the classical Trivium (Grammar, Logic, Rhetoric) — but not a recreation of it. The Trivium provided the structural intuition: knowledge work naturally separates into layers, and each layer has optimal hardware. The system takes that insight and builds its own thing.

It stores knowledge as subject-predicate-object triples in SQLite, runs symbolic CPU operations to find patterns, and uses a local LLM (Ollama) for verification and decomposition.

| Layer | Inspired By | Geometry | Operation | Hardware |
|-------|-------------|----------|-----------|----------|
| Word | Grammar | Point (0D) | Atomic facts via LLM extraction | GPU |
| Logic | Logic | Line (1D) | Transitive chains, set operations | CPU |
| Idea | Rhetoric | Plane (2D) | Cross-domain synthesis, analogy | CPU |

---

## Architecture

### Five Pulse Operations
Each pulse cycle runs these on the knowledge base:
1. **Merge** — Combine entries sharing subject+predicate
2. **Transitive** — If A->B and B->C, propose A->C (output as Maybe)
3. **Cross-Domain** — Find different subjects sharing the same property (~40% of generated knowledge)
4. **Analogy** — Find subjects with similar predicate profiles (~30% of generated knowledge)
5. **Verification** — LLM judges Maybe entries as True/False, decomposes reasoning into new facts

### Three-Layer Quality Control Pipeline
Every generated entry passes through (in order):
1. **Domain Scope** (`domain_scope.py`) — Blocks cross-domain nonsense (e.g., "orange trains_via backpropagation")
2. **Predicate Transparency** (`predicate_rules.py`) — Blocks invalid property inheritance (e.g., "baptism boils_at 100C")
3. **Bridge Rescue** (`bridges.py`) — Recovers legitimate cross-domain connections with documented reasoning

Key metric: blocking 39 bad transitive entries prevented 166 downstream garbage entries (5.3x cascade multiplier). Quality control early in the pipeline has outsized impact.

### Predicate Transparency Rules
- **Transparent links** (is_a, subclass_of): properties pass through
- **Opaque links** (uses, causes, symbolizes): block intrinsic properties, allow relational
- **Semi-transparent links** (has_part, made_of): identity properties pass, intrinsic don't

### Bridge System — 24 Types in 6 Categories
| Category | Types | What They Capture |
|----------|-------|-------------------|
| How knowledge connects | historical, inspirational, structural, causal, emergent, technological | Direct cross-domain mechanisms |
| How knowledge means | cultural, etymological, linguistic, mythological | Context shapes interpretation |
| How knowledge scales | scale, pedagogical, contradictory | Patterns across size, learning, conflicts |
| How knowledge feels | experiential, aesthetic, emotional, sensory | Abstract meets subjective experience |
| How knowledge weighs | ethical, religious, spiritual, philosophical | Moral/existential dimension |
| How knowledge misleads | metaphorical, rumor, humor, pattern | Wrong, incomplete, or purely structural |

---

## File Inventory

### Source Code (`ClaudeInput/three_layer_pulse_system/`)

| File | Role | Metaphor |
|------|------|----------|
| `curriculum_pulse.py` | Main brain — curriculum + pulse engine + Flask dashboard | The Brain |
| `ingest.py` | Knowledge ingestion (textbooks, Wikipedia, Wikidata, ConceptNet) | The Mouth |
| `query.py` | Query engine — natural language, graph walk, path finding | The Voice |
| `optimize.py` | Database optimization — normalize, hierarchy, centrality, FTS5 | The Janitor |
| `domain_scope.py` | Domain awareness — blocks cross-domain nonsense | The Immune System |
| `bridges.py` | Cross-domain connections with reasoning + 96 seed bridges | The Diplomat |
| `predicate_rules.py` | Predicate transparency — which property chains are valid | The Grammar |
| `pulse_production.py` | Simpler standalone pulse engine (no curriculum) | Standalone |
| `test_water.py` | Full pipeline integration test on water domain | Test |

### Seed Data
| File | Contents |
|------|----------|
| `ai_kb_big.json` | 150 AI/ML seed entries |
| `water_seed.json` | 140 water domain seed entries |

### Generated
| File | Description |
|------|-------------|
| `curriculum.db` | Shared SQLite database (all scripts read/write this) |

---

## Database

All scripts share a single SQLite database (`curriculum.db`). Core table is `entries`:
- `subject`, `predicate`, `object` — the triple
- `truth_value` — T (True), F (False), M (Maybe/unverified)
- `source` — where it came from (seed, idea:transitive, idea:cross_domain, llm:decomposition, etc.)
- `generation` — which pulse cycle produced it
- `domain`, `strength`, `evidence_for`, `evidence_against`

Supporting tables: `hierarchy`, `centrality`, `entries_fts`, `synonyms`, `entry_domains`, `predicate_map`, `bridges`, `confusion_map`

### Hash Indexing (Planned)
FNV-1a 64-bit hash columns for fast lookups. Pattern: hash narrows to ~1 row, string match confirms (collision safety). See `ClaudeInput/hash_indexing_spec.md` for full spec.

---

## Dependencies
- **Python 3.8+** (standard library: sqlite3, json, argparse, collections, itertools)
- **requests** — for Ollama HTTP API communication
- **flask** — optional, for the web dashboard
- **Ollama** — local LLM runner (llama3.2, mistral, or phi3)

---

## Workflow Conventions

### Folders
- **ClaudeInput/** — User drops files here for Claude to process/integrate
- **ClaudeOutput/** — Claude places generated artifacts here for user

### Terminology
- **Pulse** — One cycle of merge + transitive + cross-domain + analogy
- **Bridge** — A documented cross-domain connection with type, strength, and reasoning
- **Transparent/Opaque/Semi-transparent** — Predicate link types for transitive filtering
- **Intrinsic/Relational/Identity** — Property types that determine inheritance behavior
- **Confusion Map** — Auto-generated map of where understanding commonly fails

---

## Integration Points (Not Yet Wired)

These are documented next steps from the session report:

1. **predicate_rules.py -> curriculum_pulse.py** — TransitiveFilter needs to be imported into the main pulse engine's transitive chaining step
2. **bridges.py -> query.py** — Query engine should check bridge table alongside main KB
3. **Bridge-Aware Pulse Engine** — When pulse discovers cross-domain pattern, check existing bridges; if no match, flag as candidate for LLM verification
4. **Hash Indexing** — FNV-1a columns on all major tables per hash_indexing_spec.md
5. **Cross-Domain Specificity Scoring** — Filter out generic shared properties (>50% frequency)
6. **Predicate Rule Learning** — Auto-learn new transparency rules from LLM verification patterns

---

## Test Results (Reference)

**AI Domain:** 150 seed -> 819 entries (11 generations, ~70% from Idea layer)
**Water Domain:** 140 seed -> 472 entries (with quality filters, ~72% from Idea layer)
- 39 garbage entries blocked by predicate transparency
- 166 downstream garbage prevented (5.3x cascade)
- 14 of 24 bridge types activated from water alone
- Zero false negatives in blocking
