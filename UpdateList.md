# Update List

Tracked issues and improvements noted during piece-by-piece code import.

---

## Project-Wide

- **Unicode on Windows**: All scripts use box-drawing characters (═, ─, etc.) that crash on Windows without `PYTHONUTF8=1`. Need a project-wide fix — either strip fancy chars, add encoding headers, or set env var in a launcher script.

---

## predicate_rules.py

- [x] `has_part` and `contains` were duplicated across `OPAQUE_LINKS` and `SEMI_TRANSPARENT_LINKS` — removed from OPAQUE (fixed during import)
- [ ] `profile_size` meta-predicate filter (line 162) is a hardcoded band-aid from test_water.py's merge step. Should be a configurable set or a proper meta-predicate category.

---

## domain_scope.py

- [ ] **Phantom domains**: Compatible lists reference undefined domains (religion, medicine, engineering, economics, ecology, psychology, agriculture, linguistics, materials, politics, literature, astronomy). Nothing ever classifies into them. Either define them or remove the dead references.
- [ ] **Bare `except:` blocks** (lines 281, 413): Swallow all exceptions silently. Should be specific exception types so real bugs don't vanish.
- [ ] **Aggressive substring matching** (line 301-303): `kw in term_lower` causes false hits — "ion" matches "animation", "graph" matches "paragraph", "set" matches "reset". Needs word-boundary awareness or minimum length threshold.
- [ ] **`--build` flag declared but not implemented** (line 878): Falls through to else clause. Either implement or remove.
- [ ] **Single-domain baseline for multi-domain terms**: `"water": "chemistry"` and `"ice": "everyday"` should be multi-domain lists like `"earth"` is, since water/ice span chemistry, physics, earth_science, everyday.

---

## optimize.py

- [ ] **Bare `except:` blocks** (lines 311-312, 674-675): Same pattern as domain_scope.py — swallows all exceptions. Should catch `sqlite3.OperationalError` for the "table doesn't exist" case.
- [ ] **Deduplicate keeps oldest, not best** (line 624): Docstring says "keeping the one with most evidence" but code just keeps lowest ID (oldest). Should compare `evidence_for` fields if the intent is to keep the best.
- [ ] **Domain keyword overlap with domain_scope.py**: Both files define independent domain keyword lists (optimize has 8 domains, domain_scope has 10). These will drift. Should share a single source of truth.
- [ ] **Substring matching in domain clustering** (lines 571-572): Same `LIKE '%kw%'` false-hit problem as domain_scope.py — "ion" matches "animation", etc.
- [ ] **`import time` inside main()** (line 809): Minor — should be at top of file with other imports.

---

## bridges.py

- [ ] **SQL string interpolation** (lines 326-331, 390-393): Builds `IN (...)` clauses via f-strings instead of parameterized queries. Safe in practice (values from class constants), but bad pattern.
- [ ] **Bare `except:` block** (line 505): Float parse in `verify_bridge_with_llm`. Should be `except (ValueError, TypeError)`.
- [ ] **Duplicate check is directional** (line 163-166): Checks `(term_a, term_b, type)` but not `(term_b, term_a, type)`. With `bidirectional=True`, reversed duplicates will slip through.
- [ ] **No `close()` method on BridgeSystem**: Opens sqlite connection in `__init__` but never closes it. Should add `close()` or context manager (`__enter__`/`__exit__`).
- [ ] **Import path for `domain_scope`** (line 287): `BridgeDiscovery.__init__` does `from domain_scope import DomainClassifier` — will need path update when code moves to src/.
- [ ] **Seed bridge coverage gaps**: Only 1 historical, 1 metaphorical, 1 causal seed bridge. Other types have 3-6. Consider adding more examples for underrepresented types.

---

## pulse_production.py

- [ ] **Default DB is `knowledge.db`, not `curriculum.db`** (lines 71, 614): Every other script uses `curriculum.db`. Will create a separate database if run without `--db`. Standardize to `curriculum.db`.
- [ ] **Schema mismatch**: `entries` table is missing `domain`, `strength` columns that other scripts expect. Has extra columns (`quantifier`, `needs_verification`) that others don't use. Schemas need harmonizing.
- [ ] **No quality filters on transitive step** (line 381): `_transitive()` uses a hardcoded skip set instead of `TransitiveFilter` from predicate_rules.py. This is the key integration point — wiring it prevents garbage transitives (5.3x cascade).
- [ ] **No domain scope check on cross-domain step** (line 418): `_cross_domain()` doesn't check domain compatibility via domain_scope.py.
- [ ] **Bare `except:` block** (line 235): Should catch `(requests.ConnectionError, requests.Timeout)`.
- [ ] **`f` shadows builtin** (line 188): `f = self.count("F")` shadows Python's builtin. Minor but bad habit.

---

## ingest.py

- [ ] **Duplicate `KnowledgeDB` and `OllamaLLM` classes**: Third copy of these classes across the codebase. Each version has different schemas/methods. Need a single shared module.
- [ ] **Schema mismatch**: Has `grade_level` column that others don't. Missing `domain`, `strength` columns. Three incompatible entry schemas across project.
- [ ] **BUG: Missing methods in agent verify task** (lines 931-938): Agent loop's `verify` handler calls `db.find()`, `llm.verify_entry()`, and `db.update_truth()` — none of which exist in this file's class definitions. Would crash at runtime. Copied from pulse_production.py without carrying over the methods.
- [ ] **Wikidata entity resolution incomplete** (lines 648-653): Entity references stored as raw Q-IDs (e.g., Q12345) instead of labels. Comment mentions "second pass resolves them" but no second pass exists.
- [ ] **Duplicate Wikidata property mappings**: `P1542` mapped to both `has_effect` (line 544) and `causes` (line 564). `P1376` mapped to `capital_of` twice (lines 552, 577). Python dict keeps last value, silently losing first.
- [ ] **Bare `except:` blocks** (lines 217, 446, 795, 859): Same project-wide pattern.

---

## query.py

- [ ] **4th copy of `KnowledgeDB` and `OllamaLLM`**: Project needs a single shared module for these classes.
- [x] **No bridge table integration**: ~~query engine ignores bridge table~~ → WIRED. BridgeSystem integrated into QueryEngine. `/bridges` command added. Bridge results shown in `/raw` and `query()`.
- [ ] **No FTS5 usage**: `search()` uses `LIKE '%term%'` instead of the FTS5 index that optimize.py builds. Missing faster, better search.
- [ ] **No synonym expansion**: optimize.py builds a synonym table, but query engine doesn't use it.
- [ ] **Bare `except:` blocks** (lines 201, 224, 262): Same project-wide pattern.
- [ ] **`show_stats` references `grade_level` column** (line 613): Not all entry table schemas include this column — will error if table was created by a different script.

---

## curriculum_pulse.py

- [ ] **5th copy of `KnowledgeDB` and `OllamaLLM`**: The "canonical" version but still duplicated everywhere. Project needs a shared `db.py` and `llm.py` module.
- [x] **No TransitiveFilter integration** (line 672): ~~uses hardcoded skip set~~ → WIRED. `self.transitive_filter.check_chain()` now gates `_transitive()`. Hardcoded skip set kept for meta-predicates.
- [x] **No DomainClassifier integration**: ~~`_cross_domain()` doesn't use domain_scope.py~~ → WIRED. `self.domain_classifier.compatibility_score()` now gates `_cross_domain()`. Blocks 0.0 < compat < 0.3.
- [ ] **Schema missing `domain`, `strength`**: Same schema mismatch as other scripts.
- [ ] **Bare `except:` in LLM._check()** (line 241): Same project-wide pattern.

---

## test_water.py

- [ ] **Import paths assume same directory** (lines 12-15): `from domain_scope import DomainClassifier`, etc. Will need path adjustment (sys.path or package structure) depending on final directory layout.
- [ ] **Creates its own schema variant**: `entries` table missing `quantifier`, `verified` columns that pulse_production.py creates. No `grade_level` from ingest.py either. Three+ incompatible schemas across project.
- [ ] **`profile_size` band-aid in merge step** (line ~180): Merge creates synthetic `profile_size` entries to count how many predicates a subject has. These aren't real knowledge — they're a merge artifact that leaks into the KB. predicate_rules.py has a hardcoded filter for this (line 162).
- [ ] **Hardcoded similarity threshold** (line ~320): Cross-domain uses `jaccard >= 0.3` with no config. Should be tunable.
- [ ] **Only integration test in project**: This is the ONLY file that actually imports and wires TransitiveFilter + DomainClassifier. It's the proof-of-concept for how curriculum_pulse.py should work. High value reference.

---

## Project-Wide Summary

### Critical (blocks correct operation)
1. ~~**5 copies of KnowledgeDB, 4 copies of OllamaLLM**~~ → DONE. Extracted shared `src/db.py` + `src/llm.py`. All 4 scripts updated to use shared modules (~1050 lines removed).
2. ~~**Schema fragmentation**~~ → DONE. Single canonical schema with all 14 columns in `db.py`. Supporting tables created on-demand via `enable_*()` methods.
3. ~~**BUG in ingest.py agent verify**~~ → DONE. `db.find()`, `llm.verify_entry()`, `db.update_truth()` now exist in shared modules.
4. ~~**pulse_production.py uses `knowledge.db`**~~ → DONE. Changed `--db` default to `curriculum.db`.

### High (quality/correctness)
5. ~~**No TransitiveFilter in pulse engines**~~ → DONE for curriculum_pulse.py. pulse_production.py still uses hardcoded skip set.
6. ~~**No DomainClassifier in pulse engines**~~ → DONE for curriculum_pulse.py. pulse_production.py still unwired.
7. ~~**No bridge integration in query.py**~~ → DONE. BridgeSystem wired into QueryEngine + /bridges command added.
8. **Bare `except:` everywhere** — 15+ instances across all files. Silently swallowing real bugs.

### Medium (quality of life)
9. **Substring matching false positives** — domain_scope.py and optimize.py both hit "ion" in "animation", etc.
10. **Phantom domains in domain_scope.py** — 12 domains referenced in compatibility lists but never classified into.
11. **Duplicate Wikidata mappings in ingest.py** — silently losing property definitions.
12. **Unicode/Windows** — all scripts crash without PYTHONUTF8=1 due to box-drawing characters.

### Low (cleanup)
13. **SQL string interpolation in bridges.py** — safe in practice but bad pattern.
14. **Dedup keeps oldest not best in optimize.py** — docstring/code mismatch.
15. **`import time` inside main() in optimize.py** — minor style.
