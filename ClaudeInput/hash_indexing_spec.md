# Hash-Based Index Optimization — Implementation Spec
## Supplement to Architecture Specification v3

---

## Overview

This document specifies hash-based lookup accelerators for the Three-Layer Pulse System. The core principle: **keep auto-increment INTEGER PRIMARY KEYs for relational integrity, add hash columns as fast-path lookup indexes.**

All hash columns are computed once at insertion time and remain stable because they operate on canonical forms (post-synonym-resolution). This means hash values never need recomputation unless the synonym layer remaps a canonical term — an event rare enough to handle as a batch migration.

---

## Hash Function Selection

### Recommended: FNV-1a (Fowler-Noll-Vo)

**Why FNV-1a:**
- Fast to compute (bit shifts and XORs, no division)
- Good distribution for short strings (which canonical terms are)
- Deterministic across platforms and sessions (unlike Python's `hash()` which is randomized per-process since Python 3.3)
- Simple to implement — under 10 lines in any language
- 64-bit variant gives effectively zero collision risk at knowledge-base scale

**Reference Implementation (Python):**
```python
def fnv1a_64(data: str) -> int:
    """FNV-1a 64-bit hash. Input is a UTF-8 string."""
    FNV_OFFSET = 14695981039346656037
    FNV_PRIME = 1099511628211
    h = FNV_OFFSET
    for byte in data.encode('utf-8'):
        h ^= byte
        h = (h * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h
```

**Reference Implementation (C / SQLite extension):**
```c
uint64_t fnv1a_64(const char *data, size_t len) {
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < len; i++) {
        h ^= (uint64_t)(unsigned char)data[i];
        h *= 1099511628211ULL;
    }
    return h;
}
```

**Critical Rule:** All hash inputs must be **lowercase, trimmed, canonical forms**. Hash computation always happens AFTER synonym resolution, never before. The normalization pipeline is: `raw input → synonym layer → canonical form → lowercase + trim → hash`.

---

## Layer 0 — Synonym Table Modifications

### Current Schema
```sql
synonyms:
  surface_form  TEXT
  canonical     TEXT
  type          TEXT
  confidence    REAL
  source        TEXT
```

### Modified Schema
```sql
synonyms:
  surface_form       TEXT
  surface_form_hash  INTEGER NOT NULL    -- fnv1a_64(lowercase(surface_form))
  canonical          TEXT
  canonical_hash     INTEGER NOT NULL    -- fnv1a_64(lowercase(canonical))
  type               TEXT
  confidence         REAL
  source             TEXT

-- Indexes
CREATE INDEX idx_syn_surface ON synonyms(surface_form_hash);
CREATE INDEX idx_syn_canonical ON synonyms(canonical_hash);
```

### Lookup Pattern
```sql
-- Resolving a raw term to its canonical form
-- Hash narrows to ~1 row, string match confirms (collision safety)
SELECT canonical FROM synonyms
WHERE surface_form_hash = ? AND surface_form = ?;
```

### Why This Matters
Every input to the system passes through synonym resolution. Every query, every ingestion, every rules engine operation. This table gets hit more than any other. The hash index turns full-table string scans into integer lookups. At 100K+ synonym entries, this is the difference between milliseconds and seconds per normalization call.

---

## Layer 1 — Facts Table Modifications

### Current Schema
```sql
entries:
  id                INTEGER PRIMARY KEY
  subject           TEXT NOT NULL
  predicate         TEXT NOT NULL
  object            TEXT NOT NULL
  truth_value       TEXT DEFAULT 'M'
  domain            TEXT DEFAULT 'general'
  source            TEXT DEFAULT 'seed'
  evidence          TEXT DEFAULT ''
  generation        INTEGER DEFAULT 0
  scope             TEXT DEFAULT NULL
  strength          REAL DEFAULT 0.5
  eliminated_count  INTEGER DEFAULT 0
  created_at        REAL
  last_verified     REAL
```

### Modified Schema
```sql
entries:
  id                INTEGER PRIMARY KEY
  subject           TEXT NOT NULL
  predicate         TEXT NOT NULL
  object            TEXT NOT NULL
  subject_hash      INTEGER NOT NULL     -- fnv1a_64(subject)
  predicate_hash    INTEGER NOT NULL     -- fnv1a_64(predicate)
  object_hash       INTEGER NOT NULL     -- fnv1a_64(object)
  triple_hash       INTEGER NOT NULL     -- fnv1a_64(subject + "|" + predicate + "|" + object)
  truth_value       TEXT DEFAULT 'M'
  domain            TEXT DEFAULT 'general'
  source            TEXT DEFAULT 'seed'
  evidence          TEXT DEFAULT ''
  generation        INTEGER DEFAULT 0
  scope             TEXT DEFAULT NULL
  strength          REAL DEFAULT 0.5
  eliminated_count  INTEGER DEFAULT 0
  created_at        REAL
  last_verified     REAL

-- Indexes: ordered by expected query frequency
CREATE INDEX idx_entry_subject    ON entries(subject_hash);
CREATE INDEX idx_entry_predicate  ON entries(predicate_hash);
CREATE INDEX idx_entry_subj_pred  ON entries(subject_hash, predicate_hash);
CREATE INDEX idx_entry_triple     ON entries(triple_hash);
CREATE INDEX idx_entry_object     ON entries(object_hash);
```

### Lookup Patterns

```sql
-- DUPLICATE DETECTION (before insertion)
-- "Does this exact triple already exist?"
SELECT id FROM entries
WHERE triple_hash = ? AND subject = ? AND predicate = ? AND object = ?;

-- SUBJECT QUERY (most common — rules engine graph traversal)
-- "Give me everything about water"
SELECT * FROM entries
WHERE subject_hash = ? AND subject = ?;

-- PREDICATE WALK (transitive chaining)
-- "Find all is_a relationships"
SELECT * FROM entries
WHERE predicate_hash = ? AND predicate = ?;

-- SUBJECT + PREDICATE (the workhorse query for rules engine)
-- "What is water is_a?"  /  "What does water cause?"
SELECT * FROM entries
WHERE subject_hash = ? AND predicate_hash = ?
  AND subject = ? AND predicate = ?;

-- REVERSE LOOKUP (finding what points TO a concept)
-- "What things are is_a animal?"
SELECT * FROM entries
WHERE object_hash = ? AND predicate_hash = ?
  AND object = ? AND predicate = ?;
```

### Why This Matters
The rules engine is the most compute-intensive offline process. Every pass — generative, reductive, cross-validation, coherence — is built on graph traversals that start with subject and/or predicate lookups. The composite `subject_hash + predicate_hash` index directly accelerates the inner loop of BFS traversal, transitive chaining, and constraint checking. At scale (100K+ triples), this turns O(n) scans into O(1) lookups per hop.

The `triple_hash` index solves the deduplication problem from the original question. Before inserting any new entry, compute the triple hash and check for existence. No scan required.

### Hash Computation for Triples
```python
def compute_triple_hash(subject: str, predicate: str, obj: str) -> int:
    """
    Delimiter-separated to prevent collisions between
    ("ab", "c", "d") and ("a", "bc", "d").
    Pipe character chosen because it won't appear in canonical forms.
    """
    return fnv1a_64(f"{subject}|{predicate}|{obj}")
```

---

## Layer 2 — Relationships Table Modifications

### Modified Schema (additions only)
```sql
relationships:
  -- existing columns unchanged --
  name_hash         INTEGER NOT NULL     -- fnv1a_64(name)

-- Index
CREATE INDEX idx_rel_name ON relationships(name_hash);
```

### Lookup Pattern
```sql
-- Find relationship by descriptive name
SELECT * FROM relationships
WHERE name_hash = ? AND name = ?;
```

### Note
Layer 2 is less performance-critical for hash lookups because it's primarily accessed via foreign key joins from `relationship_l1_links` (which use integer ID references already). The `name_hash` is mainly useful for the digestive system checking "does this relationship already exist?" during L1 → L2 promotion.

---

## Layer 3 — Chains Table Modifications

### Current Schema (relevant portion)
```sql
chains:
  id                INTEGER PRIMARY KEY
  entry_point       TEXT NOT NULL
  entry_point_hash  TEXT              -- already present but as TEXT
  conclusion        TEXT NOT NULL
  ...
```

### Modified Schema
```sql
chains:
  id                    INTEGER PRIMARY KEY
  entry_point           TEXT NOT NULL
  entry_point_hash      INTEGER NOT NULL   -- fnv1a_64(normalized_entry_point)
  conclusion            TEXT NOT NULL
  conclusion_hash       INTEGER NOT NULL   -- fnv1a_64(normalized_conclusion)
  ...

-- Indexes
CREATE INDEX idx_chain_entry ON chains(entry_point_hash);
CREATE INDEX idx_chain_conclusion ON chains(conclusion_hash);
```

### Critical: Hash the Canonical Decomposed Form

The entry point hash should NOT be computed from the raw user query. It should be computed from the **post-decomposition canonical form** produced by the Decomposition Wrapper.

```
Raw query:          "Why does ice float on water?"
After decomposition: [("ice", "floats_on", "water")]
Canonical form:      "ice|floats_on|water"
Hash:                fnv1a_64("ice|floats_on|water")

Raw query:          "How come ice floats?"
After decomposition: [("ice", "floats_on", "water")]
Canonical form:      "ice|floats_on|water"
Hash:                SAME HASH → cache hit
```

This means two differently-worded questions that decompose to the same structured query will hit the same cached chain. The synonym layer handles "ice" vs "frozen water"; the decomposition wrapper handles "why does X" vs "how come X"; the hash unifies them.

### Conclusion Hash Use Case
The `conclusion_hash` enables reverse lookups: "Are there any chains that concluded X?" This is useful for the audit system when an L1 fact changes — you can find chains whose conclusions depend on that fact without walking the full dependency graph.

---

## Known Analogies Table Modifications

### Modified Schema (additions only)
```sql
known_analogies:
  -- existing columns unchanged --
  source_hash       INTEGER NOT NULL     -- fnv1a_64(source_concept)
  target_hash       INTEGER NOT NULL     -- fnv1a_64(target_concept)
  bridge_hash       INTEGER NOT NULL     -- fnv1a_64(source_concept + "|" + target_concept)

-- Indexes
CREATE INDEX idx_analogy_source ON known_analogies(source_hash);
CREATE INDEX idx_analogy_target ON known_analogies(target_hash);
CREATE INDEX idx_analogy_bridge ON known_analogies(bridge_hash);
```

### Lookup Pattern
```sql
-- "What analogies exist for atoms?"
SELECT * FROM known_analogies
WHERE source_hash = ? AND source_concept = ?;

-- "What concepts are compared TO solar systems?"
SELECT * FROM known_analogies
WHERE target_hash = ? AND target_concept = ?;

-- "Does this specific analogy already exist?"
SELECT * FROM known_analogies
WHERE bridge_hash = ? AND source_concept = ? AND target_concept = ?;
```

---

## Insertion Workflow (All Layers)

Every insertion follows the same pattern:

```
1. Receive raw data
2. Pass through synonym layer (Layer 0) → get canonical forms
3. Compute hash(es) from canonical forms
4. Check for existing entry via hash (duplicate detection)
   → EXISTS: update/merge/skip depending on context
   → NEW: proceed to step 5
5. Insert with precomputed hash values
6. Hash columns are now available for all future lookups
```

The hashes are **write-once, read-many**. Computed at insertion, used on every subsequent query. This front-loads a trivial cost (one hash computation per insert) to save a significant cost (string comparison avoidance on every lookup).

---

## Migration Path (If Existing Data Exists)

If you've already populated tables before adding hash columns:

```sql
-- Add columns
ALTER TABLE entries ADD COLUMN subject_hash INTEGER;
ALTER TABLE entries ADD COLUMN predicate_hash INTEGER;
ALTER TABLE entries ADD COLUMN object_hash INTEGER;
ALTER TABLE entries ADD COLUMN triple_hash INTEGER;

-- Backfill (run via Python script that computes fnv1a_64 for each row)
-- Then set NOT NULL constraints and create indexes

-- Verify: count of DISTINCT triple_hash should be close to row count
-- Any duplicates indicate either hash collisions (rare) or actual duplicate triples (clean up)
SELECT triple_hash, COUNT(*) as cnt FROM entries
GROUP BY triple_hash HAVING cnt > 1;
```

---

## Performance Expectations

| Operation | Before (string index) | After (hash index) | Notes |
|---|---|---|---|
| Synonym resolution | O(log n) B-tree | O(1) hash lookup | Biggest win — called on every operation |
| Triple duplicate check | O(n) scan or O(log n) composite index | O(1) hash check | Prevents redundant entries at insertion |
| Subject lookup (L1) | O(log n) B-tree on TEXT | O(1) integer index | Core rules engine operation |
| Subject+Predicate (L1) | O(log n) composite B-tree | O(1) composite hash | The inner loop accelerator |
| Chain cache check (L3) | O(log n) TEXT comparison | O(1) integer match | Query-time fast path |

Note: "O(1)" assumes low collision rate, which FNV-1a 64-bit provides. At 1 million entries, expected collisions ≈ 0. At 1 billion entries, still negligible.

---

## Collision Safety

All hash lookups use the **hash-then-confirm** pattern:
```sql
WHERE hash_column = [computed_hash] AND text_column = [original_text]
```

The hash narrows results to (almost certainly) one row. The string comparison confirms it's not a collision. This is defense-in-depth — FNV-1a 64-bit collisions are astronomically unlikely at any practical database size, but the confirmation costs almost nothing since it's only comparing one row.

**Never trust a hash match alone for identity. Always confirm with the original value.**

---

## What This Doesn't Change

- **PRIMARY KEYs remain auto-increment INTEGER.** All foreign key relationships (`relationship_l1_links.entry_id`, `chain_dependencies.entry_id`, etc.) continue using integer IDs. Hash columns are lookup accelerators, not primary keys.
- **The rules engine logic is unchanged.** Same traversal patterns, same passes, same constraint checking. Only the underlying query performance improves.
- **The digestive system, audit system, and overseer are unchanged.** They benefit passively from faster queries but require no modifications to their logic.

---

*Spec version: 1.0*
*Supplements: architecture_spec_v3.md*
*Date: February 22, 2026*
*Status: Ready for implementation alongside Phase 1*
