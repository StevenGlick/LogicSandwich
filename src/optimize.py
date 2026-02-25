#!/usr/bin/env python3
"""
KNOWLEDGE BASE OPTIMIZER
════════════════════════

Run this periodically (or after big ingests) to optimize the database.
Think of it like defragmenting — reorganizes the data so everything
runs faster and smarter.

WHAT IT DOES:
  1. Normalizes predicates (uses/utilizes/employs → uses)
  2. Builds hierarchy index (is_a chains → fast ancestor lookups)
  3. Scores centrality (how connected/important each entry is)
  4. Creates full-text search index (fuzzy matching)
  5. Clusters by domain (physics, biology, history, etc.)
  6. Adds synonym expansion for queries

USAGE:
  python3 optimize.py                     # run all optimizations
  python3 optimize.py --normalize         # just predicate normalization
  python3 optimize.py --hierarchy         # just hierarchy index
  python3 optimize.py --centrality        # just centrality scoring
  python3 optimize.py --fts               # just full-text search index
  python3 optimize.py --domains           # just domain clustering
  python3 optimize.py --report            # show optimization report
  python3 optimize.py --db mydata.db      # use specific database
"""

import sqlite3
import json
import argparse
import re
from collections import defaultdict, deque


# ═══════════════════════════════════════════════════════════
# PREDICATE NORMALIZATION
# ═══════════════════════════════════════════════════════════
# Problem: "uses", "utilizes", "employs", "uses_tool" all mean
#          roughly the same thing but fragment the graph.
# Solution: Map variants to canonical predicates.
# ═══════════════════════════════════════════════════════════

# Canonical predicate → list of variants that mean the same thing
PREDICATE_SYNONYMS = {
    # Taxonomy
    "is_a":             ["is_type_of", "is_kind_of", "type_of", "kind_of",
                         "instance_of", "is_instance_of", "classified_as",
                         "is_classified_as", "belongs_to_category"],
    "subclass_of":      ["subcategory_of", "is_subclass_of", "sub_type_of",
                         "specialization_of", "narrower_than"],
    "has_part":         ["has_component", "contains_part", "includes",
                         "comprises", "consists_of_part"],
    "part_of":          ["component_of", "is_part_of", "belongs_to",
                         "contained_in", "member_of"],
    
    # Usage / Tools
    "uses":             ["utilizes", "employs", "makes_use_of", "uses_tool",
                         "uses_technique", "relies_on", "depends_on"],
    "used_for":         ["utilized_for", "employed_for", "serves_as",
                         "functions_as", "purpose_is"],
    
    # Creation
    "created_by":       ["made_by", "built_by", "developed_by", "designed_by",
                         "invented_by", "authored_by", "constructed_by"],
    "produces":         ["generates", "outputs", "creates", "yields",
                         "results_in", "gives"],
    
    # Causation
    "causes":           ["leads_to", "results_in", "triggers", "induces",
                         "brings_about", "gives_rise_to"],
    "caused_by":        ["due_to", "result_of", "caused_from",
                         "originates_from", "stems_from"],
    "prevents":         ["blocks", "stops", "inhibits", "suppresses"],
    "enables":          ["allows", "permits", "makes_possible",
                         "facilitates", "supports"],
    
    # Properties
    "has_property":     ["has_quality", "has_attribute", "has_characteristic",
                         "characterized_by", "described_as"],
    "made_of":          ["composed_of", "consists_of", "material_is",
                         "constructed_from", "built_from"],
    
    # Location
    "located_in":       ["found_in", "situated_in", "exists_in",
                         "occurs_in", "present_in", "lives_in"],
    
    # Comparison
    "similar_to":       ["resembles", "like", "comparable_to",
                         "analogous_to", "akin_to"],
    "different_from":   ["distinct_from", "unlike", "contrasts_with",
                         "differs_from", "not_same_as"],
    "larger_than":      ["bigger_than", "greater_than", "exceeds",
                         "more_than"],
    "smaller_than":     ["less_than", "tinier_than", "under"],
    
    # Requirements
    "requires":         ["needs", "depends_on", "must_have",
                         "prerequisite_is", "demands"],
    
    # Definition
    "defined_as":       ["means", "definition_is", "refers_to",
                         "known_as", "described_as"],
    
    # Temporal
    "preceded_by":      ["comes_after", "follows", "succeeded"],
    "followed_by":      ["comes_before", "precedes", "leads_into"],
}

def build_predicate_lookup():
    """Build reverse lookup: variant → canonical."""
    lookup = {}
    for canonical, variants in PREDICATE_SYNONYMS.items():
        lookup[canonical] = canonical  # map canonical to itself
        for v in variants:
            lookup[v] = canonical
    return lookup

def normalize_predicates(conn):
    """
    Normalize all predicates in the database to canonical forms.
    This merges fragmented relationships.
    """
    cur = conn.cursor()
    lookup = build_predicate_lookup()
    
    # Create normalization table
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS predicate_map (
            original TEXT PRIMARY KEY,
            canonical TEXT NOT NULL
        );
    """)
    
    # Get all unique predicates
    cur.execute("SELECT DISTINCT predicate FROM entries")
    all_preds = [r[0] for r in cur.fetchall()]
    
    normalized = 0
    for pred in all_preds:
        canonical = lookup.get(pred, pred)  # default to itself if unknown
        cur.execute("INSERT OR REPLACE INTO predicate_map VALUES (?,?)",
                   (pred, canonical))
        if canonical != pred:
            # Update entries to use canonical predicate
            cur.execute("UPDATE entries SET predicate=? WHERE predicate=?",
                       (canonical, pred))
            count = cur.rowcount
            if count > 0:
                normalized += count
                print(f"    {pred:30s} → {canonical} ({count} entries)")
    
    conn.commit()
    print(f"  Normalized {normalized} entries across {len(all_preds)} predicates")
    return normalized


# ═══════════════════════════════════════════════════════════
# HIERARCHY INDEX
# ═══════════════════════════════════════════════════════════
# Precompute "is_a" ancestry chains so we can instantly
# answer "is X a type of Y?" and "give me everything under Y"
# ═══════════════════════════════════════════════════════════

def build_hierarchy(conn):
    """
    Build a hierarchy table from is_a chains.
    
    If dog is_a mammal and mammal is_a animal,
    this creates:
      dog → mammal (depth 1)
      dog → animal (depth 2)
      mammal → animal (depth 1)
    
    Then "find all animals" instantly returns dog, mammal, etc.
    """
    cur = conn.cursor()
    
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS hierarchy (
            child TEXT NOT NULL,
            ancestor TEXT NOT NULL,
            depth INTEGER NOT NULL,
            path TEXT DEFAULT '',
            PRIMARY KEY (child, ancestor)
        );
        CREATE INDEX IF NOT EXISTS idx_hier_child ON hierarchy(child);
        CREATE INDEX IF NOT EXISTS idx_hier_ancestor ON hierarchy(ancestor);
        DELETE FROM hierarchy;
    """)
    
    # Get all is_a relationships
    cur.execute("""
        SELECT subject, object FROM entries 
        WHERE predicate IN ('is_a', 'subclass_of', 'is_type_of', 'instance_of')
        AND truth_value='T' AND object NOT LIKE '%,%'
    """)
    
    # Build parent lookup
    parents = defaultdict(set)
    for row in cur.fetchall():
        child, parent = row
        if child != parent:  # no self-loops
            parents[child].add(parent)
    
    # Walk chains using BFS for each node
    total_entries = 0
    for start in parents:
        queue = deque()
        visited = set()
        
        # Add direct parents
        for p in parents[start]:
            queue.append((p, 1, [start, p]))
            visited.add(p)
        
        while queue:
            current, depth, path = queue.popleft()
            
            cur.execute(
                "INSERT OR IGNORE INTO hierarchy (child, ancestor, depth, path) VALUES (?,?,?,?)",
                (start, current, depth, "→".join(path)))
            total_entries += 1
            
            # Walk up to grandparents
            if depth < 10:  # cap depth to prevent infinite loops
                for gp in parents.get(current, set()):
                    if gp not in visited:
                        visited.add(gp)
                        queue.append((gp, depth + 1, path + [gp]))
    
    conn.commit()
    
    # Report
    cur.execute("SELECT COUNT(DISTINCT child) FROM hierarchy")
    children = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT ancestor) FROM hierarchy")
    ancestors = cur.fetchone()[0]
    
    print(f"  Hierarchy: {total_entries} paths, {children} children, {ancestors} ancestors")
    
    # Show top-level categories
    cur.execute("""
        SELECT ancestor, COUNT(*) as cnt FROM hierarchy 
        GROUP BY ancestor ORDER BY cnt DESC LIMIT 15
    """)
    print(f"  Top categories:")
    for row in cur.fetchall():
        bar = "█" * min(row[1], 30)
        print(f"    {row[0]:30s} {row[1]:4d} descendants {bar}")
    
    return total_entries


# ═══════════════════════════════════════════════════════════
# CENTRALITY SCORING
# ═══════════════════════════════════════════════════════════
# Score each subject by how connected it is.
# High centrality = foundational concept many things depend on.
# Low centrality = leaf fact, few connections.
# Used to prioritize results in queries.
# ═══════════════════════════════════════════════════════════

def compute_centrality(conn):
    """
    Compute connection centrality for each subject.
    
    Score = (outgoing connections) + (incoming connections) + (hierarchy children)
    
    This is a simple degree centrality. For really big graphs
    you'd want PageRank, but degree centrality is fast and
    good enough for our purposes.
    """
    cur = conn.cursor()
    
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS centrality (
            subject TEXT PRIMARY KEY,
            out_degree INTEGER DEFAULT 0,
            in_degree INTEGER DEFAULT 0,
            hierarchy_children INTEGER DEFAULT 0,
            score REAL DEFAULT 0.0
        );
        DELETE FROM centrality;
    """)
    
    # Count outgoing connections per subject
    cur.execute("""
        SELECT subject, COUNT(*) FROM entries 
        WHERE truth_value='T' AND subject NOT LIKE '%,%'
        GROUP BY subject
    """)
    out_degrees = {r[0]: r[1] for r in cur.fetchall()}
    
    # Count incoming connections per object
    cur.execute("""
        SELECT object, COUNT(*) FROM entries 
        WHERE truth_value='T' AND object NOT LIKE '%,%'
        GROUP BY object
    """)
    in_degrees = {r[0]: r[1] for r in cur.fetchall()}
    
    # Count hierarchy children
    hier_children = {}
    try:
        cur.execute("""
            SELECT ancestor, COUNT(*) FROM hierarchy 
            GROUP BY ancestor
        """)
        hier_children = {r[0]: r[1] for r in cur.fetchall()}
    except sqlite3.OperationalError:
        pass  # hierarchy table might not exist yet
    
    # Combine all subjects
    all_subjects = set(out_degrees.keys()) | set(in_degrees.keys())
    
    for subj in all_subjects:
        out_d = out_degrees.get(subj, 0)
        in_d = in_degrees.get(subj, 0)
        hier = hier_children.get(subj, 0)
        # Score: weighted sum (hierarchy children count extra because
        # they indicate a concept is a category/parent)
        score = out_d + in_d + (hier * 2)
        
        cur.execute(
            "INSERT INTO centrality (subject, out_degree, in_degree, hierarchy_children, score) VALUES (?,?,?,?,?)",
            (subj, out_d, in_d, hier, score))
    
    conn.commit()
    
    # Report top concepts
    cur.execute("SELECT * FROM centrality ORDER BY score DESC LIMIT 20")
    print(f"  Top concepts by centrality:")
    for r in cur.fetchall():
        bar = "█" * min(int(r[4])//2, 30)
        print(f"    {r[0]:30s} score={r[4]:5.0f} (out={r[1]} in={r[2]} hier={r[3]}) {bar}")
    
    return len(all_subjects)


# ═══════════════════════════════════════════════════════════
# FULL TEXT SEARCH (FTS5)
# ═══════════════════════════════════════════════════════════
# SQLite's built-in full-text search engine.
# Handles fuzzy matching, stemming, and ranking.
# This is what makes "what causes weather" find entries
# about rain, wind, clouds, etc.
# ═══════════════════════════════════════════════════════════

def build_fts_index(conn):
    """
    Build a full-text search index over the knowledge base.
    
    FTS5 lets you search with:
      - Simple terms: "weather"
      - Phrases: "nuclear fusion" 
      - Prefix: "photo*" (matches photosynthesis, photography...)
      - Boolean: "rain OR snow"
      - Near: NEAR("sun" "energy", 5)
    """
    cur = conn.cursor()
    
    # Drop and rebuild (FTS tables can't be altered)
    cur.executescript("""
        DROP TABLE IF EXISTS entries_fts;
        CREATE VIRTUAL TABLE entries_fts USING fts5(
            entry_id,
            subject,
            predicate,
            object,
            evidence,
            source,
            content='',
            tokenize='porter unicode61'
        );
    """)
    # porter = stemming (searches → search, running → run)
    # unicode61 = handles accented characters
    
    # Populate
    cur.execute("""
        SELECT id, subject, predicate, object, evidence_for, source 
        FROM entries WHERE truth_value='T'
    """)
    
    count = 0
    for row in cur.fetchall():
        # Expand underscores to spaces for better text matching
        subj = row[1].replace("_", " ")
        pred = row[2].replace("_", " ")
        obj = row[3].replace("_", " ")
        evidence = row[4].replace("_", " ") if row[4] else ""
        source = row[5] or ""
        
        cur.execute(
            "INSERT INTO entries_fts (entry_id, subject, predicate, object, evidence, source) VALUES (?,?,?,?,?,?)",
            (str(row[0]), subj, pred, obj, evidence, source))
        count += 1
    
    conn.commit()
    print(f"  FTS index: {count} entries indexed")
    
    # Test it
    test_queries = ["weather", "energy", "animal", "chemical"]
    for q in test_queries:
        cur.execute(
            "SELECT COUNT(*) FROM entries_fts WHERE entries_fts MATCH ?", (q,))
        hits = cur.fetchone()[0]
        if hits > 0:
            print(f"    test '{q}': {hits} hits")
    
    return count


# ═══════════════════════════════════════════════════════════
# SYNONYM TABLE
# ═══════════════════════════════════════════════════════════
# Maps common words to related terms in the database.
# When you search for "weather", also search for
# "rain", "wind", "cloud", "temperature", etc.
# ═══════════════════════════════════════════════════════════

def build_synonyms(conn):
    """
    Build synonym expansion table from the knowledge base itself.
    
    Uses is_a and part_of relationships:
    - "weather" is_a/has_part → rain, wind, cloud, temperature
    - Searching "weather" now also finds entries about rain, etc.
    
    Also adds manually curated common synonyms.
    """
    cur = conn.cursor()
    
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS synonyms (
            term TEXT NOT NULL,
            synonym TEXT NOT NULL,
            source TEXT DEFAULT 'auto',
            weight REAL DEFAULT 1.0,
            PRIMARY KEY (term, synonym)
        );
        DELETE FROM synonyms;
    """)
    
    # Auto-generate from knowledge graph
    # If A is_a B, then B-queries should include A
    cur.execute("""
        SELECT subject, object FROM entries 
        WHERE predicate IN ('is_a', 'subclass_of') 
        AND truth_value='T' AND subject NOT LIKE '%,%' AND object NOT LIKE '%,%'
    """)
    count = 0
    for row in cur.fetchall():
        child, parent = row
        # parent → child (searching "animal" should find "dog")
        cur.execute("INSERT OR IGNORE INTO synonyms VALUES (?,?,'is_a',0.8)",
                   (parent, child))
        count += 1
    
    # has_part relationships
    cur.execute("""
        SELECT subject, object FROM entries 
        WHERE predicate IN ('has_part', 'has_component', 'contains') 
        AND truth_value='T' AND subject NOT LIKE '%,%' AND object NOT LIKE '%,%'
    """)
    for row in cur.fetchall():
        whole, part = row
        cur.execute("INSERT OR IGNORE INTO synonyms VALUES (?,?,'has_part',0.6)",
                   (whole, part))
        count += 1
    
    # Manual common synonyms (expand as needed)
    MANUAL = {
        "hot": ["temperature", "heat", "thermal"],
        "cold": ["temperature", "freeze", "ice"],
        "big": ["large", "huge", "enormous", "giant"],
        "small": ["tiny", "little", "miniature"],
        "fast": ["quick", "rapid", "speed"],
        "slow": ["sluggish", "gradual"],
        "water": ["h2o", "liquid", "moisture"],
        "light": ["photon", "electromagnetic_radiation", "illumination"],
        "sound": ["audio", "acoustic", "noise", "vibration"],
        "earth": ["planet", "world", "globe"],
        "brain": ["mind", "cerebral", "neural", "cognitive"],
        "computer": ["machine", "processor", "computing"],
        "alive": ["living", "biological", "organism"],
        "dead": ["deceased", "non_living", "extinct"],
    }
    for term, syns in MANUAL.items():
        for s in syns:
            cur.execute("INSERT OR IGNORE INTO synonyms VALUES (?,?,'manual',0.7)",
                       (term, s))
            count += 1
    
    conn.commit()
    
    cur.execute("SELECT COUNT(DISTINCT term) FROM synonyms")
    terms = cur.fetchone()[0]
    print(f"  Synonyms: {count} mappings across {terms} terms")
    
    return count


# ═══════════════════════════════════════════════════════════
# DOMAIN CLUSTERING
# ═══════════════════════════════════════════════════════════
# Group entries by knowledge domain so the pulse engine
# can process one domain at a time instead of everything.
# ═══════════════════════════════════════════════════════════

DOMAIN_KEYWORDS = {
    "physics":    ["force", "energy", "mass", "velocity", "acceleration",
                   "gravity", "momentum", "wave", "frequency", "photon",
                   "electron", "quantum", "relativity", "thermodynamic",
                   "entropy", "magnetic", "electric", "nuclear", "particle",
                   "newton", "light_speed", "kinetic", "potential"],
    "chemistry":  ["atom", "molecule", "element", "compound", "bond",
                   "reaction", "acid", "base", "ion", "electron_orbital",
                   "periodic_table", "chemical", "ph", "oxidation",
                   "catalyst", "solution", "mole", "isotope"],
    "biology":    ["cell", "dna", "rna", "protein", "gene", "organism",
                   "species", "evolution", "photosynthesis", "mitosis",
                   "meiosis", "enzyme", "bacteria", "virus", "ecology",
                   "habitat", "chromosome", "membrane", "organ", "tissue"],
    "math":       ["number", "equation", "function", "variable", "theorem",
                   "proof", "calculus", "algebra", "geometry", "fraction",
                   "decimal", "derivative", "integral", "matrix",
                   "probability", "statistics", "trigonometry", "logarithm"],
    "geography":  ["continent", "country", "ocean", "mountain", "river",
                   "climate", "latitude", "longitude", "population",
                   "capital", "border", "island", "desert", "forest"],
    "history":    ["war", "revolution", "empire", "dynasty", "treaty",
                   "constitution", "civilization", "colonial", "medieval",
                   "ancient", "renaissance", "industrial", "monarchy"],
    "technology": ["computer", "algorithm", "software", "hardware",
                   "internet", "database", "programming", "processor",
                   "neural_network", "transformer", "machine_learning",
                   "artificial_intelligence", "robot", "encryption"],
    "language":   ["noun", "verb", "adjective", "grammar", "syntax",
                   "vocabulary", "phoneme", "morpheme", "sentence",
                   "paragraph", "etymology", "dialect", "alphabet"],
}

def cluster_domains(conn):
    """
    Assign domain labels to entries based on keyword matching.
    An entry can belong to multiple domains.
    """
    cur = conn.cursor()
    
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS entry_domains (
            entry_id INTEGER NOT NULL,
            domain TEXT NOT NULL,
            confidence REAL DEFAULT 1.0,
            PRIMARY KEY (entry_id, domain)
        );
        CREATE INDEX IF NOT EXISTS idx_domain ON entry_domains(domain);
        DELETE FROM entry_domains;
    """)
    
    # For each domain, find entries matching its keywords
    domain_counts = {}
    
    for domain, keywords in DOMAIN_KEYWORDS.items():
        # Build SQL LIKE conditions with word-boundary awareness
        # Use underscore boundaries to avoid "ion" matching "animation"
        conditions = []
        params = []
        for kw in keywords:
            # Match: exact, start_of_term, end_of_term, or _middle_
            conditions.append(
                "(subject = ? OR subject LIKE ? OR subject LIKE ? OR subject LIKE ?"
                " OR object = ? OR object LIKE ? OR object LIKE ? OR object LIKE ?)")
            params.extend([kw, f"{kw}_%", f"%_{kw}", f"%_{kw}_%",
                          kw, f"{kw}_%", f"%_{kw}", f"%_{kw}_%"])
        
        where = " OR ".join(conditions)
        cur.execute(f"""
            SELECT id FROM entries WHERE truth_value='T' AND ({where})
        """, params)
        
        ids = [r[0] for r in cur.fetchall()]
        for eid in ids:
            cur.execute(
                "INSERT OR IGNORE INTO entry_domains (entry_id, domain) VALUES (?,?)",
                (eid, domain))
        
        domain_counts[domain] = len(ids)
    
    conn.commit()
    
    print(f"  Domain clusters:")
    for domain in sorted(domain_counts, key=domain_counts.get, reverse=True):
        cnt = domain_counts[domain]
        bar = "█" * min(cnt//3, 30)
        print(f"    {domain:15s} {cnt:5d} entries {bar}")
    
    return domain_counts


# ═══════════════════════════════════════════════════════════
# DEDUPLICATE
# ═══════════════════════════════════════════════════════════
# After normalization, there may be true duplicates.
# (Same subject+predicate+object after normalization)
# ═══════════════════════════════════════════════════════════

def deduplicate(conn):
    """Remove true duplicates, keeping the one with most evidence."""
    cur = conn.cursor()
    
    # Find duplicates
    cur.execute("""
        SELECT subject, predicate, object, COUNT(*) as cnt, GROUP_CONCAT(id) as ids
        FROM entries
        GROUP BY subject, predicate, object
        HAVING cnt > 1
    """)
    
    dupes = cur.fetchall()
    removed = 0
    
    for row in dupes:
        ids = [int(x) for x in row[4].split(",")]
        # Keep the first (oldest), remove the rest
        keep_id = ids[0]
        remove_ids = ids[1:]
        
        for rid in remove_ids:
            cur.execute("DELETE FROM entries WHERE id=?", (rid,))
            removed += 1
    
    conn.commit()
    print(f"  Deduplication: removed {removed} duplicate entries")
    return removed


# ═══════════════════════════════════════════════════════════
# INTEGRITY CHECK
# ═══════════════════════════════════════════════════════════

def integrity_check(conn):
    """Check for common data quality issues."""
    cur = conn.cursor()
    issues = []
    
    # Self-referential entries
    cur.execute("SELECT COUNT(*) FROM entries WHERE subject=object")
    self_ref = cur.fetchone()[0]
    if self_ref:
        issues.append(f"  ⚠ {self_ref} self-referential entries (subject=object)")
    
    # Empty fields
    cur.execute("SELECT COUNT(*) FROM entries WHERE subject='' OR predicate='' OR object=''")
    empty = cur.fetchone()[0]
    if empty:
        issues.append(f"  ⚠ {empty} entries with empty fields")
        cur.execute("DELETE FROM entries WHERE subject='' OR predicate='' OR object=''")
        conn.commit()
        issues.append(f"    → cleaned {empty} empty entries")
    
    # Very long subjects (probably parsing errors)
    cur.execute("SELECT COUNT(*) FROM entries WHERE LENGTH(subject) > 80")
    long_subj = cur.fetchone()[0]
    if long_subj:
        issues.append(f"  ⚠ {long_subj} entries with very long subjects (>80 chars)")
    
    # Orphan nodes (in hierarchy but not in entries)
    try:
        cur.execute("""
            SELECT COUNT(DISTINCT child) FROM hierarchy 
            WHERE child NOT IN (SELECT DISTINCT subject FROM entries)
        """)
        orphans = cur.fetchone()[0]
        if orphans:
            issues.append(f"  ⚠ {orphans} hierarchy entries without matching knowledge entries")
    except (sqlite3.OperationalError, TypeError):
        pass
    
    if not issues:
        print(f"  ✓ No integrity issues found")
    else:
        for i in issues:
            print(i)
    
    return len(issues)


# ═══════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════

def report(conn):
    """Show a complete report on database state."""
    cur = conn.cursor()
    
    cur.execute("SELECT COUNT(*) FROM entries")
    total = cur.fetchone()[0]
    
    print(f"\n  {'═'*55}")
    print(f"  DATABASE REPORT")
    print(f"  {'═'*55}")
    print(f"\n  Total entries: {total:,}")
    
    # Truth values
    for tv in ["T", "F", "M"]:
        cur.execute("SELECT COUNT(*) FROM entries WHERE truth_value=?", (tv,))
        cnt = cur.fetchone()[0]
        pct = (cnt/total*100) if total else 0
        bar = "█" * min(int(pct/2), 30)
        label = {"T":"True","F":"False","M":"Maybe"}[tv]
        print(f"    {label:6s} {cnt:7,d} ({pct:4.1f}%) {bar}")
    
    # Unique subjects
    cur.execute("SELECT COUNT(DISTINCT subject) FROM entries WHERE subject NOT LIKE '%,%'")
    uniq_subj = cur.fetchone()[0]
    print(f"\n  Unique subjects: {uniq_subj:,}")
    
    # Unique predicates
    cur.execute("SELECT COUNT(DISTINCT predicate) FROM entries")
    uniq_pred = cur.fetchone()[0]
    print(f"  Unique predicates: {uniq_pred:,}")
    
    # Sources
    print(f"\n  Sources:")
    cur.execute("SELECT source, COUNT(*) FROM entries GROUP BY source ORDER BY COUNT(*) DESC LIMIT 10")
    for row in cur.fetchall():
        bar = "█" * min(row[1]//5, 30)
        print(f"    {row[0]:30s} {row[1]:7,d} {bar}")
    
    # Check for optimization tables
    tables = {
        "hierarchy": "Hierarchy index",
        "centrality": "Centrality scores",
        "entries_fts": "Full-text search",
        "synonyms": "Synonym expansion",
        "entry_domains": "Domain clustering",
        "predicate_map": "Predicate normalization",
    }
    
    print(f"\n  Optimization tables:")
    for table, desc in tables.items():
        try:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            cnt = cur.fetchone()[0]
            print(f"    ✓ {desc:25s} {cnt:,d} entries")
        except (sqlite3.OperationalError, TypeError):
            print(f"    ✗ {desc:25s} (not built)")
    
    # Database file size
    cur.execute("PRAGMA page_count")
    pages = cur.fetchone()[0]
    cur.execute("PRAGMA page_size")
    page_size = cur.fetchone()[0]
    size_mb = (pages * page_size) / (1024*1024)
    print(f"\n  Database size: {size_mb:.1f} MB")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Knowledge Base Optimizer")
    parser.add_argument("--db", default="curriculum.db")
    parser.add_argument("--normalize", action="store_true", help="Normalize predicates")
    parser.add_argument("--hierarchy", action="store_true", help="Build hierarchy index")
    parser.add_argument("--centrality", action="store_true", help="Compute centrality scores")
    parser.add_argument("--fts", action="store_true", help="Build full-text search index")
    parser.add_argument("--synonyms", action="store_true", help="Build synonym table")
    parser.add_argument("--domains", action="store_true", help="Cluster by domain")
    parser.add_argument("--dedup", action="store_true", help="Remove duplicates")
    parser.add_argument("--integrity", action="store_true", help="Check data integrity")
    parser.add_argument("--report", action="store_true", help="Show database report")
    parser.add_argument("--all", action="store_true", help="Run ALL optimizations")
    args = parser.parse_args()
    
    print(f"{'═'*55}")
    print(f"  🔧 KNOWLEDGE BASE OPTIMIZER")
    print(f"{'═'*55}")
    
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM entries")
    total = cur.fetchone()[0]
    print(f"  Database: {args.db} ({total:,} entries)")
    
    run_all = args.all or not any([
        args.normalize, args.hierarchy, args.centrality,
        args.fts, args.synonyms, args.domains, args.dedup,
        args.integrity, args.report
    ])
    
    if args.report:
        report(conn)
        conn.close()
        return
    
    steps = []
    if run_all or args.integrity: steps.append(("Integrity check", lambda: integrity_check(conn)))
    if run_all or args.dedup:     steps.append(("Deduplication", lambda: deduplicate(conn)))
    if run_all or args.normalize: steps.append(("Predicate normalization", lambda: normalize_predicates(conn)))
    if run_all or args.dedup:     steps.append(("Post-normalize dedup", lambda: deduplicate(conn)))
    if run_all or args.hierarchy: steps.append(("Hierarchy index", lambda: build_hierarchy(conn)))
    if run_all or args.centrality:steps.append(("Centrality scoring", lambda: compute_centrality(conn)))
    if run_all or args.fts:       steps.append(("Full-text search index", lambda: build_fts_index(conn)))
    if run_all or args.synonyms:  steps.append(("Synonym expansion", lambda: build_synonyms(conn)))
    if run_all or args.domains:   steps.append(("Domain clustering", lambda: cluster_domains(conn)))
    
    import time
    for name, fn in steps:
        print(f"\n  ── {name} ──")
        t0 = time.time()
        fn()
        print(f"  ({time.time()-t0:.1f}s)")
    
    # Show final report
    report(conn)
    
    conn.close()
    print(f"\n  ✓ Optimization complete")

if __name__ == "__main__":
    main()
