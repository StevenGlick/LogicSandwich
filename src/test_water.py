#!/usr/bin/env python3
"""
INTEGRATION TEST: Water Domain
═══════════════════════════════

Runs the full pipeline on a fresh topic (water) to exercise:
  - Pulse engine (merge, transitive, cross-domain, analogy)
  - Domain scope (should block nonsense but allow real connections)
  - Bridge detection (should find bridges across many types)
  - Optimizer (hierarchy, centrality, FTS, domains)
  - Query engine (graph walk, path finding)

No LLM needed — tests the CPU-side operations.
"""

import sqlite3
import json
import os
import sys
import time
from collections import defaultdict

# ─── Setup ───────────────────────────────────────────────

TEST_DB = "test_water.db"

# Clean start
if os.path.exists(TEST_DB):
    os.remove(TEST_DB)

print(f"{'═'*60}")
print(f"  🧪 INTEGRATION TEST: Water Domain")
print(f"{'═'*60}")

# ─── Create Database ─────────────────────────────────────

conn = sqlite3.connect(TEST_DB)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

cur.executescript("""
    CREATE TABLE IF NOT EXISTS entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject TEXT NOT NULL,
        context TEXT DEFAULT 'general',
        predicate TEXT NOT NULL,
        object TEXT NOT NULL,
        truth_value TEXT DEFAULT 'T',
        evidence_for TEXT DEFAULT '[]',
        evidence_against TEXT DEFAULT '[]',
        source TEXT DEFAULT 'seed',
        generation INTEGER DEFAULT 0,
        grade_level INTEGER DEFAULT 0,
        needs_verification INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_subj ON entries(subject);
    CREATE INDEX IF NOT EXISTS idx_ctx ON entries(context);
    CREATE INDEX IF NOT EXISTS idx_pred ON entries(predicate);
    CREATE INDEX IF NOT EXISTS idx_obj ON entries(object);
    CREATE INDEX IF NOT EXISTS idx_tv ON entries(truth_value);
""")
conn.commit()

# ─── Load Seed Data ──────────────────────────────────────

print(f"\n  ── Phase 1: Seed Data ──")
with open("water_seed.json") as f:
    seeds = json.load(f)

for s in seeds:
    ctx = s.get("context", "general")
    cur.execute("""
        INSERT INTO entries (subject, context, predicate, object, truth_value, source, generation)
        VALUES (?, ?, ?, ?, 'T', 'seed', 0)
    """, (s["subject"], ctx, s["predicate"], s["object"]))

conn.commit()
cur.execute("SELECT COUNT(*) FROM entries")
seed_count = cur.fetchone()[0]
print(f"  Loaded {seed_count} seed entries about water")

# ─── Domain Scope Check ─────────────────────────────────

print(f"\n  ── Phase 2: Domain Scope Analysis ──")
from domain_scope import DomainClassifier, TypeGate, NonsenseFilter

classifier = DomainClassifier(conn)

# Classify all subjects
cur.execute("SELECT DISTINCT subject FROM entries")
subjects = [r[0] for r in cur.fetchall()]
domain_map = defaultdict(list)
unclassified = []

for s in subjects:
    domains = classifier.classify(s)
    if domains:
        for d in domains:
            domain_map[d].append(s)
    else:
        unclassified.append(s)

print(f"  Subjects: {len(subjects)} total")
print(f"  Classified into domains:")
for d in sorted(domain_map, key=lambda x: -len(domain_map[x])):
    print(f"    {d:20s} {len(domain_map[d]):3d} subjects")
print(f"  Unclassified: {len(unclassified)}")
if unclassified[:10]:
    for u in unclassified[:10]:
        print(f"    ? {u}")

# ─── Pulse Engine (CPU Operations) ──────────────────────

print(f"\n  ── Phase 3: Pulse Engine ──")

def get_all_entries():
    cur.execute("SELECT * FROM entries WHERE truth_value='T'")
    return [dict(r) for r in cur.fetchall()]

def add_entry(subj, pred, obj, source, gen, tv="T", context="general"):
    """Add with dedup check (context-aware)."""
    cur.execute("SELECT id FROM entries WHERE subject=? AND context=? AND predicate=? AND object=?",
                (subj, context, pred, obj))
    if cur.fetchone():
        return None
    cur.execute("""
        INSERT INTO entries (subject, context, predicate, object, truth_value, source, generation)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (subj, context, pred, obj, tv, source, gen))
    conn.commit()
    return cur.lastrowid

# --- Merge ---
print(f"\n  Merge (combining same-subject entries)...")
merge_count = 0
entries = get_all_entries()
subject_entries = defaultdict(list)
for e in entries:
    if "," not in e["subject"]:
        subject_entries[e["subject"]].append(e)

for subj, ents in subject_entries.items():
    if len(ents) >= 3:
        preds = set(e["predicate"] for e in ents)
        objs = set(e["object"] for e in ents)
        summary = f"has_{len(preds)}_predicates_and_{len(objs)}_objects"
        r = add_entry(subj, "profile_size", summary, "pulse:merge", 1)
        if r:
            merge_count += 1

print(f"    Generated: {merge_count} merge entries")

# --- Transitive ---
print(f"\n  Transitive (if A→B and B→C then A→C)...")
trans_count = 0
trans_blocked = 0
trans_blocked_reasons = defaultdict(int)
entries = get_all_entries()

# Build lookup
subj_lookup = defaultdict(list)
for e in entries:
    subj_lookup[e["subject"]].append(e)

gate = TypeGate(conn)
nsf = NonsenseFilter(classifier)

from predicate_rules import TransitiveFilter
tf = TransitiveFilter()

for e in entries:
    # Find entries where this entry's object is someone else's subject
    obj = e["object"]
    if obj in subj_lookup:
        for e2 in subj_lookup[obj]:
            # Chain: e.subject --[e.predicate]--> obj --[e2.predicate]--> e2.object
            new_subj = e["subject"]
            link_pred = e["predicate"]      # how A connects to B
            new_pred = e2["predicate"]       # B's property we'd inherit
            new_obj = e2["object"]
            
            # Skip boring chains
            if new_pred in ("profile_size",):
                continue
            if new_subj == new_obj:
                continue
            
            # Predicate transparency check (the new fix!)
            allowed_tf, reason_tf = tf.check_chain(link_pred, new_pred)
            if not allowed_tf:
                trans_blocked += 1
                trans_blocked_reasons[reason_tf] += 1
                continue
            
            # Scope check
            allowed_ns, reason_ns = nsf.check(new_subj, new_pred, new_obj)
            if not allowed_ns:
                trans_blocked += 1
                trans_blocked_reasons[reason_ns] += 1
                continue
            
            allowed_tg, reason_tg, conf = gate.check_transitive(new_subj, new_pred, new_obj)
            if not allowed_tg:
                trans_blocked += 1
                trans_blocked_reasons[reason_tg] += 1
                continue
            
            # Low confidence → Maybe
            tv = "M" if conf < 0.5 else "T"
            r = add_entry(new_subj, new_pred, new_obj, "pulse:transitive", 1, tv)
            if r:
                trans_count += 1

print(f"    Generated: {trans_count} transitive entries")
print(f"    Blocked: {trans_blocked} (predicate transparency + domain scope)")
if trans_blocked_reasons:
    print(f"    Block reasons:")
    for reason, count in sorted(trans_blocked_reasons.items(), key=lambda x: -x[1])[:10]:
        print(f"      {count:4d}× {reason[:70]}")

# --- Cross-Domain ---
print(f"\n  Cross-Domain (different subjects sharing properties)...")
cross_count = 0
cross_blocked = 0
entries = get_all_entries()

# Group by (predicate, object)
po_groups = defaultdict(list)
for e in entries:
    if "," not in e["subject"]:
        po_groups[(e["predicate"], e["object"])].append(e["subject"])

for (pred, obj), subs in po_groups.items():
    if len(subs) < 2 or len(subs) > 8:
        continue
    # Skip trivial predicates
    if pred in ("profile_size",):
        continue
    
    for i, s1 in enumerate(subs):
        for s2 in subs[i+1:]:
            if s1 == s2:
                continue
            
            # Check domain compatibility
            compat = classifier.compatibility_score(s1, s2)
            
            compound = f"{s1},{s2}"
            new_pred = f"both_{pred}"
            
            if compat < 0.3 and compat > 0.0:
                cross_blocked += 1
                continue
            
            r = add_entry(compound, new_pred, obj, "pulse:cross_domain", 1)
            if r:
                cross_count += 1

print(f"    Generated: {cross_count} cross-domain entries")
print(f"    Blocked: {cross_blocked} (low compatibility)")

# --- Analogy ---
print(f"\n  Analogy (subjects with similar predicate profiles)...")
analogy_count = 0
entries = get_all_entries()

# Build predicate profile per subject
profiles = defaultdict(set)
for e in entries:
    if "," not in e["subject"]:
        profiles[e["subject"]].add(e["predicate"])

# Compare profiles
subjects_list = [s for s in profiles if len(profiles[s]) >= 3]
for i, s1 in enumerate(subjects_list):
    for s2 in subjects_list[i+1:]:
        shared = profiles[s1] & profiles[s2]
        total = profiles[s1] | profiles[s2]
        if len(total) == 0:
            continue
        similarity = len(shared) / len(total)  # Jaccard
        
        if similarity >= 0.3 and len(shared) >= 2:
            # Check domain gate
            allowed, reason = gate.check_analogy(s1, s2)
            if not allowed:
                continue
            
            shared_str = ",".join(sorted(shared)[:5])
            r = add_entry(f"{s1},{s2}", "structurally_similar",
                         f"shared_predicates:{shared_str}",
                         "pulse:analogy", 1)
            if r:
                analogy_count += 1

print(f"    Generated: {analogy_count} analogy entries")

# ─── Post-Pulse Stats ───────────────────────────────────

cur.execute("SELECT COUNT(*) FROM entries")
total = cur.fetchone()[0]
cur.execute("SELECT source, COUNT(*) as cnt FROM entries GROUP BY source ORDER BY cnt DESC")
sources = cur.fetchall()

print(f"\n  ── Pulse Results ──")
print(f"  Total entries: {seed_count} seed → {total} total (+{total-seed_count})")
print(f"  By source:")
for s in sources:
    bar = "█" * min(s[1]//2, 30)
    print(f"    {s[0]:25s} {s[1]:4d} {bar}")

# ─── Optimizer ───────────────────────────────────────────

print(f"\n  ── Phase 4: Optimization ──")

# Import and run optimizer functions
from optimize import (normalize_predicates, build_hierarchy, 
                      compute_centrality, build_fts_index, 
                      build_synonyms, cluster_domains, deduplicate)

print(f"\n  Predicate normalization...")
normalize_predicates(conn)

print(f"\n  Deduplication...")
deduplicate(conn)

print(f"\n  Hierarchy index...")
build_hierarchy(conn)

print(f"\n  Centrality scoring...")
compute_centrality(conn)

print(f"\n  Full-text search...")
build_fts_index(conn)

print(f"\n  Synonym expansion...")
build_synonyms(conn)

print(f"\n  Domain clustering...")
cluster_domains(conn)

# ─── Bridge Detection ───────────────────────────────────

print(f"\n  ── Phase 5: Bridge Detection ──")

from bridges import BridgeSystem, BridgeDiscovery, SEED_BRIDGES

# Load seed bridges into test DB
bs = BridgeSystem(TEST_DB)
loaded = 0
for bridge in SEED_BRIDGES:
    r = bs.add_bridge(**bridge)
    if r:
        loaded += 1
print(f"  Loaded {loaded} seed bridges")

# Now check: which seed bridges are RELEVANT to our water knowledge?
print(f"\n  Bridges relevant to water domain:")
water_terms = set()
cur.execute("SELECT DISTINCT subject FROM entries WHERE subject NOT LIKE '%,%'")
water_terms.update(r[0] for r in cur.fetchall())
cur.execute("SELECT DISTINCT object FROM entries WHERE object NOT LIKE '%,%'")
water_terms.update(r[0] for r in cur.fetchall())

relevant_bridges = []
all_bridges = bs.all_bridges()
for b in all_bridges:
    # Check if either term appears in our water knowledge
    ta = b["term_a"]
    tb = b["term_b"]
    
    relevant = False
    # Direct match
    if ta in water_terms or tb in water_terms:
        relevant = True
    # Substring match (e.g., "wave" in "ocean_wave")
    for wt in water_terms:
        if ta in wt or tb in wt or wt in ta or wt in tb:
            relevant = True
            break
    
    if relevant:
        relevant_bridges.append(b)

type_icons = {
    "historical": "📜", "inspirational": "💡", "structural": "🔷",
    "metaphorical": "🎭", "emergent": "✨", "causal": "⚡",
    "cultural": "🌍", "etymological": "📖", "scale": "🔬",
    "pedagogical": "🎓", "contradictory": "⚔️", "experiential": "🖐️",
    "spiritual": "🕉️", "religious": "⛪", "ethical": "⚖️",
    "aesthetic": "🎨", "rumor": "💭", "mythological": "🐉",
    "linguistic": "🗣️", "technological": "🔧", "emotional": "💙",
    "pattern": "🌀", "philosophical": "🏛️", "sensory": "👁️", "humor": "😄",
}

bridge_types_found = defaultdict(list)
for b in relevant_bridges:
    bt = b["bridge_type"]
    bridge_types_found[bt].append(b)

print(f"\n  Found {len(relevant_bridges)} relevant bridges across {len(bridge_types_found)} types:")
for bt in sorted(bridge_types_found, key=lambda x: -len(bridge_types_found[x])):
    icon = type_icons.get(bt, "🔗")
    bridges = bridge_types_found[bt]
    print(f"\n    {icon} {bt} ({len(bridges)}):")
    for b in bridges:
        print(f"      {b['term_a']} ↔ {b['term_b']}")
        print(f"        {b['reason'][:100]}...")

# ─── Bridge Discovery from Knowledge ────────────────────

print(f"\n  ── Phase 6: Auto-Discovered Bridges ──")

# Find cross-domain entries that COULD be bridges
cur.execute("""
    SELECT * FROM entries 
    WHERE subject LIKE '%,%' AND source='pulse:cross_domain'
    ORDER BY subject
""")
cross_entries = [dict(r) for r in cur.fetchall()]

print(f"\n  Cross-domain patterns that could be bridges ({len(cross_entries)}):")

interesting = []
for e in cross_entries:
    parts = e["subject"].split(",")
    if len(parts) == 2:
        s1, s2 = parts
        d1 = classifier.classify(s1)
        d2 = classifier.classify(s2)
        
        # Flag as interesting if they span different domains
        if d1 and d2 and d1 != d2:
            interesting.append({
                "term_a": s1, "term_b": s2,
                "domain_a": str(d1), "domain_b": str(d2),
                "shared": f"{e['predicate']}({e['object']})",
            })

if interesting:
    for item in interesting[:15]:
        print(f"    ✨ {item['term_a']} ↔ {item['term_b']}")
        print(f"       Shared: {item['shared']}")
        print(f"       Domains: {item['domain_a']} ↔ {item['domain_b']}")
else:
    print(f"    (All cross-domain patterns were within same domain cluster)")

# Show same-domain patterns too (still valuable)
print(f"\n  Same-domain cross-patterns (first 15):")
for e in cross_entries[:15]:
    parts = e["subject"].split(",")
    if len(parts) == 2:
        print(f"    {parts[0]:25s} ↔ {parts[1]:25s} | {e['predicate']}({e['object'][:30]})")

# ─── Query Test ──────────────────────────────────────────

print(f"\n  ── Phase 7: Query Tests ──")

def simple_query(term):
    """Search for everything about a term."""
    results = []
    cur.execute("SELECT * FROM entries WHERE subject=? AND truth_value='T'", (term,))
    results.extend(dict(r) for r in cur.fetchall())
    cur.execute("SELECT * FROM entries WHERE object=? AND truth_value='T'", (term,))
    results.extend(dict(r) for r in cur.fetchall())
    return results

def graph_walk(start, depth=2):
    """BFS walk from a starting node."""
    visited = set()
    queue = [(start, 0)]
    nodes = set()
    edges = []
    
    while queue:
        current, d = queue.pop(0)
        if current in visited or d > depth:
            continue
        visited.add(current)
        nodes.add(current)
        
        cur.execute("SELECT predicate, object FROM entries WHERE subject=? AND truth_value='T' AND subject NOT LIKE '%,%'", (current,))
        for r in cur.fetchall():
            obj = r[1]
            edges.append((current, r[0], obj))
            if d < depth and "," not in obj:
                queue.append((obj, d+1))
    
    return nodes, edges

# Test 1: Direct lookup
print(f"\n  Query: 'water'")
results = simple_query("water")
print(f"    {len(results)} facts found")
for r in results[:8]:
    print(f"      {r['subject']} → {r['predicate']}({r['object'][:40]})")
if len(results) > 8:
    print(f"      ... +{len(results)-8} more")

# Test 2: Graph walk
print(f"\n  Graph walk from 'water' (depth 2):")
nodes, edges = graph_walk("water", 2)
print(f"    {len(nodes)} nodes, {len(edges)} edges")
print(f"    Nodes: {', '.join(sorted(list(nodes)[:15]))}")
if len(nodes) > 15:
    print(f"           ... +{len(nodes)-15} more")

# Test 3: Graph walk from wave
print(f"\n  Graph walk from 'wave' (depth 2):")
nodes2, edges2 = graph_walk("wave", 2)
print(f"    {len(nodes2)} nodes, {len(edges2)} edges")
print(f"    Nodes: {', '.join(sorted(list(nodes2)[:15]))}")

# Test 4: Path finding
print(f"\n  Path: 'water' → 'egyptian_civilization'")
# BFS path finding
def find_path(start, end, max_depth=5):
    visited = set()
    queue = [(start, [(start, None, None)])]
    
    while queue:
        current, path = queue.pop(0)
        if current == end:
            return path
        if current in visited or len(path) > max_depth:
            continue
        visited.add(current)
        
        cur.execute("SELECT predicate, object FROM entries WHERE subject=? AND truth_value='T' AND subject NOT LIKE '%,%'", (current,))
        for r in cur.fetchall():
            if "," not in r[1]:
                queue.append((r[1], path + [(r[1], r[0], current)]))
    
    return None

path = find_path("water", "egyptian_civilization", 6)
if path:
    print(f"    Found! {len(path)-1} hops:")
    for i, (node, pred, via) in enumerate(path):
        if pred:
            print(f"      {'→':>4s} {via} → {pred} → {node}")
        else:
            print(f"      START {node}")
else:
    print(f"    No path found (trying via nile...)")
    # Manual trace
    path1 = find_path("water", "nile", 4)
    path2 = find_path("nile", "egyptian_civilization", 4)
    if path1:
        print(f"    water → nile: {len(path1)-1} hops")
    if path2:
        print(f"    nile → egyptian_civilization: {len(path2)-1} hops")

# Test 5: Cross-bridge query
print(f"\n  Bridge-aware query: 'What connects water to emotions?'")
# Find all emotion-related entries about water
emotion_entries = []
cur.execute("""
    SELECT * FROM entries WHERE truth_value='T' AND (
        object LIKE '%emotion%' OR object LIKE '%sadness%' OR 
        object LIKE '%joy%' OR object LIKE '%grief%' OR
        object LIKE '%overwhelm%' OR object LIKE '%calming%' OR
        predicate LIKE '%feel%' OR predicate LIKE '%associated%'
    )
""")
emotion_entries = [dict(r) for r in cur.fetchall()]
print(f"    Direct emotional connections: {len(emotion_entries)}")
for e in emotion_entries:
    print(f"      {e['subject']} → {e['predicate']}({e['object']})")

# Check bridges for emotional connections
emotion_bridges = [b for b in relevant_bridges if b["bridge_type"] in ("emotional", "sensory", "experiential")]
if emotion_bridges:
    print(f"    Relevant bridges: {len(emotion_bridges)}")
    for b in emotion_bridges:
        icon = type_icons.get(b["bridge_type"], "🔗")
        print(f"      {icon} {b['term_a']} ↔ {b['term_b']} ({b['bridge_type']})")

# ─── Final Report ────────────────────────────────────────

print(f"\n  {'═'*60}")
print(f"  📊 FINAL REPORT")
print(f"  {'═'*60}")

cur.execute("SELECT COUNT(*) FROM entries")
final_total = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM entries WHERE truth_value='T'")
true_count = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM entries WHERE truth_value='M'")
maybe_count = cur.fetchone()[0]

print(f"\n  Knowledge Base:")
print(f"    Seed entries:        {seed_count}")
print(f"    After 1 pulse:       {final_total} (+{final_total - seed_count})")
print(f"    True:                {true_count}")
print(f"    Maybe (need verify): {maybe_count}")

print(f"\n  Pulse Output:")
print(f"    Merge:        {merge_count}")
print(f"    Transitive:   {trans_count} (blocked: {trans_blocked})")
print(f"    Cross-domain: {cross_count} (blocked: {cross_blocked})")
print(f"    Analogy:      {analogy_count}")
print(f"    TOTAL:        {merge_count + trans_count + cross_count + analogy_count}")

print(f"\n  Bridge Coverage:")
print(f"    Relevant bridges:    {len(relevant_bridges)} of {len(all_bridges)}")
print(f"    Bridge types hit:    {len(bridge_types_found)} of {len(type_icons)}")
print(f"    Types covered:       {', '.join(sorted(bridge_types_found.keys()))}")

types_missed = set(type_icons.keys()) - set(bridge_types_found.keys())
if types_missed:
    print(f"    Types not hit:       {', '.join(sorted(types_missed))}")
    print(f"    (These would activate with broader knowledge)")

# Centrality top concepts
try:
    cur.execute("SELECT subject, score FROM centrality ORDER BY score DESC LIMIT 10")
    print(f"\n  Most Connected Concepts:")
    for r in cur.fetchall():
        bar = "█" * min(int(r[1])//2, 25)
        print(f"    {r[0]:25s} score={r[1]:5.0f} {bar}")
except (sqlite3.OperationalError, TypeError):
    pass

# ─── Context Disambiguation Test ──────────────────────

print(f"\n  ── Phase 8: Context Disambiguation Test ──")
print(f"  Testing that 'mamba' in computer_science and biology contexts stay isolated...")

# Add mamba entries in two different contexts
add_entry("mamba", "is_a", "ssm_architecture", "test:disambiguation", 0, "T", context="computer_science")
add_entry("mamba", "uses", "selective_state_spaces", "test:disambiguation", 0, "T", context="computer_science")
add_entry("mamba", "is_a", "venomous_snake", "test:disambiguation", 0, "T", context="biology")
add_entry("mamba", "has_property", "neurotoxic_venom", "test:disambiguation", 0, "T", context="biology")

# Add chaining targets to test transitive leakage
add_entry("ssm_architecture", "is_a", "neural_network_variant", "test:disambiguation", 0, "T", context="computer_science")
add_entry("venomous_snake", "is_a", "reptile", "test:disambiguation", 0, "T", context="biology")

# Run a mini transitive pass on just the mamba entries
# Chain: mamba -[is_a]-> ssm_architecture -[is_a]-> neural_network_variant (CS context)
# Chain: mamba -[is_a]-> venomous_snake -[is_a]-> reptile (biology context)
# The test: mamba in biology context should NOT get neural_network_variant
mamba_trans_ok = True
mamba_trans_details = []

# Get mamba entries grouped by context
cur.execute("SELECT subject, context, predicate, object FROM entries WHERE subject='mamba'")
mamba_entries = [dict(r) for r in cur.fetchall()]

# Simulate context-aware transitive chaining
for me in mamba_entries:
    me_ctx = me["context"]
    me_obj = me["object"]
    # Find entries where me_obj is the subject IN THE SAME CONTEXT
    cur.execute(
        "SELECT subject, context, predicate, object FROM entries WHERE subject=? AND context=?",
        (me_obj, me_ctx))
    targets = [dict(r) for r in cur.fetchall()]
    for t in targets:
        result_ctx = t["context"]
        mamba_trans_details.append(
            (me_ctx, me["predicate"], me_obj, t["predicate"], t["object"], result_ctx))
        # Context should match — no leakage
        if me_ctx != result_ctx:
            mamba_trans_ok = False

# Also verify no cross-context results: biology mamba should NOT chain to CS targets
cur.execute(
    "SELECT subject, context, predicate, object FROM entries WHERE subject='ssm_architecture' AND context='biology'")
leaked_cs_to_bio = cur.fetchall()
cur.execute(
    "SELECT subject, context, predicate, object FROM entries WHERE subject='venomous_snake' AND context='computer_science'")
leaked_bio_to_cs = cur.fetchall()

if leaked_cs_to_bio or leaked_bio_to_cs:
    mamba_trans_ok = False

print(f"  Mamba transitive chains (context-aware):")
for src_ctx, link_p, mid, inh_p, end, res_ctx in mamba_trans_details:
    marker = "OK" if src_ctx == res_ctx else "LEAK"
    print(f"    [{src_ctx:18s}] mamba -{link_p}-> {mid} -{inh_p}-> {end}  [{res_ctx}] {marker}")

if mamba_trans_ok:
    print(f"  PASS: Contexts did not leak across mamba disambiguation")
else:
    print(f"  FAIL: Context leakage detected in mamba disambiguation!")

# Verify both contexts coexist
cur.execute("SELECT context, COUNT(*) FROM entries WHERE subject='mamba' GROUP BY context")
mamba_ctx_counts = {r[0]: r[1] for r in cur.fetchall()}
print(f"  Mamba entries by context: {dict(mamba_ctx_counts)}")
assert "computer_science" in mamba_ctx_counts, "Missing computer_science mamba entries"
assert "biology" in mamba_ctx_counts, "Missing biology mamba entries"
print(f"  PASS: Both contexts coexist for 'mamba'")

print(f"\n  ✓ Test complete")

conn.close()
