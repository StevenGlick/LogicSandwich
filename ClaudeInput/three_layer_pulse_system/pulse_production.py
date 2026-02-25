#!/usr/bin/env python3
"""
THREE-LAYER PULSE SYSTEM — Production Starter
═══════════════════════════════════════════════

REQUIREMENTS:
  pip install requests       (for talking to Ollama and web search)
  Install Ollama:            https://ollama.com
  Pull a model:              ollama pull llama3.2
                             (or whatever fits your GPU)

WHAT THIS DOES:
  1. Stores knowledge in SQLite (a file-based database)
  2. Pulse engine runs on CPU (fast symbolic logic)
  3. Sends verification requests to Ollama (GPU-bound LLM)
  4. Loops forever, growing the knowledge base

USAGE:
  python3 pulse_production.py                     # run with defaults
  python3 pulse_production.py --model mistral      # use a different model
  python3 pulse_production.py --topic "quantum computing"  # seed a topic
  python3 pulse_production.py --cycles 100         # run 100 pulse cycles

FILES IT CREATES:
  knowledge.db    — SQLite database (your knowledge base)
  pulse_log.txt   — Human-readable log of what happened
"""

import sqlite3
import json
import time
import argparse
import sys
from collections import defaultdict
from itertools import combinations

# ═══════════════════════════════════════════════════════════
# TRY TO IMPORT REQUESTS (for Ollama + web search)
# If not installed, we fall back to simulated mode
# ═══════════════════════════════════════════════════════════
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("⚠ 'requests' not installed. Running in OFFLINE mode.")
    print("  Install with: pip install requests")
    print("  (Ollama and web search will be simulated)")
    print()


# ═══════════════════════════════════════════════════════════
# DATABASE LAYER (SQLite)
# ═══════════════════════════════════════════════════════════
# SQLite is a database that lives in a single file.
# Think of it like a super-powered JSON file that can
# handle millions of entries and lets you search fast.
# Python includes it by default — no install needed.
# ═══════════════════════════════════════════════════════════

class KnowledgeDB:
    """
    Wrapper around SQLite that stores our logic entries.
    
    Each entry is one atomic fact:
      subject → predicate(object) : truth_value
    
    Plus metadata: evidence, source, generation, etc.
    """
    
    def __init__(self, db_path="knowledge.db"):
        # Connect to (or create) the database file
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row  # lets us access columns by name
        self.cursor = self.conn.cursor()
        self._create_tables()
        print(f"  Database: {db_path}")
    
    def _create_tables(self):
        """Create the tables if they don't exist yet."""
        self.cursor.executescript("""
            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                quantifier TEXT DEFAULT 'universal',
                truth_value TEXT DEFAULT 'M',
                evidence_for TEXT DEFAULT '[]',
                evidence_against TEXT DEFAULT '[]',
                source TEXT DEFAULT 'unknown',
                generation INTEGER DEFAULT 0,
                verified INTEGER DEFAULT 0,
                needs_verification INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Indexes make searches fast
            -- (like a table of contents for the database)
            CREATE INDEX IF NOT EXISTS idx_subject ON entries(subject);
            CREATE INDEX IF NOT EXISTS idx_predicate ON entries(predicate);
            CREATE INDEX IF NOT EXISTS idx_truth ON entries(truth_value);
            CREATE INDEX IF NOT EXISTS idx_source ON entries(source);
            CREATE INDEX IF NOT EXISTS idx_sig ON entries(subject, predicate, object);
        """)
        self.conn.commit()
    
    def add(self, subject, predicate, obj, truth_value="M", 
            evidence_for=None, evidence_against=None,
            source="unknown", generation=0, needs_verification=False):
        """Add an entry if it doesn't already exist."""
        # Check for duplicate
        self.cursor.execute(
            "SELECT id FROM entries WHERE subject=? AND predicate=? AND object=?",
            (subject, predicate, obj))
        if self.cursor.fetchone():
            return None  # already exists
        
        self.cursor.execute("""
            INSERT INTO entries (subject, predicate, object, truth_value,
                evidence_for, evidence_against, source, generation, needs_verification)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            subject, predicate, obj, truth_value,
            json.dumps(evidence_for or []),
            json.dumps(evidence_against or []),
            source, generation, int(needs_verification)
        ))
        self.conn.commit()
        return self.cursor.lastrowid
    
    def find(self, **kwargs):
        """Search entries. Example: db.find(subject="transformer", truth_value="T")"""
        conditions = []
        params = []
        for key, val in kwargs.items():
            conditions.append(f"{key} = ?")
            params.append(val)
        
        where = " AND ".join(conditions) if conditions else "1=1"
        self.cursor.execute(f"SELECT * FROM entries WHERE {where}", params)
        return [dict(row) for row in self.cursor.fetchall()]
    
    def all_entries(self, truth_value=None):
        """Get all entries, optionally filtered by truth value."""
        if truth_value:
            self.cursor.execute("SELECT * FROM entries WHERE truth_value=?", (truth_value,))
        else:
            self.cursor.execute("SELECT * FROM entries")
        return [dict(row) for row in self.cursor.fetchall()]
    
    def update_truth(self, entry_id, new_truth, evidence=None):
        """Update truth value and optionally add evidence."""
        self.cursor.execute(
            "UPDATE entries SET truth_value=?, verified=1, needs_verification=0 WHERE id=?",
            (new_truth, entry_id))
        if evidence:
            row = self.find(id=entry_id)[0]
            ev = json.loads(row["evidence_for"])
            ev.append(evidence)
            self.cursor.execute(
                "UPDATE entries SET evidence_for=? WHERE id=?",
                (json.dumps(ev), entry_id))
        self.conn.commit()
    
    def count(self, truth_value=None):
        if truth_value:
            self.cursor.execute("SELECT COUNT(*) FROM entries WHERE truth_value=?", (truth_value,))
        else:
            self.cursor.execute("SELECT COUNT(*) FROM entries")
        return self.cursor.fetchone()[0]
    
    def current_generation(self):
        self.cursor.execute("SELECT MAX(generation) FROM entries")
        result = self.cursor.fetchone()[0]
        return (result or 0) + 1
    
    def signatures(self):
        """Get all (subject, predicate, object) triples for dedup."""
        self.cursor.execute("SELECT subject, predicate, object FROM entries")
        return {(r[0], r[1], r[2]) for r in self.cursor.fetchall()}
    
    def stats(self):
        """Print a summary of the database."""
        total = self.count()
        t = self.count("T")
        f = self.count("F")
        m = self.count("M")
        self.cursor.execute("SELECT source, COUNT(*) FROM entries GROUP BY source ORDER BY COUNT(*) DESC")
        sources = self.cursor.fetchall()
        
        print(f"\n  ═══ KB STATS ═══")
        print(f"  Total: {total} | T={t} F={f} M={m}")
        print(f"  Sources:")
        for src, cnt in sources:
            bar = "█" * min(cnt, 40)
            print(f"    {src:25s} {cnt:4d} {bar}")


# ═══════════════════════════════════════════════════════════
# LLM LAYER (Ollama)
# ═══════════════════════════════════════════════════════════
# Ollama runs a local LLM on your GPU.
# We talk to it over HTTP (like a web request, but to
# localhost — it never leaves your machine).
# ═══════════════════════════════════════════════════════════

class OllamaLLM:
    """
    Talks to a local Ollama instance.
    
    Ollama must be running: just open a terminal and type 'ollama serve'
    (or it runs automatically on install).
    
    We send it prompts, it sends back text. That's the whole interface.
    """
    
    def __init__(self, model="llama3.2", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.available = self._check_connection()
        if self.available:
            print(f"  LLM: {model} via Ollama ✓")
        else:
            print(f"  LLM: Ollama not available — running in SIMULATION mode")
            print(f"       (install Ollama and run 'ollama pull {model}' to enable)")
    
    def _check_connection(self):
        """See if Ollama is actually running."""
        if not HAS_REQUESTS:
            return False
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return r.status_code == 200
        except:
            return False
    
    def ask(self, prompt, system=None):
        """
        Send a prompt to the LLM, get back text.
        
        This is the fundamental operation: text in, text out.
        Everything the LLM layer does goes through this function.
        """
        if not self.available:
            return None  # caller handles simulation
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        try:
            # This is an HTTP POST request to localhost
            # The LLM runs on your GPU and responds
            r = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,  # wait for complete response
                    "options": {"temperature": 0.3}  # low = more precise
                },
                timeout=120  # give it up to 2 minutes
            )
            if r.status_code == 200:
                return r.json()["message"]["content"]
        except Exception as e:
            print(f"    ⚠ LLM error: {e}")
        return None
    
    def verify_entry(self, subject, predicate, obj, evidence):
        """
        Ask the LLM to verify a logic entry.
        Returns: ("T", "F", or "M") and reasoning.
        """
        system = """You are a logic verification system. You evaluate whether 
        claims are True (T), False (F), or Maybe (M - needs more evidence).
        
        Respond in this exact format:
        VERDICT: T/F/M
        REASONING: one paragraph explaining why
        NEW_FACTS: (optional) any new atomic facts discovered during reasoning,
        one per line in format: subject|predicate|object|T/F/M"""
        
        prompt = f"""Evaluate this claim:
        
Subject: {subject}
Predicate: {predicate}
Object: {obj}
Current evidence: {evidence}

Is this claim True, False, or Maybe (insufficient evidence)?"""
        
        response = self.ask(prompt, system)
        if not response:
            return "M", "LLM unavailable", []
        
        # Parse the response
        verdict = "M"
        reasoning = response
        new_facts = []
        
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("VERDICT:"):
                v = line.split(":")[1].strip().upper()
                if v in ("T", "F", "M"):
                    verdict = v
            elif "|" in line and line.count("|") >= 3:
                parts = line.split("|")
                if len(parts) >= 4:
                    new_facts.append({
                        "subject": parts[0].strip(),
                        "predicate": parts[1].strip(),
                        "object": parts[2].strip(),
                        "truth_value": parts[3].strip().upper()
                    })
        
        return verdict, reasoning, new_facts


# ═══════════════════════════════════════════════════════════
# PULSE ENGINE (CPU)
# ═══════════════════════════════════════════════════════════
# This is the Logic + Idea layer.
# Pure symbolic operations — runs entirely on CPU.
# ═══════════════════════════════════════════════════════════

CAP = 50  # max new entries per operation per pulse

class PulseEngine:
    def __init__(self, db):
        self.db = db
    
    def pulse(self):
        """Run one complete pulse cycle. Returns count of new entries."""
        gen = self.db.current_generation()
        ex = self.db.signatures()
        total_new = 0
        
        ops = [
            ("merge", self._merge),
            ("transitive", self._transitive),
            ("cross_domain", self._cross_domain),
            ("analogy", self._analogy),
        ]
        
        for name, fn in ops:
            count = fn(gen, ex)
            total_new += count
        
        return total_new
    
    def _merge(self, gen, existing):
        """Combine same-subject same-predicate entries."""
        entries = self.db.all_entries("T")
        groups = defaultdict(list)
        for e in entries:
            if "," not in e["object"]:
                groups[(e["subject"], e["predicate"])].append(e["object"])
        
        count = 0
        for (subj, pred), objects in groups.items():
            unique = sorted(set(objects))
            if len(unique) < 2:
                continue
            merged = ",".join(unique)
            if (subj, pred, merged) in existing:
                continue
            
            self.db.add(subj, pred, merged, "T",
                evidence_for=[f"merged {len(unique)} entries"],
                source="idea:merge", generation=gen)
            existing.add((subj, pred, merged))
            count += 1
            if count >= CAP:
                break
        return count
    
    def _transitive(self, gen, existing):
        """If A→B and B→C, propose A→C."""
        entries = self.db.all_entries("T")
        skip = {"structurally_similar_to", "commonly_confused_with",
                "analogous_to", "inspired_by", "defined_by"}
        
        # Build lookup: subject → [(predicate, object)]
        by_subject = defaultdict(list)
        for e in entries:
            if "," not in e["object"] and e["predicate"] not in skip:
                by_subject[e["subject"]].append(e)
        
        count = 0
        for e in entries:
            if "," in e["object"] or e["predicate"] in skip:
                continue
            obj = e["object"]
            if obj not in by_subject:
                continue
            for e2 in by_subject[obj]:
                if e2["predicate"] in skip or "," in e2["object"]:
                    continue
                sig = (e["subject"], e2["predicate"], e2["object"])
                if sig in existing or e["subject"] == e2["object"]:
                    continue
                
                self.db.add(
                    e["subject"], e2["predicate"], e2["object"], "M",
                    evidence_for=[f"transitive: {e['subject']}→{obj}→{e2['object']}"],
                    source="idea:transitive", generation=gen,
                    needs_verification=True)
                existing.add(sig)
                count += 1
                if count >= CAP:
                    return count
        return count
    
    def _cross_domain(self, gen, existing):
        """Find where different subjects share exact same predicate+object."""
        entries = self.db.all_entries("T")
        skip = {"is_a", "structurally_similar_to", "commonly_confused_with"}
        
        po_groups = defaultdict(list)
        for e in entries:
            if "," not in e["object"] and e["predicate"] not in skip:
                po_groups[(e["predicate"], e["object"])].append(e["subject"])
        
        count = 0
        for (pred, obj), subjects in po_groups.items():
            unique_subjects = sorted(set(subjects))
            if len(unique_subjects) < 2:
                continue
            for s1, s2 in combinations(unique_subjects, 2):
                combined = f"{s1},{s2}"
                sig = (combined, f"both_{pred}", obj)
                if sig in existing:
                    continue
                self.db.add(combined, f"both_{pred}", obj, "T",
                    evidence_for=["cross-domain pattern"],
                    source="idea:cross_domain", generation=gen)
                existing.add(sig)
                count += 1
                if count >= CAP:
                    return count
        return count
    
    def _analogy(self, gen, existing):
        """Find structurally similar subjects (share predicate types)."""
        entries = self.db.all_entries("T")
        skip = {"structurally_similar_to", "commonly_confused_with", "analogous_to"}
        
        profiles = defaultdict(set)
        for e in entries:
            if not e["subject"].startswith("category(") and e["predicate"] not in skip:
                profiles[e["subject"]].add(e["predicate"])
        
        count = 0
        subjects = sorted(profiles.keys())
        for s1, s2 in combinations(subjects, 2):
            shared = profiles[s1] & profiles[s2] - skip
            total = (profiles[s1] | profiles[s2]) - skip
            if not total:
                continue
            sim = len(shared) / len(total)
            if sim >= 0.6:
                sig = (s1, "structurally_similar_to", s2)
                if sig in existing:
                    continue
                self.db.add(s1, "structurally_similar_to", s2, "T",
                    evidence_for=[f"similarity={sim:.0%}, shared={sorted(shared)}"],
                    source="idea:analogy", generation=gen)
                existing.add(sig)
                count += 1
                if count >= CAP:
                    return count
        return count


# ═══════════════════════════════════════════════════════════
# VERIFICATION LOOP (bridges CPU and GPU)
# ═══════════════════════════════════════════════════════════

class Verifier:
    def __init__(self, db, llm):
        self.db = db
        self.llm = llm
    
    def verify_batch(self, batch_size=10):
        """
        Take unverified entries, send them to the LLM for verification.
        This is where CPU work meets GPU work.
        """
        # Find entries that need verification
        unverified = self.db.find(needs_verification=1)
        if not unverified:
            return 0
        
        batch = unverified[:batch_size]
        verified = 0
        new_entries = 0
        
        for entry in batch:
            evidence = json.loads(entry["evidence_for"])
            
            if self.llm.available:
                # REAL LLM VERIFICATION (GPU)
                verdict, reasoning, new_facts = self.llm.verify_entry(
                    entry["subject"], entry["predicate"], 
                    entry["object"], evidence)
                
                self.db.update_truth(entry["id"], verdict, f"LLM: {reasoning[:100]}")
                
                # Any new facts the LLM discovered get added
                gen = self.db.current_generation()
                for nf in new_facts:
                    added = self.db.add(
                        nf["subject"], nf["predicate"], nf["object"],
                        nf.get("truth_value", "M"),
                        source="llm:decomposition", generation=gen)
                    if added:
                        new_entries += 1
            else:
                # SIMULATION MODE (no GPU)
                # Use simple heuristics instead of real LLM
                safe_preds = {"is_a", "uses", "has_component", "enables",
                              "requires", "outputs", "processes"}
                if entry["predicate"] in safe_preds:
                    self.db.update_truth(entry["id"], "T", "auto: safe predicate chain")
                # else leave as M
                self.db.cursor.execute(
                    "UPDATE entries SET needs_verification=0 WHERE id=?",
                    (entry["id"],))
                self.db.conn.commit()
            
            verified += 1
        
        return verified


# ═══════════════════════════════════════════════════════════
# SEED LOADER (populate initial knowledge)
# ═══════════════════════════════════════════════════════════

def seed_from_json(db, json_path):
    """Load seed entries from a JSON knowledge base file."""
    with open(json_path) as f:
        data = json.load(f)
    
    count = 0
    for entry in data["entries"]:
        added = db.add(
            entry["subject"], entry["predicate"], entry["object"],
            entry.get("truth_value", "T"),
            evidence_for=entry.get("evidence_for", []),
            evidence_against=entry.get("evidence_against", []),
            source=entry.get("source", "seed"),
            generation=0)
        if added:
            count += 1
    
    print(f"  Seeded {count} entries from {json_path}")
    return count

def seed_from_topic(db, llm, topic):
    """
    Use the LLM to generate seed entries about a topic.
    This is the cold-start: you give it a topic, it creates the initial KB.
    """
    if not llm.available:
        print(f"  ⚠ Can't seed from topic without LLM. Use a JSON file instead.")
        return 0
    
    system = """You are a knowledge extraction system. Given a topic, produce 
    20-30 atomic facts in this exact format, one per line:
    subject|predicate|object|T
    
    Rules:
    - Each fact must be atomic (one relationship only)
    - Use lowercase_with_underscores for all terms
    - Predicates should be things like: is_a, uses, has_component, requires,
      enables, produces, suffers_from, faster_than, etc.
    - Only output the facts, no explanations"""
    
    prompt = f"Generate atomic knowledge facts about: {topic}"
    
    response = llm.ask(prompt, system)
    if not response:
        return 0
    
    count = 0
    for line in response.strip().split("\n"):
        line = line.strip().strip("-•* ")
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3:
            tv = parts[3] if len(parts) > 3 and parts[3] in ("T","F","M") else "T"
            added = db.add(parts[0], parts[1], parts[2], tv,
                evidence_for=["LLM seed generation"],
                source="llm:seed", generation=0)
            if added:
                count += 1
    
    print(f"  Seeded {count} entries about '{topic}' from LLM")
    return count


# ═══════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Three-Layer Pulse System")
    parser.add_argument("--db", default="knowledge.db", help="Database file path")
    parser.add_argument("--model", default="llama3.2", help="Ollama model name")
    parser.add_argument("--seed-json", default=None, help="Seed from JSON file")
    parser.add_argument("--topic", default=None, help="Seed from topic via LLM")
    parser.add_argument("--cycles", type=int, default=20, help="Number of pulse cycles")
    parser.add_argument("--verify-batch", type=int, default=10, help="Entries to verify per cycle")
    args = parser.parse_args()
    
    print(f"{'═'*60}")
    print(f"  THREE-LAYER PULSE SYSTEM")
    print(f"{'═'*60}")
    
    # Initialize components
    db = KnowledgeDB(args.db)
    llm = OllamaLLM(args.model)
    engine = PulseEngine(db)
    verifier = Verifier(db, llm)
    
    # Seed if needed
    if db.count() == 0:
        if args.seed_json:
            seed_from_json(db, args.seed_json)
        elif args.topic:
            seed_from_topic(db, llm, args.topic)
        else:
            print("  ⚠ Empty database. Use --seed-json or --topic to add initial knowledge.")
            print("  Example: python3 pulse_production.py --seed-json ai_kb_big.json")
            return
    
    db.stats()
    
    # Main loop
    print(f"\n{'═'*60}")
    print(f"  RUNNING {args.cycles} PULSE CYCLES")
    print(f"{'═'*60}")
    
    total_new = 0
    total_verified = 0
    
    for cycle in range(1, args.cycles + 1):
        t0 = time.time()
        
        # CPU: Pulse (fast symbolic logic)
        new = engine.pulse()
        total_new += new
        
        # GPU: Verify (LLM inference)
        ver = verifier.verify_batch(args.verify_batch)
        total_verified += ver
        
        elapsed = time.time() - t0
        
        # Status line
        t_count = db.count("T")
        m_count = db.count("M")
        total = db.count()
        print(f"  Cycle {cycle:3d} | +{new:3d} new | {ver:2d} verified | "
              f"T={t_count} M={m_count} total={total} | {elapsed:.1f}s")
        
        # If nothing new was produced and nothing left to verify, we've converged
        if new == 0 and m_count == 0:
            print(f"\n  ✓ CONVERGED at cycle {cycle}")
            break
        
        # Brief pause to prevent hammering (only matters with real LLM)
        if llm.available and ver > 0:
            time.sleep(0.5)
    
    # Final report
    print(f"\n{'═'*60}")
    print(f"  COMPLETE")
    print(f"{'═'*60}")
    print(f"  Total new entries: {total_new}")
    print(f"  Total verified: {total_verified}")
    db.stats()
    
    # Show confusion map
    confused = db.find(predicate="commonly_confused_with")
    if confused:
        print(f"\n  ═══ CONFUSION MAP ═══")
        for c in confused:
            print(f"    ⚠ {c['subject']} ≠ {c['object']}")
    
    # Show unresolved
    unresolved = db.all_entries("M")
    if unresolved:
        print(f"\n  ═══ UNRESOLVED ({len(unresolved)}) ═══")
        for u in unresolved[:10]:
            print(f"    ? {u['subject']} → {u['predicate']}({u['object']})")
        if len(unresolved) > 10:
            print(f"    ... +{len(unresolved)-10} more")

if __name__ == "__main__":
    main()
