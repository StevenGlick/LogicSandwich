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
# SHARED MODULES
# ═══════════════════════════════════════════════════════════
from db import KnowledgeDB
from llm import create_llm
from predicate_rules import TransitiveFilter, PredicateRuleLearner, parse_transitive_evidence
from bridges import BridgeSystem


# ═══════════════════════════════════════════════════════════
# PULSE ENGINE (CPU)
# ═══════════════════════════════════════════════════════════
# This is the Logic + Idea layer.
# Pure symbolic operations — runs entirely on CPU.
# ═══════════════════════════════════════════════════════════

CAP = 50  # max new entries per operation per pulse

class PulseEngine:
    def __init__(self, db, bridges=None):
        self.db = db
        self.transitive_filter = TransitiveFilter()
        self.bridges = bridges
    
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
                # Predicate transparency: block invalid property inheritance
                allowed, reason = self.transitive_filter.check_chain(
                    e["predicate"], e2["predicate"])
                if not allowed:
                    continue

                self.db.add(
                    e["subject"], e2["predicate"], e2["object"], "M",
                    evidence_for=[f"transitive[{e['predicate']}→{e2['predicate']}]: {e['subject']}→{obj}→{e2['object']}"],
                    source="idea:transitive", generation=gen,
                    needs_verification=True)
                existing.add(sig)
                count += 1
                if count >= CAP:
                    return count
        return count
    
    def _cross_domain(self, gen, existing):
        """Find where different subjects share exact same predicate+object.
        If a bridge system is available, check for existing bridges and
        create candidate bridges for new cross-domain discoveries."""
        entries = self.db.all_entries("T")
        skip = {"is_a", "structurally_similar_to", "commonly_confused_with"}

        po_groups = defaultdict(list)
        all_subjects = set()
        for e in entries:
            if "," not in e["subject"]:
                all_subjects.add(e["subject"])
                # Only single subjects go into groups — prevents recursive compounding
                if "," not in e["object"] and e["predicate"] not in skip:
                    po_groups[(e["predicate"], e["object"])].append(e["subject"])

        # Specificity threshold: skip properties shared by >50% of all subjects
        total_subjects = max(len(all_subjects), 1)
        count = 0
        for (pred, obj), subjects in po_groups.items():
            unique_subjects = sorted(set(subjects))
            if len(unique_subjects) < 2:
                continue
            if len(unique_subjects) / total_subjects > 0.5:
                continue  # too generic to be meaningful
            for s1, s2 in combinations(unique_subjects, 2):
                combined = f"{s1},{s2}"
                sig = (combined, f"both_{pred}", obj)
                if sig in existing:
                    continue

                # Bridge awareness: check if this pair has an existing bridge
                evidence = ["cross-domain pattern"]
                if self.bridges:
                    b1 = self.bridges.find_bridges(s1)
                    has_bridge = any(
                        b["term_a"] == s2 or b["term_b"] == s2 for b in b1)
                    if has_bridge:
                        evidence = [f"cross-domain pattern (bridge-validated: {s1}↔{s2})"]
                    else:
                        # No bridge exists — register as candidate
                        self.bridges.add_bridge(
                            s1, s2, "emergent",
                            reason=f"both share {pred}({obj})",
                            strength=0.3,
                            discovered_by="pulse")

                self.db.add(combined, f"both_{pred}", obj, "T",
                    evidence_for=evidence,
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
        self.rule_learner = PredicateRuleLearner(db)

    def verify_batch(self, batch_size=10):
        """
        Take unverified entries, send them to the LLM for verification.
        This is where CPU work meets GPU work.
        """
        # Find entries that need verification
        unverified = self.db.find(needs_verification=1)
        # If LLM is available, also pick up M entries that simulation mode couldn't resolve
        if not unverified and self.llm.available:
            unverified = self.db.all_entries("M")
        if not unverified:
            return 0

        batch = unverified[:batch_size]
        verified = 0
        new_entries = 0
        total = len(batch)

        for i, entry in enumerate(batch, 1):
            claim = f"{entry['subject']}→{entry['predicate']}({entry['object']})"
            print(f"    verifying [{i}/{total}] {claim[:60]}...", end="", flush=True)
            evidence = json.loads(entry["evidence_for"])

            if self.llm.available:
                # REAL LLM VERIFICATION
                t0 = time.time()
                verdict, reasoning, new_facts = self.llm.verify_entry(
                    entry["subject"], entry["predicate"],
                    entry["object"], evidence)
                elapsed = time.time() - t0

                self.db.update_truth(entry["id"], verdict, f"LLM: {reasoning[:100]}")
                print(f" {verdict} ({elapsed:.1f}s)")

                # Record outcome for predicate rule learning
                if entry.get("source") == "idea:transitive":
                    self._record_predicate_outcome(evidence, verdict)

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
                safe_preds = {"is_a", "uses", "has_component", "enables",
                              "requires", "outputs", "processes"}
                if entry["predicate"] in safe_preds:
                    self.db.update_truth(entry["id"], "T", "auto: safe predicate chain")
                    print(" T (auto)")
                else:
                    self.db.cursor.execute(
                        "UPDATE entries SET needs_verification=0 WHERE id=?",
                        (entry["id"],))
                    self.db.conn.commit()
                    print(" M (skipped)")

            verified += 1

        return verified

    def _record_predicate_outcome(self, evidence, verdict):
        """Extract predicate pair from transitive evidence and record to learner."""
        for ev in evidence:
            if not isinstance(ev, str):
                continue
            pair = parse_transitive_evidence(ev)
            if pair:
                link_pred, inherited_pred = pair
                approved = (verdict == "T")
                self.rule_learner.record_outcome(link_pred, inherited_pred, approved)
                break  # One recording per entry


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
    parser.add_argument("--db", default="curriculum.db", help="Database file path")
    parser.add_argument("--model", default=None, help="LLM model name")
    parser.add_argument("--backend", choices=["ollama", "claude"], default="ollama")
    parser.add_argument("--api-key", default=None, help="Claude API key (or set ANTHROPIC_API_KEY)")
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
    llm = create_llm(backend=args.backend, model=args.model, api_key=args.api_key)
    bridges = BridgeSystem(args.db)
    engine = PulseEngine(db, bridges)
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

    # Apply and report predicate rule learning
    suggestions = verifier.rule_learner.suggest_rules()
    applied = verifier.rule_learner.apply_rules(engine.transitive_filter, suggestions)
    if applied or suggestions["block"] or suggestions["allow"]:
        verifier.rule_learner.report()
        if applied:
            print(f"  → Applied {applied} learned rule(s) to TransitiveFilter")

if __name__ == "__main__":
    main()
