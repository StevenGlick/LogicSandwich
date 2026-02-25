#!/usr/bin/env python3
"""
KNOWLEDGE QUERY ENGINE
══════════════════════

Ask questions of the knowledge base in plain English.
The system:
  1. Parses your question (LLM → search terms)
  2. Searches the database (SQLite, milliseconds)
  3. Walks the knowledge graph (follows connections)
  4. Synthesizes an answer (LLM + cited facts)

Every claim in the answer traces back to a specific entry.
No hallucination — only answers from verified knowledge.

USAGE:
  python3 query.py "What is a transformer?"
  python3 query.py "How does photosynthesis work?"
  python3 query.py "What's the difference between RNA and DNA?"
  python3 query.py --interactive                    # chat mode
  python3 query.py --raw transformer                # direct DB lookup
  python3 query.py --graph transformer 3            # show graph, depth 3
  python3 query.py --path transformer attention      # find path between concepts
  python3 query.py --unresolved                     # show what needs resolving
  python3 query.py --confusions                     # show confusion map
  python3 query.py --stats                          # database overview
"""

import sqlite3
import json
import sys
import re
import argparse
from collections import defaultdict, deque


# ═══════════════════════════════════════════════════════════
# SHARED MODULES
# ═══════════════════════════════════════════════════════════
from db import KnowledgeDB
from llm import create_llm
from bridges import BridgeSystem

# ═══════════════════════════════════════════════════════════
# QUERY ENGINE
# ═══════════════════════════════════════════════════════════

class QueryEngine:
    def __init__(self, db, llm):
        self.db = db
        self.llm = llm
        self.bridges = BridgeSystem(db.db_path)
    
    def query(self, question):
        """
        Full query pipeline:
        1. Parse question → search terms       (Grammar/Word layer)
        2. Search DB + walk graph               (Logic layer)
        3. Synthesize answer from facts          (Idea/Rhetoric layer)
        """
        print(f"\n  Question: {question}")
        print(f"  {'─'*50}")
        
        # ── STEP 1: Parse (Grammar layer) ──
        # Extract what to search for
        search_terms = []
        parsed = None
        
        if self.llm.available:
            print(f"  🔤 Parsing question...", end=" ", flush=True)
            parsed = self.llm.parse_question(question)
            if parsed:
                search_terms = (parsed.get("subjects", []) + 
                              parsed.get("keywords", []))
                print(f"terms: {search_terms}")
        
        if not search_terms:
            # Fallback: extract significant words
            stop_words = {"what","how","why","does","is","are","the","a","an",
                         "in","of","to","and","or","can","do","this","that",
                         "it","for","with","from","about","between","be","was",
                         "were","will","would","could","should","not","but"}
            search_terms = [w.lower().replace("?","").replace("'s","")
                          for w in question.split() 
                          if w.lower().replace("?","") not in stop_words and len(w) > 2]
            print(f"  🔤 Extracted terms: {search_terms}")
        
        # ── STEP 2: Search + Graph Walk (Logic layer) ──
        print(f"  🔍 Searching database...", end=" ", flush=True)
        
        # Direct search (FTS5 with synonym expansion, falls back to LIKE)
        direct_results = self.db.search_fts(search_terms)
        print(f"{len(direct_results)} direct hits")
        
        # Graph walk from top subjects
        graph_results = []
        subjects_found = set()
        for entry in direct_results[:10]:
            subj = entry["subject"]
            if "," not in subj and subj not in subjects_found:
                subjects_found.add(subj)
                neighbors = self.db.get_neighbors(subj, depth=1)
                graph_results.extend(neighbors)
        
        # Deduplicate
        seen_ids = set()
        all_facts = []
        for e in direct_results + graph_results:
            if e["id"] not in seen_ids:
                seen_ids.add(e["id"])
                all_facts.append(e)
        
        print(f"  📊 {len(all_facts)} total facts gathered ({len(direct_results)} direct + {len(graph_results)} via graph)")

        # Bridge lookup: find cross-domain connections
        related_bridges = []
        for term in search_terms:
            related_bridges.extend(self.bridges.find_bridges(term))
        # Deduplicate bridges by id
        seen_bridge_ids = set()
        unique_bridges = []
        for b in related_bridges:
            if b["id"] not in seen_bridge_ids:
                seen_bridge_ids.add(b["id"])
                unique_bridges.append(b)
        if unique_bridges:
            print(f"  🌉 {len(unique_bridges)} relevant bridges found")

        # Bridge expansion: pull facts about bridged terms
        bridge_facts = []
        if unique_bridges:
            bridge_terms = set()
            for b in unique_bridges[:5]:  # cap to avoid explosion
                bridge_terms.add(b["term_a"])
                bridge_terms.add(b["term_b"])
            # Only expand terms we haven't already searched
            new_terms = bridge_terms - set(search_terms) - subjects_found
            for bt in new_terms:
                extras = self.db.get_about(bt)
                for e in extras:
                    if e["id"] not in seen_ids:
                        seen_ids.add(e["id"])
                        bridge_facts.append(e)
            if bridge_facts:
                all_facts.extend(bridge_facts)
                print(f"  🌉 +{len(bridge_facts)} facts via bridge expansion")

        # Try to find paths between key subjects
        path = None
        if len(search_terms) >= 2:
            path = self.db.find_path(search_terms[0], search_terms[1])
            if path:
                print(f"  🔗 Path: {' '.join(path)}")
        
        # ── STEP 3: Synthesize (Idea layer) ──
        if not all_facts:
            print(f"  ❌ No relevant facts found in knowledge base.")
            print(f"     Try adding knowledge with ingest.py or curriculum_pulse.py")
            return None
        
        if self.llm.available:
            print(f"  💡 Synthesizing answer...")
            answer = self.llm.synthesize_answer(question, all_facts, path, unique_bridges)
            if answer:
                print(f"\n  {'═'*50}")
                print(f"  ANSWER:")
                print(f"  {'═'*50}")
                # Wrap text nicely
                for line in answer.split("\n"):
                    if line.strip():
                        # Word wrap at 70 chars
                        words = line.split()
                        current_line = "  "
                        for word in words:
                            if len(current_line) + len(word) > 72:
                                print(current_line)
                                current_line = "  " + word
                            else:
                                current_line += " " + word if current_line.strip() else "  " + word
                        if current_line.strip():
                            print(current_line)
                print(f"  {'═'*50}")
                print(f"  Based on {len(all_facts)} facts from knowledge base")
                return answer
        
        # No LLM — show raw facts
        print(f"\n  {'═'*50}")
        print(f"  RAW FACTS (no LLM available for synthesis):")
        print(f"  {'═'*50}")
        
        # Group by subject for readability
        by_subject = defaultdict(list)
        for e in all_facts:
            if "," not in e["subject"]:  # skip merged entries for display
                by_subject[e["subject"]].append(e)
        
        for subj in sorted(by_subject.keys()):
            entries = by_subject[subj]
            print(f"\n  {subj}:")
            for e in entries[:8]:
                tv = {"T":"✓","F":"✗","M":"?"}[e["truth_value"]]
                print(f"    [{tv}] → {e['predicate']}({e['object']})")
                if e.get("evidence_for"):
                    ev = json.loads(e["evidence_for"]) if isinstance(e["evidence_for"],str) else e["evidence_for"]
                    if ev:
                        print(f"        evidence: {ev[0][:60]}")
            if len(entries) > 8:
                print(f"    ... +{len(entries)-8} more")
        
        if path:
            print(f"\n  Connection path: {' '.join(path)}")

        # Show bridges if any
        if unique_bridges:
            print(f"\n  {'─'*50}")
            print(f"  BRIDGES (cross-domain connections):")
            for b in unique_bridges[:10]:
                print(f"    🌉 {b['term_a']} ↔ {b['term_b']} ({b['bridge_type']})")
                if b.get("reason"):
                    print(f"       {b['reason'][:80]}")

        print(f"\n  {len(all_facts)} facts total" +
              (f", {len(unique_bridges)} bridges" if unique_bridges else ""))
        return all_facts
    
    def raw_lookup(self, term):
        """Direct database lookup — show everything about a term."""
        print(f"\n  Raw lookup: {term}")
        print(f"  {'─'*50}")
        
        # As subject
        about = self.db.get_about(term)
        if about:
            print(f"\n  {term} →")
            for e in about:
                tv = {"T":"✓","F":"✗","M":"?"}[e["truth_value"]]
                print(f"    [{tv}][{e['id']:4d}] {e['predicate']:25s} → {e['object']}")
                src = e.get("source","")
                if src: print(f"    {'':31s} source: {src}")
        
        # As object (what points to this?)
        reverse = self.db.get_reverse(term)
        if reverse:
            print(f"\n  → {term}")
            for e in reverse:
                tv = {"T":"✓","F":"✗","M":"?"}[e["truth_value"]]
                print(f"    [{tv}][{e['id']:4d}] {e['subject']:25s} ← {e['predicate']}")
        
        if not about and not reverse:
            print(f"  Nothing found for '{term}'")
            # Fuzzy search
            fuzzy = self.db.search(term)
            if fuzzy:
                print(f"  But found {len(fuzzy)} partial matches:")
                subjects = sorted(set(e["subject"] for e in fuzzy if "," not in e["subject"]))
                for s in subjects[:15]:
                    print(f"    → {s}")

        # Bridges involving this term
        term_bridges = self.bridges.find_bridges(term)
        if term_bridges:
            print(f"\n  Bridges ({len(term_bridges)}):")
            for b in term_bridges:
                print(f"    🌉 {b['term_a']} ↔ {b['term_b']} ({b['bridge_type']})")
                if b.get("reason"):
                    print(f"       {b['reason'][:80]}")

    def show_graph(self, term, depth=2):
        """Show the knowledge graph around a term."""
        print(f"\n  Graph around '{term}' (depth {depth}):")
        print(f"  {'─'*50}")
        
        entries = self.db.get_neighbors(term, depth)
        if not entries:
            print(f"  Nothing found")
            return
        
        # Build adjacency display
        nodes = set()
        edges = []
        for e in entries:
            if "," in e["subject"] or "," in e["object"]:
                continue
            nodes.add(e["subject"])
            nodes.add(e["object"])
            edges.append((e["subject"], e["predicate"], e["object"]))
        
        print(f"  Nodes: {len(nodes)}")
        print(f"  Edges: {len(edges)}")
        
        # Show as tree from the root term
        root = term.lower().replace(" ", "_")
        shown = set()
        
        def show_node(node, indent=0, max_indent=depth*2):
            if indent > max_indent or node in shown:
                return
            shown.add(node)
            
            # Find edges FROM this node
            outgoing = [(p, o) for s, p, o in edges if s == node]
            # Find edges TO this node
            incoming = [(s, p) for s, p, o in edges if o == node]
            
            prefix = "  " + "  │ " * indent
            if indent == 0:
                print(f"\n  ◉ {node}")
            
            for pred, obj in sorted(outgoing):
                print(f"{prefix}├─ {pred} → {obj}")
                if obj not in shown:
                    show_node(obj, indent + 1)
            
            for subj, pred in sorted(incoming):
                if subj not in shown:
                    print(f"{prefix}├─ ← {pred} ← {subj}")
        
        show_node(root)
    
    def show_path(self, start, end):
        """Find and display path between two concepts."""
        print(f"\n  Finding path: {start} → ... → {end}")
        print(f"  {'─'*50}")
        
        path = self.db.find_path(start, end)
        if path:
            print(f"\n  ", end="")
            for i, step in enumerate(path):
                if step.startswith("--"):
                    print(f" {step} ", end="")
                else:
                    print(f"[{step}]", end="")
            print(f"\n\n  Path length: {len([s for s in path if not s.startswith('--')])} nodes")
        else:
            print(f"  No path found (try increasing max_depth)")
    
    def interactive(self):
        """Chat mode — keep asking questions."""
        print(f"\n  {'═'*50}")
        print(f"  KNOWLEDGE BASE QUERY — Interactive Mode")
        print(f"  {'═'*50}")
        print(f"  Database: {self.db.count()} entries")
        print(f"  LLM: {'✓ ' + self.llm.model if self.llm.available else '✗ raw mode'}")
        print(f"  Commands:")
        print(f"    /raw <term>       — direct lookup")
        print(f"    /graph <term>     — show graph")
        print(f"    /path <a> <b>     — find connection")
        print(f"    /bridges <term>   — cross-domain bridges")
        print(f"    /stats            — database stats")
        print(f"    /confusions       — show confusion map")
        print(f"    /unresolved       — show M entries")
        print(f"    /quit             — exit")
        print(f"  {'─'*50}")
        
        while True:
            try:
                q = input("\n  You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            
            if not q:
                continue
            if q.lower() in ("/quit", "/exit", "/q"):
                break
            
            if q.startswith("/raw "):
                self.raw_lookup(q[5:].strip())
            elif q.startswith("/graph "):
                parts = q[7:].strip().split()
                term = parts[0]
                depth = int(parts[1]) if len(parts) > 1 else 2
                self.show_graph(term, depth)
            elif q.startswith("/path "):
                parts = q[6:].strip().split()
                if len(parts) >= 2:
                    self.show_path(parts[0], parts[1])
                else:
                    print("  Usage: /path start end")
            elif q.startswith("/bridges "):
                term = q[9:].strip()
                found = self.bridges.find_bridges(term)
                if found:
                    print(f"\n  Bridges involving '{term}' ({len(found)}):")
                    for b in found:
                        print(f"    🌉 {b['term_a']} ↔ {b['term_b']} ({b['bridge_type']}, strength={b.get('strength',0):.1f})")
                        if b.get("reason"):
                            print(f"       {b['reason'][:100]}")
                else:
                    print(f"  No bridges found for '{term}'")
            elif q == "/stats":
                show_stats(self.db)
            elif q == "/confusions":
                show_confusions(self.db)
            elif q == "/unresolved":
                show_unresolved(self.db)
            else:
                self.query(q)
        
        print("\n  Goodbye!")


# ═══════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════

def show_stats(db):
    total = db.count()
    t = db.count("T"); m = db.count("M"); f = db.count("F")
    print(f"\n  Total: {total:,} entries")
    print(f"  T={t:,} | F={f:,} | M={m:,}")
    
    db.cur.execute("""
        SELECT source, COUNT(*) as cnt FROM entries 
        GROUP BY source ORDER BY cnt DESC LIMIT 15
    """)
    print(f"\n  By source:")
    for row in db.cur.fetchall():
        bar = "█" * min(row[1]//5, 40)
        print(f"    {row[0]:30s} {row[1]:7,d} {bar}")
    
    db.cur.execute("""
        SELECT grade_level, COUNT(*) as cnt FROM entries 
        WHERE grade_level > 0
        GROUP BY grade_level ORDER BY grade_level
    """)
    rows = db.cur.fetchall()
    if rows:
        print(f"\n  By grade:")
        for row in rows:
            print(f"    Grade {row[0]:2d}: {row[1]:,d} entries")

def show_confusions(db):
    confusions = db.get_confusions()
    if not confusions:
        print(f"\n  No confusions recorded yet")
        return
    print(f"\n  ⚠ CONFUSION MAP ({len(confusions)} entries)")
    for c in confusions:
        print(f"    {c['subject']:30s} ≠ {c['confused_with']}")
        if c.get('reason'):
            print(f"    {'':30s}   {c['reason'][:60]}")

def show_unresolved(db):
    unresolved = db.get_unresolved(30)
    if not unresolved:
        print(f"\n  All entries resolved!")
        return
    total_m = db.count("M")
    print(f"\n  ? UNRESOLVED ({total_m} total, showing 30)")
    for e in unresolved:
        print(f"    [{e['id']:4d}] {e['subject']} → {e['predicate']}({e['object']})")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Knowledge Base Query Engine")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("--db", default="curriculum.db")
    parser.add_argument("--model", default=None, help="LLM model name")
    parser.add_argument("--backend", choices=["ollama", "claude"], default="ollama")
    parser.add_argument("--api-key", default=None, help="Claude API key (or set ANTHROPIC_API_KEY)")
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--raw", help="Direct term lookup")
    parser.add_argument("--graph", nargs="+", help="Show graph: term [depth]")
    parser.add_argument("--path", nargs=2, help="Find path between two terms")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--confusions", action="store_true")
    parser.add_argument("--unresolved", action="store_true")
    args = parser.parse_args()
    
    db = KnowledgeDB(args.db)
    llm = create_llm(backend=args.backend, model=args.model, api_key=args.api_key)
    engine = QueryEngine(db, llm)
    
    if args.stats:
        show_stats(db); return
    if args.confusions:
        show_confusions(db); return
    if args.unresolved:
        show_unresolved(db); return
    if args.raw:
        engine.raw_lookup(args.raw); return
    if args.graph:
        term = args.graph[0]
        depth = int(args.graph[1]) if len(args.graph) > 1 else 2
        engine.show_graph(term, depth); return
    if args.path:
        engine.show_path(args.path[0], args.path[1]); return
    if args.interactive:
        engine.interactive(); return
    if args.question:
        engine.query(args.question); return
    
    # Default: interactive
    engine.interactive()

if __name__ == "__main__":
    main()
