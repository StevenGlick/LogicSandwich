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

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ═══════════════════════════════════════════════════════════
# DATABASE (read-only for queries)
# ═══════════════════════════════════════════════════════════

class KnowledgeDB:
    def __init__(self, db_path="curriculum.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.cur = self.conn.cursor()
    
    def search(self, term):
        """Find all entries where term appears in subject, predicate, or object."""
        term_norm = term.strip().lower().replace(" ", "_")
        self.cur.execute("""
            SELECT * FROM entries WHERE 
                subject LIKE ? OR object LIKE ? OR subject=? OR object=?
            ORDER BY truth_value, generation
        """, (f"%{term_norm}%", f"%{term_norm}%", term_norm, term_norm))
        return [dict(r) for r in self.cur.fetchall()]
    
    def search_exact(self, **kwargs):
        """Exact field match."""
        conds = []; params = []
        for k,v in kwargs.items():
            conds.append(f"{k}=?"); params.append(v)
        self.cur.execute(
            f"SELECT * FROM entries WHERE {' AND '.join(conds)}", params)
        return [dict(r) for r in self.cur.fetchall()]
    
    def search_multi(self, terms):
        """Find entries matching ANY of the given terms."""
        results = {}
        for term in terms:
            for entry in self.search(term):
                results[entry["id"]] = entry
        return sorted(results.values(), key=lambda e: e["id"])
    
    def get_neighbors(self, subject, depth=1):
        """
        Get all entries connected to a subject, up to N hops.
        This is the graph walk — the Logic layer query.
        """
        visited = set()
        frontier = {subject.lower().replace(" ", "_")}
        all_entries = []
        
        for d in range(depth):
            next_frontier = set()
            for node in frontier:
                if node in visited:
                    continue
                visited.add(node)
                
                # Find entries where this node is subject
                self.cur.execute(
                    "SELECT * FROM entries WHERE subject=? AND truth_value='T'",
                    (node,))
                for r in self.cur.fetchall():
                    entry = dict(r)
                    all_entries.append(entry)
                    obj = entry["object"]
                    if "," not in obj and obj not in visited:
                        next_frontier.add(obj)
                
                # Find entries where this node is object
                self.cur.execute(
                    "SELECT * FROM entries WHERE object=? AND truth_value='T'",
                    (node,))
                for r in self.cur.fetchall():
                    entry = dict(r)
                    all_entries.append(entry)
                    subj = entry["subject"]
                    if "," not in subj and subj not in visited:
                        next_frontier.add(subj)
            
            frontier = next_frontier
        
        # Deduplicate
        seen = set()
        unique = []
        for e in all_entries:
            key = (e["subject"], e["predicate"], e["object"])
            if key not in seen:
                seen.add(key)
                unique.append(e)
        
        return unique
    
    def find_path(self, start, end, max_depth=5):
        """
        Find the shortest path between two concepts in the graph.
        BFS through the knowledge graph.
        """
        start = start.lower().replace(" ", "_")
        end = end.lower().replace(" ", "_")
        
        # BFS
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            if len(path) > max_depth:
                continue
            
            # Get all connections from current node
            self.cur.execute(
                "SELECT subject, predicate, object FROM entries WHERE "
                "(subject=? OR object=?) AND truth_value='T'",
                (current, current))
            
            for row in self.cur.fetchall():
                s, p, o = row
                # Determine the neighbor
                if s == current and "," not in o:
                    neighbor = o
                elif o == current and "," not in s:
                    neighbor = s
                else:
                    continue
                
                if neighbor == end:
                    return path + [f"--{p}-->", neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [f"--{p}-->", neighbor]))
        
        return None  # no path found
    
    def get_about(self, subject):
        """Get everything known about a subject (direct facts only)."""
        subject = subject.lower().replace(" ", "_")
        self.cur.execute(
            "SELECT * FROM entries WHERE subject=? AND truth_value='T' AND object NOT LIKE '%,%'",
            (subject,))
        return [dict(r) for r in self.cur.fetchall()]
    
    def get_reverse(self, obj):
        """What points TO this object?"""
        obj = obj.lower().replace(" ", "_")
        self.cur.execute(
            "SELECT * FROM entries WHERE object=? AND truth_value='T' AND subject NOT LIKE '%,%'",
            (obj,))
        return [dict(r) for r in self.cur.fetchall()]
    
    def count(self, tv=None):
        if tv:
            self.cur.execute("SELECT COUNT(*) FROM entries WHERE truth_value=?", (tv,))
        else:
            self.cur.execute("SELECT COUNT(*) FROM entries")
        return self.cur.fetchone()[0]
    
    def get_confusions(self):
        try:
            self.cur.execute("SELECT * FROM confusion_map ORDER BY grade_discovered")
            return [dict(r) for r in self.cur.fetchall()]
        except:
            return []
    
    def get_unresolved(self, limit=30):
        self.cur.execute(
            "SELECT * FROM entries WHERE truth_value='M' LIMIT ?", (limit,))
        return [dict(r) for r in self.cur.fetchall()]


# ═══════════════════════════════════════════════════════════
# LLM
# ═══════════════════════════════════════════════════════════

class OllamaLLM:
    def __init__(self, model="llama3.2", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.available = self._check()
    
    def _check(self):
        if not HAS_REQUESTS: return False
        try:
            return requests.get(f"{self.base_url}/api/tags", timeout=3).status_code == 200
        except: return False
    
    def ask(self, prompt, system=None):
        if not self.available: return None
        messages = []
        if system: messages.append({"role":"system","content":system})
        messages.append({"role":"user","content":prompt})
        try:
            r = requests.post(f"{self.base_url}/api/chat",
                json={"model":self.model,"messages":messages,
                      "stream":False,"options":{"temperature":0.3}},
                timeout=120)
            if r.status_code == 200:
                return r.json()["message"]["content"]
        except Exception as e:
            print(f"  ⚠ LLM error: {e}")
        return None
    
    def parse_question(self, question):
        """Extract search terms from a natural language question."""
        system = """Extract the key concepts from this question for database lookup.
Output ONLY a JSON object with these fields:
{
  "subjects": ["term1", "term2"],
  "predicates": ["relationship1"],
  "keywords": ["other", "important", "words"]
}
Use lowercase_with_underscores. Be thorough — include synonyms."""
        
        response = self.ask(question, system)
        if not response: return None
        
        # Try to parse JSON from response
        try:
            # Strip markdown code fences if present
            cleaned = re.sub(r'```json?\s*', '', response)
            cleaned = re.sub(r'```', '', cleaned).strip()
            return json.loads(cleaned)
        except:
            # Fallback: extract words
            words = [w.lower().replace("?","").replace("'s","") 
                    for w in question.split() if len(w) > 3]
            return {"subjects": words, "predicates": [], "keywords": words}
    
    def synthesize_answer(self, question, facts, path=None):
        """Given facts from the KB, synthesize a natural language answer."""
        facts_text = "\n".join([
            f"  [{e['id']}] {e['subject']} → {e['predicate']}({e['object']}) "
            f"[{e['truth_value']}] (source: {e['source']})"
            for e in facts[:40]  # cap to avoid token overflow
        ])
        
        path_text = ""
        if path:
            path_text = f"\nConnection path: {' '.join(path)}\n"
        
        system = """You are answering questions using ONLY the provided knowledge base facts.

Rules:
- ONLY use information from the provided facts
- When you state something, cite the entry ID in brackets like [42]
- If the facts don't contain enough info, say so honestly
- If facts conflict, note the conflict
- Keep the answer clear and concise
- If a fact has truth_value M, mention it's unverified"""
        
        prompt = f"""Question: {question}
{path_text}
Known facts:
{facts_text}

Answer the question using only these facts. Cite entry IDs."""
        
        return self.ask(prompt, system)


# ═══════════════════════════════════════════════════════════
# QUERY ENGINE
# ═══════════════════════════════════════════════════════════

class QueryEngine:
    def __init__(self, db, llm):
        self.db = db
        self.llm = llm
    
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
        
        # Direct search
        direct_results = self.db.search_multi(search_terms)
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
            answer = self.llm.synthesize_answer(question, all_facts, path)
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
        
        print(f"\n  {len(all_facts)} facts total")
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
    parser.add_argument("--model", default="llama3.2")
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--raw", help="Direct term lookup")
    parser.add_argument("--graph", nargs="+", help="Show graph: term [depth]")
    parser.add_argument("--path", nargs=2, help="Find path between two terms")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--confusions", action="store_true")
    parser.add_argument("--unresolved", action="store_true")
    args = parser.parse_args()
    
    db = KnowledgeDB(args.db)
    llm = OllamaLLM(args.model)
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
