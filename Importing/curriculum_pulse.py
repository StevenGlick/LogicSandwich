#!/usr/bin/env python3
"""
THREE-LAYER PULSE SYSTEM — Curriculum Edition
══════════════════════════════════════════════

A knowledge system that learns like a student:
  - Starts with kindergarten concepts
  - Progresses through grade levels
  - Each level builds on what came before
  - Auto-generates questions to test understanding
  - Pulse engine finds patterns across everything learned

SETUP:
  pip install requests flask          (web UI + Ollama communication)
  Install Ollama: https://ollama.com
  Pull a model:   ollama pull llama3.2   (or whatever fits your GPU)

USAGE:
  python3 curriculum_pulse.py                          # run headless
  python3 curriculum_pulse.py --dashboard              # run with web UI
  python3 curriculum_pulse.py --model mistral           # different model
  python3 curriculum_pulse.py --start-grade 6           # skip to grade 6
  python3 curriculum_pulse.py --pulses-per-grade 10     # more processing per grade

DASHBOARD:
  When running with --dashboard, open http://localhost:5000 in your browser
  to watch the knowledge base grow in real time.

FILES:
  curriculum.db    — SQLite knowledge database
  pulse_log.txt    — Human-readable log
"""

import sqlite3
import json
import time
import argparse
import sys
import threading
import os
from collections import defaultdict
from itertools import combinations
from datetime import datetime

# ═══════════════════════════════════════════════════════════
# OPTIONAL IMPORTS
# ═══════════════════════════════════════════════════════════
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from flask import Flask, render_template_string, jsonify
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False


# ═══════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════

class KnowledgeDB:
    def __init__(self, db_path="curriculum.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        self._create_tables()
    
    def _create_tables(self):
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
                grade_level INTEGER DEFAULT 0,
                verified INTEGER DEFAULT 0,
                needs_verification INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT,
                grade_level INTEGER,
                message TEXT,
                details TEXT DEFAULT ''
            );
            
            CREATE TABLE IF NOT EXISTS confusion_map (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT,
                confused_with TEXT,
                reason TEXT,
                grade_discovered INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_subject ON entries(subject);
            CREATE INDEX IF NOT EXISTS idx_predicate ON entries(predicate);
            CREATE INDEX IF NOT EXISTS idx_truth ON entries(truth_value);
            CREATE INDEX IF NOT EXISTS idx_sig ON entries(subject, predicate, object);
            CREATE INDEX IF NOT EXISTS idx_grade ON entries(grade_level);
        """)
        self.conn.commit()
    
    def add(self, subject, predicate, obj, truth_value="M",
            evidence_for=None, evidence_against=None,
            source="unknown", generation=0, grade_level=0,
            needs_verification=False):
        self.cursor.execute(
            "SELECT id FROM entries WHERE subject=? AND predicate=? AND object=?",
            (subject, predicate, obj))
        if self.cursor.fetchone():
            return None
        self.cursor.execute("""
            INSERT INTO entries (subject, predicate, object, truth_value,
                evidence_for, evidence_against, source, generation,
                grade_level, needs_verification)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (subject, predicate, obj, truth_value,
              json.dumps(evidence_for or []), json.dumps(evidence_against or []),
              source, generation, grade_level, int(needs_verification)))
        self.conn.commit()
        return self.cursor.lastrowid
    
    def find(self, **kwargs):
        conditions = []; params = []
        for k, v in kwargs.items():
            conditions.append(f"{k}=?"); params.append(v)
        where = " AND ".join(conditions) if conditions else "1=1"
        self.cursor.execute(f"SELECT * FROM entries WHERE {where}", params)
        return [dict(r) for r in self.cursor.fetchall()]
    
    def all_entries(self, truth_value=None):
        if truth_value:
            self.cursor.execute("SELECT * FROM entries WHERE truth_value=?", (truth_value,))
        else:
            self.cursor.execute("SELECT * FROM entries")
        return [dict(r) for r in self.cursor.fetchall()]
    
    def update_truth(self, eid, new_truth, evidence=None):
        self.cursor.execute(
            "UPDATE entries SET truth_value=?, verified=1, needs_verification=0 WHERE id=?",
            (new_truth, eid))
        if evidence:
            row = dict(self.cursor.execute("SELECT * FROM entries WHERE id=?", (eid,)).fetchone())
            ev = json.loads(row["evidence_for"])
            ev.append(evidence)
            self.cursor.execute("UPDATE entries SET evidence_for=? WHERE id=?",
                (json.dumps(ev), eid))
        self.conn.commit()
    
    def count(self, truth_value=None):
        if truth_value:
            self.cursor.execute("SELECT COUNT(*) FROM entries WHERE truth_value=?", (truth_value,))
        else:
            self.cursor.execute("SELECT COUNT(*) FROM entries")
        return self.cursor.fetchone()[0]
    
    def current_generation(self):
        self.cursor.execute("SELECT MAX(generation) FROM entries")
        r = self.cursor.fetchone()[0]
        return (r or 0) + 1
    
    def signatures(self):
        self.cursor.execute("SELECT subject, predicate, object FROM entries")
        return {(r[0],r[1],r[2]) for r in self.cursor.fetchall()}
    
    def log_event(self, event_type, grade, message, details=""):
        self.cursor.execute(
            "INSERT INTO log (event_type, grade_level, message, details) VALUES (?,?,?,?)",
            (event_type, grade, message, details))
        self.conn.commit()
    
    def add_confusion(self, subject, confused_with, reason, grade):
        self.cursor.execute(
            "SELECT id FROM confusion_map WHERE subject=? AND confused_with=?",
            (subject, confused_with))
        if not self.cursor.fetchone():
            self.cursor.execute(
                "INSERT INTO confusion_map (subject, confused_with, reason, grade_discovered) VALUES (?,?,?,?)",
                (subject, confused_with, reason, grade))
            self.conn.commit()
            return True
        return False
    
    def get_log(self, limit=50):
        self.cursor.execute("SELECT * FROM log ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) for r in self.cursor.fetchall()]
    
    def get_confusions(self):
        self.cursor.execute("SELECT * FROM confusion_map ORDER BY grade_discovered")
        return [dict(r) for r in self.cursor.fetchall()]
    
    def stats_by_grade(self):
        self.cursor.execute("""
            SELECT grade_level, truth_value, COUNT(*) 
            FROM entries GROUP BY grade_level, truth_value
            ORDER BY grade_level
        """)
        result = defaultdict(lambda: {"T":0,"F":0,"M":0})
        for row in self.cursor.fetchall():
            result[row[0]][row[1]] = row[2]
        return dict(result)
    
    def stats_by_source(self):
        self.cursor.execute("""
            SELECT source, COUNT(*) FROM entries 
            GROUP BY source ORDER BY COUNT(*) DESC
        """)
        return [(r[0],r[1]) for r in self.cursor.fetchall()]


# ═══════════════════════════════════════════════════════════
# LLM (Ollama)
# ═══════════════════════════════════════════════════════════

class OllamaLLM:
    def __init__(self, model="llama3.2", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.available = self._check()
        self.call_count = 0
    
    def _check(self):
        if not HAS_REQUESTS: return False
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except: return False
    
    def ask(self, prompt, system=None, temperature=0.3):
        if not self.available: return None
        messages = []
        if system: messages.append({"role":"system","content":system})
        messages.append({"role":"user","content":prompt})
        try:
            r = requests.post(f"{self.base_url}/api/chat",
                json={"model":self.model,"messages":messages,
                      "stream":False,"options":{"temperature":temperature}},
                timeout=180)
            if r.status_code == 200:
                self.call_count += 1
                return r.json()["message"]["content"]
        except Exception as e:
            print(f"    ⚠ LLM error: {e}")
        return None
    
    def generate_facts(self, topic, grade_level, existing_subjects=None):
        """Ask LLM to generate atomic facts about a topic at a grade level."""
        grade_desc = CURRICULUM.get(grade_level, {}).get("name", f"Grade {grade_level}")
        
        context = ""
        if existing_subjects:
            sample = sorted(existing_subjects)[:30]
            context = f"\nThe student already knows about: {', '.join(sample)}"
        
        system = f"""You are building a knowledge base for a student at the {grade_desc} level.
Generate 15-25 atomic facts about the topic. Each fact is ONE relationship.

Output format — one fact per line, no numbering, no bullets:
subject|predicate|object

Rules:
- lowercase_with_underscores for all terms  
- Simple vocabulary appropriate for {grade_desc}
- Build on concepts the student already knows
- Predicates: is_a, has_part, made_of, needs, produces, bigger_than, 
  contains, causes, used_for, lives_in, eats, can, cannot, etc.
- Each fact must be atomic: ONE subject, ONE predicate, ONE object
- No explanations, just the facts
{context}"""

        prompt = f"Generate atomic knowledge facts about: {topic}"
        response = self.ask(prompt, system, temperature=0.5)
        if not response: return []
        
        facts = []
        for line in response.strip().split("\n"):
            line = line.strip().strip("-•*0123456789. ")
            if "|" not in line: continue
            parts = [p.strip().lower().replace(" ","_") for p in line.split("|")]
            if len(parts) >= 3 and parts[0] and parts[1] and parts[2]:
                facts.append({"subject":parts[0],"predicate":parts[1],"object":parts[2]})
        return facts
    
    def ask_question(self, question, known_facts=None):
        """Ask the LLM a question, get back atomic facts as answer."""
        context = ""
        if known_facts:
            fact_lines = [f"  {f['subject']} → {f['predicate']}({f['object']})" 
                         for f in known_facts[:20]]
            context = f"\nRelevant known facts:\n" + "\n".join(fact_lines)
        
        system = """You are a knowledge decomposition system. 
Given a question, break your answer into atomic facts.

Output format:
REASONING: 1-3 sentences explaining your thinking
FACTS:
subject|predicate|object|T_or_M
subject|predicate|object|T_or_M

Also flag common confusions:
CONFUSED: thing_a|thing_b|reason_why_confused

Only output this format, nothing else."""

        prompt = f"Question: {question}{context}"
        response = self.ask(prompt, system)
        if not response: return [], [], ""
        
        facts = []; confusions = []; reasoning = ""
        section = None
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("REASONING:"):
                reasoning = line.split(":",1)[1].strip()
                section = "reasoning"
                continue
            elif line.startswith("FACTS:"):
                section = "facts"; continue
            elif line.startswith("CONFUSED:"):
                section = "confused"
                line = line.split(":",1)[1].strip()
            
            if section == "reasoning" and not line.startswith(("FACTS","CONFUSED")):
                reasoning += " " + line
            
            if "|" in line:
                parts = [p.strip().lower().replace(" ","_") for p in line.split("|")]
                if section == "confused" and len(parts) >= 3:
                    confusions.append({"a":parts[0],"b":parts[1],"reason":parts[2] if len(parts)>2 else ""})
                elif len(parts) >= 3:
                    tv = parts[3].upper() if len(parts)>3 and parts[3].upper() in ("T","F","M") else "T"
                    facts.append({"subject":parts[0],"predicate":parts[1],"object":parts[2],"truth_value":tv})
        
        return facts, confusions, reasoning
    
    def verify_entry(self, subject, predicate, obj, evidence):
        """Ask LLM to verify a single claim."""
        system = """Evaluate this claim as True (T), False (F), or Maybe (M).

Output format:
VERDICT: T/F/M
REASON: one sentence
NEW_FACTS: (optional, one per line)
subject|predicate|object|T_or_M"""
        
        prompt = f"Claim: {subject} → {predicate}({obj})\nEvidence: {evidence}"
        response = self.ask(prompt, system)
        if not response: return "M", "unavailable", []
        
        verdict = "M"; reason = ""; new_facts = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("VERDICT:"):
                v = line.split(":")[1].strip().upper()[:1]
                if v in "TFM": verdict = v
            elif line.startswith("REASON:"):
                reason = line.split(":",1)[1].strip()
            elif "|" in line and line.count("|") >= 2:
                parts = [p.strip().lower().replace(" ","_") for p in line.split("|")]
                if len(parts) >= 3:
                    tv = parts[3].upper() if len(parts)>3 and parts[3].upper() in "TFM" else "T"
                    new_facts.append({"subject":parts[0],"predicate":parts[1],"object":parts[2],"truth_value":tv})
        
        return verdict, reason, new_facts


# ═══════════════════════════════════════════════════════════
# CURRICULUM
# ═══════════════════════════════════════════════════════════
# Structured like actual school progression.
# Each grade introduces topics that build on previous ones.
# ═══════════════════════════════════════════════════════════

CURRICULUM = {
    0: {
        "name": "Kindergarten",
        "topics": [
            "basic colors and shapes",
            "counting numbers 1 to 10",
            "animals and their babies",
            "parts of the human body",
            "the five senses",
            "seasons and weather",
        ],
        "questions": [
            "What shape has three sides?",
            "What do your eyes do?",
            "What season comes after winter?",
            "Is a puppy a baby dog or a baby cat?",
        ]
    },
    1: {
        "name": "1st Grade",
        "topics": [
            "states of matter: solid liquid gas",
            "living things versus non-living things",
            "basic plant parts: roots stems leaves flowers",
            "the solar system: sun moon earth",
            "simple addition and subtraction",
            "basic maps and directions",
        ],
        "questions": [
            "Is ice a solid or a liquid?",
            "Are rocks alive?",
            "What do plants need to grow?",
            "Does the Earth go around the Sun or the Sun around the Earth?",
        ]
    },
    2: {
        "name": "2nd Grade",
        "topics": [
            "the water cycle: evaporation condensation precipitation",
            "habitats: desert forest ocean arctic",
            "food chains: producer consumer decomposer",
            "basic measurement: length weight volume",
            "telling time and calendars",
            "community helpers and jobs",
        ],
        "questions": [
            "Where does rain come from?",
            "What eats what in a food chain?",
            "Can a desert have cold weather?",
            "Is a minute longer than a second?",
        ]
    },
    3: {
        "name": "3rd Grade",
        "topics": [
            "multiplication and division basics",
            "earth layers: crust mantle core",
            "types of rocks: igneous sedimentary metamorphic",
            "the human skeleton and muscles",
            "simple machines: lever pulley inclined plane",
            "ecosystems and biodiversity",
        ],
        "questions": [
            "How are metamorphic rocks formed?",
            "What is a lever and how does it help?",
            "Are bones alive or dead?",
            "Is multiplication just repeated addition?",
        ]
    },
    4: {
        "name": "4th Grade",
        "topics": [
            "electricity and circuits",
            "magnetism and compasses",
            "fractions and decimals",
            "the digestive system",
            "the water table and groundwater",
            "american geography: states regions",
            "types of energy: kinetic potential thermal",
        ],
        "questions": [
            "Does electricity flow like water?",
            "What makes a compass point north?",
            "Is half the same as point five?",
            "What happens to food after you swallow it?",
        ]
    },
    5: {
        "name": "5th Grade",
        "topics": [
            "cells: the basic unit of life",
            "photosynthesis: how plants make food",
            "the periodic table basics: elements atoms",
            "gravity and force",
            "the american revolution and founding documents",
            "volume area and perimeter formulas",
        ],
        "questions": [
            "Do plants eat sunlight?",
            "What is the smallest thing that is alive?",
            "Is gravity a force or a thing?",
            "What is an atom made of?",
        ]
    },
    6: {
        "name": "6th Grade",
        "topics": [
            "plate tectonics: earthquakes and volcanoes",
            "the atmosphere: layers and composition",
            "ratios proportions and percentages",
            "ancient civilizations: mesopotamia egypt greece rome",
            "introduction to algebra: variables and equations",
            "the scientific method",
        ],
        "questions": [
            "Why do earthquakes happen?",
            "What is the atmosphere made of?",
            "Is algebra just arithmetic with letters?",
            "What makes something a civilization?",
        ]
    },
    7: {
        "name": "7th Grade",
        "topics": [
            "cell division: mitosis and meiosis",
            "genetics: dna genes chromosomes heredity",
            "chemical reactions: reactants products catalysts",
            "the human circulatory system: heart blood vessels",
            "probability and statistics basics",
            "world geography: continents climates cultures",
        ],
        "questions": [
            "Is DNA the same thing as a gene?",
            "What is the difference between mitosis and meiosis?",
            "Can a chemical reaction be reversed?",
            "Is probability just guessing?",
        ]
    },
    8: {
        "name": "8th Grade",
        "topics": [
            "newton's three laws of motion",
            "waves: light sound electromagnetic spectrum",
            "the periodic table: groups periods properties",
            "american civil war causes and effects",
            "linear equations and graphing",
            "evolution and natural selection",
        ],
        "questions": [
            "Is light a wave or a particle?",
            "Does evolution mean we came from monkeys?",
            "What does F equals ma actually mean?",
            "Can sound travel through space?",
        ]
    },
    9: {
        "name": "9th Grade (Freshman)",
        "topics": [
            "biology: cell organelles and their functions",
            "taxonomy: kingdoms phyla classes of life",
            "basic chemistry: bonds molecules compounds",
            "world history: renaissance reformation enlightenment",
            "geometry: proofs theorems congruence",
            "introduction to literature analysis",
        ],
        "questions": [
            "What is the difference between an element and a compound?",
            "Why do we classify living things into groups?",
            "What came first, the Renaissance or the Enlightenment?",
            "Is a proof the same as evidence?",
        ]
    },
    10: {
        "name": "10th Grade (Sophomore)",
        "topics": [
            "chemistry: atomic structure electron orbitals",
            "chemical bonding: ionic covalent metallic",
            "stoichiometry: balancing equations moles",
            "world history: industrial revolution colonialism",
            "advanced algebra: quadratics polynomials",
            "ecology: biomes carbon cycle nitrogen cycle",
        ],
        "questions": [
            "Is a covalent bond stronger than an ionic bond?",
            "What is a mole in chemistry versus in biology?",
            "Did the industrial revolution help or hurt people?",
            "Are all quadratic equations solvable?",
        ]
    },
    11: {
        "name": "11th Grade (Junior)",
        "topics": [
            "physics: thermodynamics and entropy",
            "electromagnetism: fields waves radiation",
            "american history: constitution amendments civil rights",
            "trigonometry: sine cosine tangent unit circle",
            "introduction to calculus: limits derivatives",
            "organic chemistry basics: carbon compounds",
        ],
        "questions": [
            "What is entropy and why does it always increase?",
            "Is calculus just advanced algebra?",
            "What is the relationship between electricity and magnetism?",
            "Why is carbon special in chemistry?",
        ]
    },
    12: {
        "name": "12th Grade (Senior)",
        "topics": [
            "physics: quantum mechanics basics wave-particle duality",
            "physics: special relativity time dilation",
            "calculus: integrals and applications",
            "statistics: distributions hypothesis testing",
            "philosophy: epistemology ethics logic",
            "economics: supply demand markets",
            "computer science: algorithms data structures basics",
        ],
        "questions": [
            "Can something be a wave and a particle at the same time?",
            "Does time actually slow down at high speeds?",
            "Is an integral the opposite of a derivative?",
            "What is the difference between knowledge and belief?",
            "Can a computer think?",
        ]
    },
    13: {
        "name": "College Freshman",
        "topics": [
            "discrete mathematics: sets logic proofs",
            "calculus of several variables",
            "general biology: molecular biology biochemistry",
            "general physics: mechanics waves thermodynamics",
            "introduction to programming: variables loops functions",
            "critical thinking and argumentation",
        ],
        "questions": [
            "Is math discovered or invented?",
            "What is the relationship between biology and chemistry?",
            "How is programming different from mathematics?",
            "Can logic prove everything that is true?",
        ]
    },
}


# ═══════════════════════════════════════════════════════════
# PULSE ENGINE (CPU)
# ═══════════════════════════════════════════════════════════

CAP = 50

class PulseEngine:
    def __init__(self, db):
        self.db = db
    
    def pulse(self):
        gen = self.db.current_generation()
        ex = self.db.signatures()
        total = 0
        for name, fn in [("merge",self._merge),("transitive",self._transitive),
                          ("cross_domain",self._cross_domain),("analogy",self._analogy)]:
            total += fn(gen, ex)
        return total
    
    def _merge(self, gen, ex):
        entries = self.db.all_entries("T")
        groups = defaultdict(list)
        for e in entries:
            if "," not in e["object"]:
                groups[(e["subject"],e["predicate"])].append(e["object"])
        count = 0
        for (s,p), objs in groups.items():
            uniq = sorted(set(objs))
            if len(uniq) < 2: continue
            merged = ",".join(uniq)
            if (s,p,merged) in ex: continue
            self.db.add(s,p,merged,"T",
                evidence_for=[f"merged {len(uniq)}"],source="idea:merge",generation=gen)
            ex.add((s,p,merged)); count += 1
            if count >= CAP: break
        return count
    
    def _transitive(self, gen, ex):
        entries = self.db.all_entries("T")
        skip = {"structurally_similar_to","commonly_confused_with","analogous_to","inspired_by"}
        by_subj = defaultdict(list)
        for e in entries:
            if "," not in e["object"] and e["predicate"] not in skip:
                by_subj[e["subject"]].append(e)
        count = 0
        for e in entries:
            if "," in e["object"] or e["predicate"] in skip: continue
            obj = e["object"]
            if obj not in by_subj: continue
            for e2 in by_subj[obj]:
                if e2["predicate"] in skip or "," in e2["object"]: continue
                sig = (e["subject"],e2["predicate"],e2["object"])
                if sig in ex or e["subject"]==e2["object"]: continue
                self.db.add(e["subject"],e2["predicate"],e2["object"],"M",
                    evidence_for=[f"transitive:{e['subject']}→{obj}→{e2['object']}"],
                    source="idea:transitive",generation=gen,needs_verification=True)
                ex.add(sig); count += 1
                if count >= CAP: return count
        return count
    
    def _cross_domain(self, gen, ex):
        entries = self.db.all_entries("T")
        skip = {"is_a","structurally_similar_to","commonly_confused_with"}
        po = defaultdict(list)
        for e in entries:
            if "," not in e["object"] and e["predicate"] not in skip:
                po[(e["predicate"],e["object"])].append(e["subject"])
        count = 0
        for (p,o), subs in po.items():
            uniq = sorted(set(subs))
            if len(uniq) < 2: continue
            for s1,s2 in combinations(uniq,2):
                sig = (f"{s1},{s2}",f"both_{p}",o)
                if sig in ex: continue
                self.db.add(f"{s1},{s2}",f"both_{p}",o,"T",
                    evidence_for=["cross-domain"],source="idea:cross_domain",generation=gen)
                ex.add(sig); count += 1
                if count >= CAP: return count
        return count
    
    def _analogy(self, gen, ex):
        entries = self.db.all_entries("T")
        skip = {"structurally_similar_to","commonly_confused_with","analogous_to"}
        profiles = defaultdict(set)
        for e in entries:
            if not e["subject"].startswith("category(") and e["predicate"] not in skip and "," not in e["subject"]:
                profiles[e["subject"]].add(e["predicate"])
        count = 0
        subjects = sorted(profiles.keys())
        for s1,s2 in combinations(subjects,2):
            shared = profiles[s1] & profiles[s2] - skip
            total = (profiles[s1] | profiles[s2]) - skip
            if not total: continue
            if len(shared)/len(total) >= 0.6:
                sig = (s1,"structurally_similar_to",s2)
                if sig in ex: continue
                self.db.add(s1,"structurally_similar_to",s2,"T",
                    evidence_for=[f"sim={len(shared)/len(total):.0%}"],
                    source="idea:analogy",generation=gen)
                ex.add(sig); count += 1
                if count >= CAP: return count
        return count


# ═══════════════════════════════════════════════════════════
# VERIFIER
# ═══════════════════════════════════════════════════════════

class Verifier:
    def __init__(self, db, llm):
        self.db = db
        self.llm = llm
    
    def verify_batch(self, batch_size=5):
        unverified = self.db.find(needs_verification=1)
        if not unverified: return 0
        batch = unverified[:batch_size]
        count = 0
        for entry in batch:
            ev = json.loads(entry["evidence_for"])
            if self.llm.available:
                verdict, reason, new_facts = self.llm.verify_entry(
                    entry["subject"],entry["predicate"],entry["object"],ev)
                self.db.update_truth(entry["id"], verdict, f"LLM: {reason[:100]}")
                gen = self.db.current_generation()
                for nf in new_facts:
                    self.db.add(nf["subject"],nf["predicate"],nf["object"],
                        nf.get("truth_value","M"),source="llm:verification",generation=gen)
                # Check if LLM flagged a confusion
                if verdict == "F":
                    self.db.add_confusion(
                        entry["subject"], entry["object"],
                        f"False transitive: {reason[:100]}",
                        entry.get("grade_level",0))
            else:
                safe = {"is_a","uses","has_part","has_component","needs",
                        "made_of","contains","lives_in","eats","produces",
                        "enables","requires","causes"}
                if entry["predicate"] in safe:
                    self.db.update_truth(entry["id"],"T","auto: safe chain")
                else:
                    self.db.cursor.execute(
                        "UPDATE entries SET needs_verification=0 WHERE id=?",(entry["id"],))
                    self.db.conn.commit()
            count += 1
        return count


# ═══════════════════════════════════════════════════════════
# CURRICULUM ENGINE (orchestrates learning)
# ═══════════════════════════════════════════════════════════

class CurriculumEngine:
    def __init__(self, db, llm, pulse_engine, verifier):
        self.db = db
        self.llm = llm
        self.engine = pulse_engine
        self.verifier = verifier
        self.current_grade = 0
        self.running = True
    
    def get_known_subjects(self):
        entries = self.db.all_entries("T")
        return {e["subject"] for e in entries if "," not in e["subject"]}
    
    def teach_grade(self, grade, pulses_per_grade=5):
        """Teach one grade level: introduce topics, ask questions, pulse."""
        info = CURRICULUM.get(grade)
        if not info:
            return False
        
        name = info["name"]
        print(f"\n{'═'*60}")
        print(f"  📚 {name}")
        print(f"{'═'*60}")
        self.db.log_event("grade_start", grade, f"Starting {name}")
        
        known = self.get_known_subjects()
        
        # ── TEACH TOPICS ──
        for topic in info["topics"]:
            if not self.running: return False
            print(f"\n  📖 Teaching: {topic}")
            
            if self.llm.available:
                facts = self.llm.generate_facts(topic, grade, known)
                added = 0
                for f in facts:
                    result = self.db.add(
                        f["subject"], f["predicate"], f["object"], "T",
                        evidence_for=[f"curriculum: {name}, topic: {topic}"],
                        source=f"curriculum:grade_{grade}",
                        generation=self.db.current_generation(),
                        grade_level=grade)
                    if result: added += 1
                print(f"    +{added} facts from LLM")
                known = self.get_known_subjects()
            else:
                print(f"    (skipped — no LLM)")
            
            self.db.log_event("topic_taught", grade, topic,
                f"entries: {self.db.count()}")
        
        # ── PULSE (process what we just learned) ──
        print(f"\n  ⚡ Pulsing ({pulses_per_grade} cycles)...")
        for i in range(pulses_per_grade):
            if not self.running: return False
            new = self.engine.pulse()
            ver = self.verifier.verify_batch(10)
            total = self.db.count()
            t = self.db.count("T"); m = self.db.count("M")
            print(f"    Pulse {i+1}: +{new} new, {ver} verified | T={t} M={m} total={total}")
            if new == 0: break
        
        # ── ASK QUESTIONS ──
        if info.get("questions"):
            print(f"\n  ❓ Asking questions...")
            for q in info["questions"]:
                if not self.running: return False
                print(f"    Q: \"{q}\"")
                
                if self.llm.available:
                    # Find relevant known facts for context
                    keywords = [w.lower().replace("?","") for w in q.split() 
                               if len(w) > 3]
                    relevant = []
                    for kw in keywords:
                        relevant.extend(self.db.find(subject=kw))
                        relevant.extend(self.db.find(object=kw))
                    relevant = relevant[:20]
                    
                    facts, confusions, reasoning = self.llm.ask_question(q, relevant)
                    
                    if reasoning:
                        print(f"      → {reasoning[:80]}")
                    
                    for f in facts:
                        result = self.db.add(
                            f["subject"],f["predicate"],f["object"],
                            f.get("truth_value","T"),
                            evidence_for=[f"Q&A: {q}"],
                            source="llm:question",
                            generation=self.db.current_generation(),
                            grade_level=grade)
                        if result:
                            print(f"      + {f['subject']}→{f['predicate']}({f['object']})")
                    
                    for c in confusions:
                        added = self.db.add_confusion(c["a"],c["b"],c.get("reason",""),grade)
                        if added:
                            print(f"      ⚠ CONFUSION: {c['a']} ≠ {c['b']}")
                else:
                    print(f"      (skipped — no LLM)")
                
                self.db.log_event("question", grade, q)
        
        # ── POST-QUESTION PULSES ──
        print(f"\n  ⚡ Post-question pulse...")
        for i in range(3):
            if not self.running: return False
            new = self.engine.pulse()
            ver = self.verifier.verify_batch(5)
            if new == 0: break
        
        # ── GRADE SUMMARY ──
        total = self.db.count()
        t = self.db.count("T"); m = self.db.count("M"); f = self.db.count("F")
        confusions = self.db.get_confusions()
        
        print(f"\n  ── {name} COMPLETE ──")
        print(f"  KB: {total} entries | T={t} F={f} M={m}")
        print(f"  Confusions found: {len(confusions)}")
        
        self.db.log_event("grade_complete", grade, 
            f"{name} complete: {total} entries, T={t} F={f} M={m}")
        
        return True
    
    def run(self, start_grade=0, end_grade=13, pulses_per_grade=5):
        """Run the full curriculum."""
        print(f"\n  Starting curriculum from grade {start_grade} to {end_grade}")
        print(f"  LLM: {'✓ ' + self.llm.model if self.llm.available else '✗ simulation mode'}")
        print(f"  Database: {self.db.db_path} ({self.db.count()} existing entries)")
        
        for grade in range(start_grade, end_grade + 1):
            if not self.running: break
            if grade not in CURRICULUM: continue
            self.teach_grade(grade, pulses_per_grade)
            self.current_grade = grade
        
        # Final report
        self._final_report()
    
    def _final_report(self):
        print(f"\n{'═'*60}")
        print(f"  📊 FINAL REPORT")
        print(f"{'═'*60}")
        
        total = self.db.count()
        t = self.db.count("T"); m = self.db.count("M"); f = self.db.count("F")
        print(f"\n  Total knowledge: {total} entries")
        print(f"  T={t} | F={f} | M={m}")
        
        # By grade
        by_grade = self.db.stats_by_grade()
        print(f"\n  ── By Grade Level ──")
        for grade in sorted(by_grade.keys()):
            info = CURRICULUM.get(grade, {"name": f"Grade {grade}"})
            stats = by_grade[grade]
            total_g = sum(stats.values())
            bar = "█" * min(total_g // 3, 40)
            print(f"  {info['name']:25s} T={stats['T']:4d} M={stats['M']:3d} F={stats['F']:2d} {bar}")
        
        # By source
        print(f"\n  ── By Source ──")
        for src, cnt in self.db.stats_by_source():
            bar = "█" * min(cnt // 3, 40)
            print(f"    {src:25s} {cnt:5d} {bar}")
        
        # Confusion map
        confusions = self.db.get_confusions()
        if confusions:
            print(f"\n  ── Confusion Map ({len(confusions)}) ──")
            for c in confusions:
                gname = CURRICULUM.get(c["grade_discovered"],{}).get("name","?")
                print(f"    ⚠ {c['subject']:25s} ≠ {c['confused_with']}")
                print(f"      Grade: {gname} | {c['reason'][:60]}")
        
        if self.llm.available:
            print(f"\n  LLM calls: {self.llm.call_count}")


# ═══════════════════════════════════════════════════════════
# WEB DASHBOARD (Flask)
# ═══════════════════════════════════════════════════════════

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Pulse System Dashboard</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family: 'Courier New', monospace; background:#0a0a0a; color:#e0e0e0; padding:20px; }
  h1 { color:#00ff88; margin-bottom:20px; font-size:1.5em; }
  h2 { color:#00aaff; margin:15px 0 8px; font-size:1.1em; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:15px; margin-bottom:15px; }
  .card { background:#1a1a1a; border:1px solid #333; border-radius:8px; padding:15px; }
  .stat { font-size:2em; color:#00ff88; font-weight:bold; }
  .stat-label { color:#888; font-size:0.85em; }
  .bar { display:inline-block; background:#00ff88; height:14px; margin-right:5px; }
  .bar-m { background:#ffaa00; }
  .bar-f { background:#ff4444; }
  .log-entry { padding:4px 0; border-bottom:1px solid #222; font-size:0.85em; }
  .log-time { color:#666; }
  .confusion { color:#ffaa00; padding:4px 0; }
  table { width:100%; border-collapse:collapse; font-size:0.85em; }
  td, th { padding:4px 8px; text-align:left; border-bottom:1px solid #222; }
  th { color:#00aaff; }
  .grade-row td:first-child { color:#00ff88; font-weight:bold; }
  #auto-refresh { color:#666; font-size:0.8em; }
  .full-width { grid-column: 1 / -1; }
</style>
</head>
<body>
<h1>⚡ Three-Layer Pulse System <span id="auto-refresh">refreshes every 3s</span></h1>

<div class="grid">
  <div class="card">
    <div class="stat" id="total">-</div>
    <div class="stat-label">Total Entries</div>
  </div>
  <div class="card">
    <div class="stat" id="grade">-</div>
    <div class="stat-label">Current Grade</div>
  </div>
</div>

<div class="grid">
  <div class="card">
    <div class="stat-label">Truth Values</div>
    <div style="margin-top:8px">
      <span style="color:#00ff88;font-size:1.4em" id="tv-t">-</span> T &nbsp;
      <span style="color:#ffaa00;font-size:1.4em" id="tv-m">-</span> M &nbsp;
      <span style="color:#ff4444;font-size:1.4em" id="tv-f">-</span> F
    </div>
  </div>
  <div class="card">
    <div class="stat-label">LLM Calls</div>
    <div class="stat" id="llm-calls">-</div>
  </div>
</div>

<div class="grid">
  <div class="card">
    <h2>By Grade</h2>
    <table id="grade-table"><tr><th>Grade</th><th>T</th><th>M</th><th>F</th><th></th></tr></table>
  </div>
  <div class="card">
    <h2>By Source</h2>
    <div id="sources"></div>
  </div>
</div>

<div class="grid">
  <div class="card">
    <h2>⚠ Confusion Map</h2>
    <div id="confusions"></div>
  </div>
  <div class="card">
    <h2>📋 Recent Log</h2>
    <div id="log"></div>
  </div>
</div>

<div class="grid">
  <div class="card full-width">
    <h2>? Unresolved (M)</h2>
    <div id="unresolved"></div>
  </div>
</div>

<script>
function refresh() {
  fetch('/api/stats').then(r=>r.json()).then(d => {
    document.getElementById('total').textContent = d.total;
    document.getElementById('grade').textContent = d.current_grade_name;
    document.getElementById('tv-t').textContent = d.t;
    document.getElementById('tv-m').textContent = d.m;
    document.getElementById('tv-f').textContent = d.f;
    document.getElementById('llm-calls').textContent = d.llm_calls;
    
    let gt = '';
    for (let g of d.by_grade) {
      let total = g.T + g.M + g.F;
      let bar = '<span class="bar" style="width:'+(g.T/2)+'px"></span>';
      bar += '<span class="bar bar-m" style="width:'+(g.M/2)+'px"></span>';
      gt += '<tr class="grade-row"><td>'+g.name+'</td><td>'+g.T+'</td><td>'+g.M+'</td><td>'+g.F+'</td><td>'+bar+'</td></tr>';
    }
    document.getElementById('grade-table').innerHTML = '<tr><th>Grade</th><th>T</th><th>M</th><th>F</th><th></th></tr>' + gt;
    
    let sh = '';
    for (let s of d.by_source) {
      sh += '<div style="margin:3px 0"><span class="bar" style="width:'+Math.min(s[1]/2,200)+'px"></span> '+s[0]+' ('+s[1]+')</div>';
    }
    document.getElementById('sources').innerHTML = sh;
    
    let ch = '';
    for (let c of d.confusions) {
      ch += '<div class="confusion">⚠ '+c.subject+' ≠ '+c.confused_with+'<br><span style="color:#888;font-size:0.85em">  '+c.reason+'</span></div>';
    }
    document.getElementById('confusions').innerHTML = ch || '<div style="color:#666">None found yet</div>';
    
    let lh = '';
    for (let l of d.log) {
      lh += '<div class="log-entry"><span class="log-time">'+l.timestamp+'</span> ['+l.event_type+'] '+l.message+'</div>';
    }
    document.getElementById('log').innerHTML = lh;
    
    let uh = '';
    for (let u of d.unresolved) {
      uh += '<div style="padding:2px 0;font-size:0.85em">? '+u.subject+' → '+u.predicate+'('+u.object+')</div>';
    }
    document.getElementById('unresolved').innerHTML = uh || '<div style="color:#666">All resolved!</div>';
  });
}
refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>
"""

def create_dashboard(db, llm, curriculum_engine):
    """Create a Flask web app for monitoring."""
    if not HAS_FLASK:
        print("  ⚠ Flask not installed. Dashboard disabled.")
        print("    Install with: pip install flask")
        return None
    
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        return render_template_string(DASHBOARD_HTML)
    
    @app.route('/api/stats')
    def stats():
        total = db.count()
        t = db.count("T"); m = db.count("M"); f = db.count("F")
        
        by_grade_raw = db.stats_by_grade()
        by_grade = []
        for g in sorted(by_grade_raw.keys()):
            name = CURRICULUM.get(g, {}).get("name", f"Grade {g}")
            by_grade.append({"name":name,"T":by_grade_raw[g]["T"],
                            "M":by_grade_raw[g]["M"],"F":by_grade_raw[g]["F"]})
        
        grade_name = CURRICULUM.get(curriculum_engine.current_grade,{}).get(
            "name", f"Grade {curriculum_engine.current_grade}")
        
        unresolved = db.all_entries("M")[:20]
        
        return jsonify({
            "total": total, "t": t, "m": m, "f": f,
            "current_grade_name": grade_name,
            "llm_calls": llm.call_count,
            "by_grade": by_grade,
            "by_source": db.stats_by_source(),
            "confusions": db.get_confusions(),
            "log": db.get_log(20),
            "unresolved": [{"subject":u["subject"],"predicate":u["predicate"],
                           "object":u["object"]} for u in unresolved],
        })
    
    return app


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Curriculum Pulse System")
    parser.add_argument("--db", default="curriculum.db")
    parser.add_argument("--model", default="llama3.2")
    parser.add_argument("--start-grade", type=int, default=0)
    parser.add_argument("--end-grade", type=int, default=13)
    parser.add_argument("--pulses-per-grade", type=int, default=5)
    parser.add_argument("--dashboard", action="store_true", help="Run web dashboard")
    parser.add_argument("--dashboard-port", type=int, default=5000)
    parser.add_argument("--seed-json", default=None, help="Also load a JSON seed file")
    args = parser.parse_args()
    
    print(f"{'═'*60}")
    print(f"  🎓 CURRICULUM PULSE SYSTEM")
    print(f"{'═'*60}")
    
    db = KnowledgeDB(args.db)
    llm = OllamaLLM(args.model)
    engine = PulseEngine(db)
    verifier = Verifier(db, llm)
    curriculum = CurriculumEngine(db, llm, engine, verifier)
    
    # Optional JSON seed
    if args.seed_json and os.path.exists(args.seed_json):
        with open(args.seed_json) as f:
            data = json.load(f)
        count = 0
        for e in data.get("entries", []):
            r = db.add(e["subject"],e["predicate"],e["object"],
                e.get("truth_value","T"),
                evidence_for=e.get("evidence_for",[]),
                source=e.get("source","seed"), grade_level=0)
            if r: count += 1
        print(f"  Loaded {count} seed entries from {args.seed_json}")
    
    # Dashboard
    if args.dashboard:
        app = create_dashboard(db, llm, curriculum)
        if app:
            print(f"\n  🌐 Dashboard: http://localhost:{args.dashboard_port}")
            thread = threading.Thread(
                target=lambda: app.run(port=args.dashboard_port, debug=False, use_reloader=False),
                daemon=True)
            thread.start()
            time.sleep(1)
    
    # Run curriculum
    try:
        curriculum.run(args.start_grade, args.end_grade, args.pulses_per_grade)
    except KeyboardInterrupt:
        print(f"\n\n  ⏹ Stopped by user")
        curriculum.running = False
        curriculum._final_report()
    
    print(f"\n  Database saved: {args.db}")

if __name__ == "__main__":
    main()
