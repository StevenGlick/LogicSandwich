#!/usr/bin/env python3
"""
KNOWLEDGE BRIDGE SYSTEM
═══════════════════════

Captures meaningful cross-domain connections that the main knowledge
base would normally block or miss. These are the "wait, those are
connected?" moments that make human experts valuable.

THE PROBLEM:
  Main KB: "apple is_a fruit" (everyday domain)
  Main KB: "gravity is_a force" (physics domain)
  Domain gate: BLOCKS "apple → related_to → gravity" (food ↔ physics)
  
  But Newton's apple IS connected to gravity — for a specific,
  documentable reason. That connection is valuable knowledge that
  would be lost.

THE SOLUTION:
  Bridge table: a separate space where cross-domain connections live
  with their justification, type, and strength attached. The main KB
  stays clean. Bridges capture the nuance.

BRIDGE TYPES:
  historical    — Real event created the connection (Newton's apple)
  inspirational — One domain borrowed structure from another (neural nets from biology)
  structural    — Same math/pattern in different domains (waves in sound AND light)
  metaphorical  — Useful for teaching but not literally true (atom = tiny solar system)  
  emergent      — System discovered a meaningful cross-domain pattern
  causal        — One domain actually affects another (climate → agriculture)

USAGE:
  from bridges import BridgeSystem
  
  bs = BridgeSystem("curriculum.db")
  bs.add_bridge("apple", "gravity", "historical",
      reason="Newton observed falling apple, leading to theory of gravitation",
      strength=0.9)
  
  # Query: find all bridges involving gravity
  bridges = bs.find_bridges("gravity")
  
  # Query: find bridges between two domains
  bridges = bs.bridges_between("biology", "computer_science")

  # Standalone:
  python3 bridges.py --analyze                    # find potential bridges in existing data
  python3 bridges.py --show                       # show all bridges
  python3 bridges.py --between biology physics    # bridges between two domains
"""

import sqlite3
import json
import argparse
from collections import defaultdict
from domain_scope import VALID_CONTEXTS


# ═══════════════════════════════════════════════════════════
# BRIDGE TABLE
# ═══════════════════════════════════════════════════════════

class BridgeSystem:
    
    VALID_TYPES = {
        # Original types
        "historical",      # real event created the connection
        "inspirational",   # one domain borrowed from another
        "structural",      # same math/pattern in different domains
        "metaphorical",    # useful for teaching, not literally true
        "emergent",        # system discovered it
        "causal",          # one domain actually affects another
        # Knowledge context types
        "cultural",        # same concept, different meaning across cultures
        "etymological",    # word origin reveals hidden domain connection
        "scale",           # same pattern at different scales (fractal, power law)
        "pedagogical",     # learning one domain requires/unlocks another
        "contradictory",   # two domains seem to conflict; resolution is insight
        "experiential",    # abstract concept maps to physical sensation
        # Meaning and value types
        "spiritual",       # personal transcendent experience maps to formal system
        "religious",       # institutional doctrine connects to knowledge domain
        "ethical",         # knowledge creates moral weight or obligation
        "aesthetic",       # formal structure maps to beauty/art/music
        # Meta-knowledge types
        "rumor",           # widely believed falsehood — fact ABOUT a belief
        "mythological",    # narrative/archetype that encodes real pattern
        "linguistic",      # language structure shapes thought (beyond etymology)
        "technological",   # pure science becomes applied tool
        # Embodied/perceptual types
        "emotional",       # emotions mapped onto non-emotional things (sad=blue, heavy news)
        "pattern",         # shared form/rhythm/shape across unrelated things
        "philosophical",   # deep conceptual parallel between domains of thought
        "sensory",         # how different senses map onto each other (synaesthesia, cross-modal)
        "humor",           # why something is funny reveals hidden structure or expectation
    }
    
    def __init__(self, db_path="curriculum.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.cur = self.conn.cursor()
        self._create_tables()
    
    def _create_tables(self):
        self.cur.executescript("""
            CREATE TABLE IF NOT EXISTS bridges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                term_a TEXT NOT NULL,
                term_b TEXT NOT NULL,
                domain_a TEXT DEFAULT '',
                domain_b TEXT DEFAULT '',
                bridge_type TEXT NOT NULL,
                reason TEXT NOT NULL,
                strength REAL DEFAULT 0.5,
                bidirectional INTEGER DEFAULT 1,
                teaching_note TEXT DEFAULT '',
                discovered_by TEXT DEFAULT 'manual',
                verified INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS bridge_examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bridge_id INTEGER NOT NULL,
                example TEXT NOT NULL,
                source TEXT DEFAULT '',
                FOREIGN KEY (bridge_id) REFERENCES bridges(id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_bridge_a ON bridges(term_a);
            CREATE INDEX IF NOT EXISTS idx_bridge_b ON bridges(term_b);
            CREATE INDEX IF NOT EXISTS idx_bridge_type ON bridges(bridge_type);
            CREATE INDEX IF NOT EXISTS idx_bridge_domains ON bridges(domain_a, domain_b);
        """)
        self.conn.commit()
    
    def add_bridge(self, term_a, term_b, bridge_type, reason,
                   strength=0.5, domain_a="", domain_b="",
                   bidirectional=True, teaching_note="",
                   discovered_by="manual", examples=None):
        """
        Add a bridge between two terms from different domains.
        
        Args:
            term_a, term_b: The two terms being connected
            bridge_type: One of VALID_TYPES
            reason: WHY these are connected (the valuable part)
            strength: 0.0-1.0, how strong/reliable the connection is
            domain_a, domain_b: What domains each term belongs to
            bidirectional: Is the connection meaningful both ways?
            teaching_note: How to use this bridge for explanation
            discovered_by: "manual", "llm", "pulse", "user"
            examples: List of example strings illustrating the bridge
        """
        if bridge_type not in self.VALID_TYPES:
            print(f"  ⚠ Unknown bridge type: {bridge_type}")
            print(f"    Valid types: {', '.join(sorted(self.VALID_TYPES))}")
            return None

        # Validate domain_a/domain_b against VALID_CONTEXTS (warn only, don't block)
        if domain_a and domain_a not in VALID_CONTEXTS:
            print(f"  ⚠ domain_a '{domain_a}' not in VALID_CONTEXTS (bridge will still be added)")
        if domain_b and domain_b not in VALID_CONTEXTS:
            print(f"  ⚠ domain_b '{domain_b}' not in VALID_CONTEXTS (bridge will still be added)")

        term_a = term_a.lower().replace(" ", "_")
        term_b = term_b.lower().replace(" ", "_")
        
        # Check for duplicate
        self.cur.execute(
            "SELECT id FROM bridges WHERE term_a=? AND term_b=? AND bridge_type=?",
            (term_a, term_b, bridge_type))
        if self.cur.fetchone():
            return None  # already exists
        
        self.cur.execute("""
            INSERT INTO bridges (term_a, term_b, domain_a, domain_b,
                bridge_type, reason, strength, bidirectional,
                teaching_note, discovered_by)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (term_a, term_b, domain_a, domain_b,
              bridge_type, reason, strength, int(bidirectional),
              teaching_note, discovered_by))
        
        bridge_id = self.cur.lastrowid
        
        # Add examples if provided
        if examples:
            for ex in examples:
                self.cur.execute(
                    "INSERT INTO bridge_examples (bridge_id, example) VALUES (?,?)",
                    (bridge_id, ex))
        
        self.conn.commit()
        return bridge_id
    
    def find_bridges(self, term):
        """Find all bridges involving a term."""
        term = term.lower().replace(" ", "_")
        self.cur.execute("""
            SELECT * FROM bridges 
            WHERE term_a=? OR term_b=?
            ORDER BY strength DESC
        """, (term, term))
        return [dict(r) for r in self.cur.fetchall()]
    
    def bridges_between_domains(self, domain_a, domain_b):
        """Find all bridges between two domains."""
        self.cur.execute("""
            SELECT * FROM bridges
            WHERE (domain_a=? AND domain_b=?) OR (domain_a=? AND domain_b=?)
            ORDER BY strength DESC
        """, (domain_a, domain_b, domain_b, domain_a))
        return [dict(r) for r in self.cur.fetchall()]
    
    def bridges_by_type(self, bridge_type):
        """Get all bridges of a specific type."""
        self.cur.execute(
            "SELECT * FROM bridges WHERE bridge_type=? ORDER BY strength DESC",
            (bridge_type,))
        return [dict(r) for r in self.cur.fetchall()]
    
    def get_examples(self, bridge_id):
        """Get examples for a specific bridge."""
        self.cur.execute(
            "SELECT example, source FROM bridge_examples WHERE bridge_id=?",
            (bridge_id,))
        return [dict(r) for r in self.cur.fetchall()]
    
    def all_bridges(self):
        self.cur.execute("SELECT * FROM bridges ORDER BY bridge_type, strength DESC")
        return [dict(r) for r in self.cur.fetchall()]
    
    def count(self):
        self.cur.execute("SELECT COUNT(*) FROM bridges")
        return self.cur.fetchone()[0]


# ═══════════════════════════════════════════════════════════
# BRIDGE DISCOVERY
# ═══════════════════════════════════════════════════════════
# Analyzes the existing knowledge base to find POTENTIAL
# bridges that a human or LLM should verify.
# ═══════════════════════════════════════════════════════════

class BridgeDiscovery:
    """
    Scans the knowledge base for cross-domain patterns that
    might be meaningful bridges rather than nonsense.
    
    The key distinction:
      "orange and neural_net both have layers" → NONSENSE (coincidence)
      "biological_neuron and artificial_neuron both process signals" → BRIDGE (inspirational)
    
    Heuristics for detecting real bridges:
      1. Shared predicate is meaningful (not just "has" or "is_a")
      2. The shared object is specific (not just "thing" or "type")
      3. There's a plausible causal or historical connection
      4. The connection has been noted by multiple sources
    """
    
    # Predicates that indicate meaningful cross-domain connections
    BRIDGE_PREDICATES = {
        "inspired_by", "modeled_after", "analogous_to",
        "borrows_from", "similar_mechanism_to",
        "same_equation_as", "isomorphic_to",
        "led_to_discovery_of", "enabled",
        "structurally_similar_to", "behaves_like",
    }
    
    # Predicates that are too generic for bridges
    NOISE_PREDICATES = {
        "is_a", "has", "has_part", "has_property",
        "related_to", "similar_to",
    }
    
    # Shared properties that suggest real structural bridges
    STRUCTURAL_PROPERTIES = {
        "branching", "oscillation", "feedback_loop",
        "equilibrium", "gradient", "diffusion",
        "resonance", "phase_transition", "entropy",
        "exponential_growth", "network_effects",
        "self_organization", "emergence",
        "conservation", "symmetry", "recursion",
    }
    
    def __init__(self, db_path="curriculum.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.cur = self.conn.cursor()
        
        # Try to import domain classifier
        try:
            from domain_scope import DomainClassifier
            self.classifier = DomainClassifier(self.conn)
        except ImportError:
            self.classifier = None
    
    def find_potential_bridges(self):
        """
        Scan the KB for cross-domain patterns that might be bridges.
        Returns candidates with confidence scores.
        """
        candidates = []
        
        # Strategy 1: Find cross-domain entries that share specific properties
        candidates.extend(self._find_shared_mechanisms())
        
        # Strategy 2: Find entries using bridge predicates
        candidates.extend(self._find_bridge_predicates())
        
        # Strategy 3: Find structural isomorphisms
        candidates.extend(self._find_structural_patterns())
        
        # Deduplicate and sort by confidence
        seen = set()
        unique = []
        for c in candidates:
            key = (c["term_a"], c["term_b"])
            rev_key = (c["term_b"], c["term_a"])
            if key not in seen and rev_key not in seen:
                seen.add(key)
                unique.append(c)
        
        unique.sort(key=lambda x: -x["confidence"])
        return unique
    
    def _find_shared_mechanisms(self):
        """Find different-domain subjects sharing specific predicates."""
        candidates = []
        
        # Get all T entries with non-noise predicates
        noise = "','".join(self.NOISE_PREDICATES)
        self.cur.execute(f"""
            SELECT subject, predicate, object FROM entries
            WHERE truth_value='T' AND subject NOT LIKE '%,%'
            AND predicate NOT IN ('{noise}')
        """)
        
        # Group by (predicate, object)
        po_groups = defaultdict(list)
        for row in self.cur.fetchall():
            po_groups[(row[1], row[2])].append(row[0])
        
        for (pred, obj), subjects in po_groups.items():
            if len(subjects) < 2:
                continue
            
            # Check if subjects span different domains
            if not self.classifier:
                continue
            
            domain_groups = defaultdict(list)
            for s in subjects:
                domains = self.classifier.classify(s)
                for d in domains:
                    domain_groups[d].append(s)
            
            if len(domain_groups) < 2:
                continue  # all same domain, not a bridge
            
            # Generate cross-domain pairs
            domain_list = list(domain_groups.keys())
            for i, d1 in enumerate(domain_list):
                for d2 in domain_list[i+1:]:
                    for s1 in domain_groups[d1]:
                        for s2 in domain_groups[d2]:
                            if s1 == s2:
                                continue
                            
                            # Score: specific predicate + specific object = higher confidence
                            conf = 0.4
                            if pred in self.BRIDGE_PREDICATES:
                                conf += 0.3
                            if obj in self.STRUCTURAL_PROPERTIES:
                                conf += 0.2
                            if len(obj) > 5:  # longer = more specific
                                conf += 0.1
                            
                            candidates.append({
                                "term_a": s1,
                                "term_b": s2,
                                "domain_a": d1,
                                "domain_b": d2,
                                "shared_property": f"{pred}({obj})",
                                "confidence": min(conf, 1.0),
                                "suggested_type": "structural" if obj in self.STRUCTURAL_PROPERTIES else "emergent",
                                "reason": f"Both {pred} {obj}, across {d1}/{d2} boundary",
                            })
        
        return candidates
    
    def _find_bridge_predicates(self):
        """Find entries that explicitly use bridge-type predicates."""
        candidates = []
        
        bridge_preds = "','".join(self.BRIDGE_PREDICATES)
        self.cur.execute(f"""
            SELECT * FROM entries
            WHERE predicate IN ('{bridge_preds}') AND truth_value='T'
        """)
        
        for row in self.cur.fetchall():
            entry = dict(row)
            candidates.append({
                "term_a": entry["subject"],
                "term_b": entry["object"],
                "domain_a": "",
                "domain_b": "",
                "shared_property": entry["predicate"],
                "confidence": 0.7,
                "suggested_type": "inspirational",
                "reason": f"Explicit {entry['predicate']} relationship in KB",
            })
        
        return candidates
    
    def _find_structural_patterns(self):
        """Find structural isomorphisms — same math in different domains."""
        candidates = []
        
        # Look for subjects from different domains that share
        # structural property keywords in their objects
        self.cur.execute("""
            SELECT subject, predicate, object FROM entries
            WHERE truth_value='T' AND subject NOT LIKE '%,%'
        """)
        
        subject_properties = defaultdict(set)
        for row in self.cur.fetchall():
            obj = row[2]
            for prop in self.STRUCTURAL_PROPERTIES:
                if prop in obj:
                    subject_properties[row[0]].add(prop)
        
        # Find pairs sharing structural properties across domains
        if self.classifier:
            subjects = list(subject_properties.keys())
            for i, s1 in enumerate(subjects):
                d1 = self.classifier.classify(s1)
                for s2 in subjects[i+1:]:
                    d2 = self.classifier.classify(s2)
                    if not d1 or not d2 or d1 == d2:
                        continue
                    
                    shared = subject_properties[s1] & subject_properties[s2]
                    if shared:
                        candidates.append({
                            "term_a": s1,
                            "term_b": s2,
                            "domain_a": str(d1),
                            "domain_b": str(d2),
                            "shared_property": ", ".join(sorted(shared)),
                            "confidence": min(0.3 + len(shared) * 0.15, 0.9),
                            "suggested_type": "structural",
                            "reason": f"Shared structural properties: {', '.join(sorted(shared))}",
                        })
        
        return candidates


# ═══════════════════════════════════════════════════════════
# LLM BRIDGE VERIFICATION
# ═══════════════════════════════════════════════════════════

def verify_bridge_with_llm(llm, term_a, term_b, domain_a, domain_b, shared_property):
    """
    Ask the LLM whether a potential bridge is meaningful.
    
    Returns the bridge info or None if it's nonsense.
    """
    if not llm or not llm.available:
        return None
    
    system = """You evaluate whether a cross-domain knowledge connection is 
meaningful or coincidental.

A MEANINGFUL bridge: biological_neuron and artificial_neuron both process signals
  → YES: neural networks were literally inspired by biological neurons

A COINCIDENTAL connection: cake and sedimentary_rock both have layers
  → NO: the word "layers" means completely different things

Respond in this format:
MEANINGFUL: yes/no
TYPE: historical/inspirational/structural/metaphorical/causal
STRENGTH: 0.0-1.0
REASON: one sentence explaining the real connection (or why it's not real)
TEACHING: one sentence on how this bridge helps understanding (if meaningful)
EXAMPLES: (optional) one or two concrete examples"""
    
    prompt = f"""Is this cross-domain connection meaningful or coincidental?

Term A: {term_a} (domain: {domain_a})
Term B: {term_b} (domain: {domain_b})
Shared property: {shared_property}"""
    
    response = llm.ask(prompt, system)
    if not response:
        return None
    
    result = {"meaningful": False}
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith("MEANINGFUL:"):
            result["meaningful"] = "yes" in line.lower()
        elif line.startswith("TYPE:"):
            result["type"] = line.split(":", 1)[1].strip().lower()
        elif line.startswith("STRENGTH:"):
            try:
                result["strength"] = float(line.split(":", 1)[1].strip())
            except (ValueError, IndexError):
                result["strength"] = 0.5
        elif line.startswith("REASON:"):
            result["reason"] = line.split(":", 1)[1].strip()
        elif line.startswith("TEACHING:"):
            result["teaching"] = line.split(":", 1)[1].strip()
        elif line.startswith("EXAMPLES:"):
            result["examples"] = line.split(":", 1)[1].strip()
    
    return result if result.get("meaningful") else None


# ═══════════════════════════════════════════════════════════
# SEED BRIDGES (well-known cross-domain connections)
# ═══════════════════════════════════════════════════════════

SEED_BRIDGES = [
    {
        "term_a": "apple", "term_b": "gravity",
        "domain_a": "everyday", "domain_b": "physics",
        "bridge_type": "historical",
        "reason": "Newton reportedly observed a falling apple, inspiring his theory of universal gravitation",
        "strength": 0.95,
        "teaching_note": "The apple didn't cause the theory — Newton had been working on orbital mechanics. The apple was the moment of synthesis where terrestrial and celestial gravity unified.",
        "examples": ["Newton's Principia (1687) unified falling objects and planetary orbits"],
    },
    {
        "term_a": "biological_neuron", "term_b": "neural_network",
        "domain_a": "biology", "domain_b": "computer_science",
        "bridge_type": "inspirational",
        "reason": "Artificial neural networks were inspired by biological neuron signaling, though they diverge significantly in mechanism",
        "strength": 0.8,
        "teaching_note": "The inspiration is real but the implementation is very different. Biological neurons use electrochemical signals; artificial ones use matrix multiplication. The bridge is structural, not mechanical.",
        "examples": ["McCulloch-Pitts neuron model (1943)", "Perceptron inspired by retinal cells"],
    },
    {
        "term_a": "natural_selection", "term_b": "genetic_algorithm",
        "domain_a": "biology", "domain_b": "computer_science",
        "bridge_type": "structural",
        "reason": "Genetic algorithms directly implement biological evolution: populations, fitness, mutation, crossover, selection",
        "strength": 0.9,
        "teaching_note": "One of the strongest bridges in CS — the algorithm IS the biology, formalized. Same math, different substrate.",
        "examples": ["Tournament selection mirrors predator-prey dynamics", "Crossover mirrors sexual reproduction"],
    },
    {
        "term_a": "water_flow", "term_b": "electrical_current",
        "domain_a": "everyday", "domain_b": "physics",
        "bridge_type": "structural",
        "reason": "Electrical current follows analogous equations to fluid flow: voltage=pressure, current=flow rate, resistance=pipe narrowing",
        "strength": 0.85,
        "teaching_note": "The hydraulic analogy is one of the most effective teaching tools in physics. The math is genuinely parallel (Ohm's law ↔ Poiseuille's law).",
    },
    {
        "term_a": "atom", "term_b": "solar_system",
        "domain_a": "chemistry", "domain_b": "physics",
        "bridge_type": "metaphorical",
        "reason": "The Bohr model depicted electrons orbiting the nucleus like planets orbit the sun. Useful for initial teaching but fundamentally wrong — electrons exist as probability clouds, not orbits.",
        "strength": 0.3,
        "teaching_note": "CAUTION: This bridge is a known source of misconceptions. Mark it as metaphorical and note where it breaks down. Electron orbitals are probability distributions, not paths.",
    },
    {
        "term_a": "tree", "term_b": "tree_data_structure",
        "domain_a": "biology", "domain_b": "computer_science",
        "bridge_type": "structural",
        "reason": "Both exhibit hierarchical branching from a single root. The mathematical structure (graph with no cycles, single root, parent-child relationships) is identical.",
        "strength": 0.75,
        "teaching_note": "CS trees are drawn upside-down compared to biological trees (root at top). The branching math is genuinely the same.",
    },
    {
        "term_a": "immune_system", "term_b": "antivirus_software",
        "domain_a": "biology", "domain_b": "computer_science",
        "bridge_type": "inspirational",
        "reason": "Computer virus/antivirus concepts were directly inspired by biological virus/immune system dynamics. Signature matching mirrors antibody recognition.",
        "strength": 0.7,
        "teaching_note": "The terminology was borrowed deliberately. Modern ML-based threat detection increasingly resembles adaptive immunity.",
    },
    {
        "term_a": "river_delta", "term_b": "blood_vessel_network",
        "domain_a": "earth_science", "domain_b": "biology",
        "bridge_type": "structural",
        "reason": "Both are branching distribution networks optimized by flow dynamics. They follow the same mathematical branching laws (Murray's law for vessels, Horton's law for rivers).",
        "strength": 0.8,
        "teaching_note": "Nature converges on the same branching solutions for distribution problems regardless of scale or substrate.",
    },
    {
        "term_a": "music_harmony", "term_b": "wave_interference",
        "domain_a": "everyday", "domain_b": "physics",
        "bridge_type": "causal",
        "reason": "Musical harmony IS wave physics. Consonant intervals correspond to simple frequency ratios. Dissonance is destructive interference.",
        "strength": 0.95,
        "teaching_note": "This is not an analogy — music literally IS applied wave physics. Pythagoras discovered this with string lengths.",
    },
    {
        "term_a": "predator_prey_cycle", "term_b": "market_supply_demand",
        "domain_a": "biology", "domain_b": "economics",
        "bridge_type": "structural",
        "reason": "Both follow Lotka-Volterra oscillation dynamics. Predator/prey populations oscillate like supply/demand price cycles. Same differential equations.",
        "strength": 0.7,
        "teaching_note": "The mathematical structure is identical. Both show damped or sustained oscillations depending on parameters.",
    },
    
    # ── CULTURAL ──
    {
        "term_a": "dragon_eastern", "term_b": "dragon_western",
        "domain_a": "culture_east_asian", "domain_b": "culture_european",
        "bridge_type": "cultural",
        "reason": "Same creature name, opposite symbolic meaning. Eastern dragons represent wisdom, power, and good fortune. Western dragons represent greed, destruction, and evil.",
        "strength": 0.9,
        "teaching_note": "Demonstrates that symbols are not universal — cultural context determines meaning. A student interpreting Chinese art with Western assumptions will get it backwards.",
    },
    {
        "term_a": "white_color", "term_b": "mourning",
        "domain_a": "culture_western", "domain_b": "culture_east_asian",
        "bridge_type": "cultural",
        "reason": "White represents purity and celebration in Western culture (weddings) but mourning and death in much of East Asia (funerals). Same color, opposite emotional associations.",
        "strength": 0.85,
    },
    {
        "term_a": "number_four", "term_b": "death",
        "domain_a": "mathematics", "domain_b": "culture_east_asian",
        "bridge_type": "cultural",
        "reason": "In Japanese and Chinese, four (shi/si) is homophonous with the word for death. Buildings skip 4th floors, gifts avoid sets of four. A purely linguistic accident that shapes architecture and social behavior.",
        "strength": 0.8,
        "teaching_note": "Shows how language shapes culture shapes physical infrastructure. Pure math has no 'unlucky' numbers, but cultural math does.",
    },
    {
        "term_a": "left_hand", "term_b": "sinister",
        "domain_a": "biology", "domain_b": "culture_european",
        "bridge_type": "cultural",
        "reason": "Latin 'sinister' literally means 'left.' Left-handedness was associated with evil across European, Middle Eastern, and South Asian cultures, affecting everything from handwriting instruction to table manners.",
        "strength": 0.7,
        "teaching_note": "A biological trait (handedness) became a moral judgment through cultural bridge. 'Sinister' preserves this in modern English.",
    },
    {
        "term_a": "zero", "term_b": "void",
        "domain_a": "mathematics", "domain_b": "philosophy",
        "bridge_type": "cultural",
        "reason": "Zero was invented in India partly because Hindu-Buddhist philosophy already had a concept of void/emptiness (shunya). Greek mathematics resisted zero for centuries because their philosophy rejected the concept of nothing. Culture enabled or blocked mathematical progress.",
        "strength": 0.85,
        "teaching_note": "One of the most important cultural bridges in history. Philosophy literally determined which civilizations could do advanced mathematics.",
    },
    
    # ── ETYMOLOGICAL ──
    {
        "term_a": "malaria", "term_b": "bad_air",
        "domain_a": "medicine", "domain_b": "history",
        "bridge_type": "etymological",
        "reason": "Italian 'mala aria' (bad air). Disease was blamed on swamp vapors. The word preserves a medical misunderstanding — it's actually transmitted by mosquitoes, which happen to live near swamps.",
        "strength": 0.8,
        "teaching_note": "The etymology reveals the history of scientific error. The name is wrong but the correlation (swamps = disease) was real — just the mechanism was wrong.",
    },
    {
        "term_a": "algorithm", "term_b": "al_khwarizmi",
        "domain_a": "computer_science", "domain_b": "history",
        "bridge_type": "etymological",
        "reason": "Named after Muhammad ibn Musa al-Khwarizmi, 9th century Persian mathematician. Also gives us 'algebra' (al-jabr). Modern computing terminology traces directly to Islamic Golden Age scholarship.",
        "strength": 0.85,
        "teaching_note": "Connects modern tech to medieval Islamic mathematics. Students often don't realize CS has roots in 9th century Baghdad.",
    },
    {
        "term_a": "atom", "term_b": "uncuttable",
        "domain_a": "physics", "domain_b": "philosophy",
        "bridge_type": "etymological",
        "reason": "Greek 'atomos' means indivisible/uncuttable. Democritus proposed the concept philosophically ~400 BCE. The name stuck even after we split the atom in 1932 — the word is now literally wrong.",
        "strength": 0.9,
        "teaching_note": "Perfect example of how names outlive their accuracy. Also bridges ancient philosophy to modern physics — Democritus was right about particles, wrong about indivisibility.",
    },
    {
        "term_a": "disaster", "term_b": "bad_star",
        "domain_a": "everyday", "domain_b": "astronomy",
        "bridge_type": "etymological",
        "reason": "Italian 'disastro' from Latin 'dis-' (bad) + 'astrum' (star). Catastrophes were blamed on unfavorable astrological alignments. The word preserves pre-scientific causal reasoning.",
        "strength": 0.6,
    },
    {
        "term_a": "salary", "term_b": "salt",
        "domain_a": "economics", "domain_b": "chemistry",
        "bridge_type": "etymological",
        "reason": "Latin 'salarium' — Roman soldiers were sometimes paid in salt or given money to buy salt. Bridges modern economics to ancient resource scarcity and trade.",
        "strength": 0.7,
        "teaching_note": "Reveals that money is an abstraction of real resource value. Salt was genuinely precious — connected to 'salvation' too (preserving life).",
    },
    
    # ── SCALE ──
    {
        "term_a": "spiral_galaxy", "term_b": "hurricane",
        "domain_a": "astronomy", "domain_b": "earth_science",
        "bridge_type": "scale",
        "reason": "Both exhibit logarithmic spiral structure driven by rotational dynamics and differential velocity. Same mathematical form (golden spiral approximation) at scales differing by 10^15.",
        "strength": 0.7,
        "teaching_note": "The spiral emerges from rotation + radial flow at any scale. Not coincidence — it's a mathematical attractor for rotating systems.",
        "examples": ["Also: nautilus shells, sunflower seeds, DNA helix"],
    },
    {
        "term_a": "river_network", "term_b": "lung_bronchi",
        "domain_a": "earth_science", "domain_b": "biology",
        "bridge_type": "scale",
        "reason": "Both are fractal branching networks optimized for distribution. Same branching ratios (Horton's law / Murray's law). Nature converges on identical geometry for moving fluids from one source to many destinations.",
        "strength": 0.85,
        "teaching_note": "Also: tree roots, blood vessels, lightning bolts. All solve the same optimization problem at different scales.",
    },
    {
        "term_a": "earthquake_frequency", "term_b": "word_frequency",
        "domain_a": "earth_science", "domain_b": "linguistics",
        "bridge_type": "scale",
        "reason": "Both follow power law distributions. The Gutenberg-Richter law (earthquakes) and Zipf's law (word frequency) have the same mathematical form. A few events are very common, many events are very rare.",
        "strength": 0.8,
        "teaching_note": "Power laws appear in city sizes, wealth distribution, website traffic, species abundance. Same math everywhere — suggests a deep generative principle.",
        "examples": ["Pareto principle (80/20 rule) is a power law", "Also: citation counts, crater sizes on the Moon"],
    },
    {
        "term_a": "ant_colony", "term_b": "neuron_network",
        "domain_a": "biology", "domain_b": "biology",
        "bridge_type": "scale",
        "reason": "Individual ants follow simple rules; the colony exhibits complex intelligent behavior (pathfinding, resource allocation). Individual neurons follow simple rules; the brain exhibits consciousness. Emergence at different scales.",
        "strength": 0.75,
        "teaching_note": "Neither the ant nor the neuron 'understands' the whole. Intelligence is a property of the network, not the node. This is the core insight of emergence.",
    },
    
    # ── PEDAGOGICAL ──
    {
        "term_a": "fractions", "term_b": "music_time_signature",
        "domain_a": "mathematics", "domain_b": "music",
        "bridge_type": "pedagogical",
        "reason": "Understanding time signatures (3/4, 6/8) requires fraction literacy. Students who struggle with fractions cannot read music notation. Math is a prerequisite for music theory.",
        "strength": 0.8,
        "teaching_note": "Music can also teach fractions — hearing the difference between 3/4 and 4/4 time gives fractions a physical, auditory meaning.",
    },
    {
        "term_a": "classical_mechanics", "term_b": "quantum_mechanics",
        "domain_a": "physics", "domain_b": "physics",
        "bridge_type": "pedagogical",
        "reason": "Quantum mechanics is partly defined by how it differs from classical mechanics. You cannot understand superposition without first understanding definite position. The 'old' theory is prerequisite for the 'new' one.",
        "strength": 0.9,
        "teaching_note": "This is true for many paradigm shifts: you need Newtonian gravity before general relativity, Euclidean geometry before non-Euclidean, etc.",
    },
    {
        "term_a": "arithmetic", "term_b": "chemistry_stoichiometry",
        "domain_a": "mathematics", "domain_b": "chemistry",
        "bridge_type": "pedagogical",
        "reason": "Balancing chemical equations requires ratio/proportion skills from arithmetic. Students who can't do ratios cannot do stoichiometry. Math gates chemistry.",
        "strength": 0.85,
        "teaching_note": "This is why the curriculum teaches ratios in 6th grade and chemistry in 10th — there's a hidden dependency chain across subjects.",
    },
    {
        "term_a": "reading_comprehension", "term_b": "word_problems",
        "domain_a": "language", "domain_b": "mathematics",
        "bridge_type": "pedagogical",
        "reason": "Students who struggle with reading comprehension fail math word problems even when they can do the math. Language skills gate mathematical application. Literacy is a hidden prerequisite for numeracy.",
        "strength": 0.9,
        "teaching_note": "This is one of the most overlooked pedagogical bridges. A math teacher encountering word problem failures may actually be seeing a reading problem.",
    },
    
    # ── CONTRADICTORY ──
    {
        "term_a": "determinism", "term_b": "quantum_randomness",
        "domain_a": "physics_classical", "domain_b": "physics_quantum",
        "bridge_type": "contradictory",
        "reason": "Classical mechanics: the universe is deterministic (Laplace's demon). Quantum mechanics: fundamentally probabilistic (Born rule). Resolution: determinism is an excellent approximation at macro scale, but doesn't hold at quantum scale. Neither is 'wrong.'",
        "strength": 0.95,
        "teaching_note": "This is not a contradiction to resolve but a boundary to understand. The question 'is the universe deterministic?' has different correct answers at different scales.",
    },
    {
        "term_a": "rational_actor", "term_b": "cognitive_bias",
        "domain_a": "economics", "domain_b": "psychology",
        "bridge_type": "contradictory",
        "reason": "Economics assumes rational actors maximizing utility. Psychology demonstrates systematic cognitive biases (anchoring, loss aversion, framing). Resolution: behavioral economics bridges both — humans are predictably irrational.",
        "strength": 0.85,
        "teaching_note": "Kahneman and Tversky's work is literally this bridge made into a field. Won the Nobel Prize in Economics for psychological research.",
    },
    {
        "term_a": "evolution_random", "term_b": "development_ordered",
        "domain_a": "biology_evolution", "domain_b": "biology_development",
        "bridge_type": "contradictory",
        "reason": "Evolution operates through random mutation. Embryonic development is highly ordered and reproducible. Resolution: evolution acts on the developmental program itself — randomness at one level produces order at another through selection.",
        "strength": 0.8,
        "teaching_note": "Evo-devo (evolutionary developmental biology) is the field that bridges this. Random exploration of ordered program space.",
    },
    {
        "term_a": "light_wave", "term_b": "light_particle",
        "domain_a": "physics", "domain_b": "physics",
        "bridge_type": "contradictory",
        "reason": "Light behaves as a wave (interference, diffraction) AND as a particle (photoelectric effect). Resolution: wave-particle duality — it's neither, it's a quantum object that shows different aspects depending on measurement.",
        "strength": 0.95,
        "teaching_note": "The resolution isn't 'it's both' — it's 'our classical categories don't apply.' This is a bridge where the contradiction itself is the lesson.",
    },
    
    # ── EXPERIENTIAL ──
    {
        "term_a": "temperature", "term_b": "molecular_kinetic_energy",
        "domain_a": "everyday", "domain_b": "physics",
        "bridge_type": "experiential",
        "reason": "Temperature is average molecular kinetic energy. We experience it as hot/cold. The felt sensation maps directly to a measurable physical quantity — faster molecules = hotter feeling.",
        "strength": 0.9,
        "teaching_note": "One of the cleanest experiential bridges. Every student already knows what hot and cold feel like — you're giving a name to what their nerves already measure.",
    },
    {
        "term_a": "entropy", "term_b": "things_fall_apart",
        "domain_a": "physics", "domain_b": "everyday",
        "bridge_type": "experiential",
        "reason": "Entropy (second law of thermodynamics) is why rooms get messy, ice melts, and you can't unscramble an egg. The universal experience of disorder increasing IS entropy, felt at human scale.",
        "strength": 0.85,
        "teaching_note": "Students intuitively understand entropy before they know the word. Their bedroom is a high-entropy system. Building that bridge makes thermodynamics personal.",
    },
    {
        "term_a": "inertia", "term_b": "being_pushed_in_car",
        "domain_a": "physics", "domain_b": "everyday",
        "bridge_type": "experiential",
        "reason": "Newton's first law (objects resist changes in motion) is felt every time a car brakes and you lurch forward. The abstract law maps directly to a bodily sensation everyone has experienced.",
        "strength": 0.9,
        "teaching_note": "Start with the car experience, then name it. The body already understands F=ma before the mind does.",
    },
    {
        "term_a": "resonance_frequency", "term_b": "pushing_a_swing",
        "domain_a": "physics", "domain_b": "everyday",
        "bridge_type": "experiential",
        "reason": "Every child learns resonance by pushing a swing — push at the right frequency and the amplitude grows, push at the wrong frequency and it fights you. This is exactly resonant frequency coupling.",
        "strength": 0.95,
        "teaching_note": "Possibly the most universally experienced physics concept. Every playground teaches resonance. Bridge this to radio tuning, microwave ovens, MRI machines.",
    },
    
    # ── SPIRITUAL ──
    {
        "term_a": "meditation", "term_b": "default_mode_network",
        "domain_a": "spiritual_practice", "domain_b": "neuroscience",
        "bridge_type": "spiritual",
        "reason": "Experienced meditators describe 'ego dissolution' and 'cessation of mental chatter.' fMRI studies show reduced Default Mode Network activity — the brain region associated with self-referential thought literally quiets down. Subjective report matches objective measurement.",
        "strength": 0.85,
        "teaching_note": "One of the cleanest spiritual-to-scientific bridges. Thousands of years of contemplative reports now have measurable neural correlates. Neither side is 'more real' — they're describing the same phenomenon in different vocabularies.",
    },
    {
        "term_a": "flow_state", "term_b": "samadhi",
        "domain_a": "psychology", "domain_b": "spiritual_practice",
        "bridge_type": "spiritual",
        "reason": "Csikszentmihalyi's 'flow' (total absorption, loss of self-consciousness, time distortion) maps closely to descriptions of samadhi in Hindu/Buddhist texts. Different frameworks, strikingly similar phenomenology.",
        "strength": 0.7,
        "teaching_note": "Flow is samadhi in a lab coat. Both describe a state where the boundary between subject and activity dissolves. The spiritual traditions mapped it first; psychology gave it metrics.",
    },
    {
        "term_a": "interconnectedness", "term_b": "ecosystem_web",
        "domain_a": "spiritual_practice", "domain_b": "ecology",
        "bridge_type": "spiritual",
        "reason": "Mystical traditions across cultures describe a felt sense of interconnection with all life. Ecology demonstrates this is literally true — every organism depends on webs of relationship, nutrient cycles, atmospheric exchange. The spiritual intuition maps to measurable biological reality.",
        "strength": 0.7,
        "teaching_note": "Not saying mysticism predicted ecology. Saying the felt sense of connection has a real referent. Contemplate a tree long enough and you notice it's breathing what you exhale.",
    },
    {
        "term_a": "psychedelic_experience", "term_b": "neural_entropy",
        "domain_a": "spiritual_practice", "domain_b": "neuroscience",
        "bridge_type": "spiritual",
        "reason": "Psychedelic experiences (ego dissolution, synaesthesia, sense of cosmic unity) correlate with increased neural entropy — more diverse brain connectivity patterns. The brain becomes less ordered, and the subjective experience is of boundaries dissolving. Disorder in the brain maps to dissolution of categories in experience.",
        "strength": 0.75,
    },
    
    # ── RELIGIOUS ──
    {
        "term_a": "genesis_creation", "term_b": "big_bang",
        "domain_a": "religion_abrahamic", "domain_b": "cosmology",
        "bridge_type": "religious",
        "reason": "Genesis: 'Let there be light' — the universe begins with light emerging from void. Big Bang cosmology: the universe begins with a hot dense state, photons decouple ~380,000 years in (CMB). Georges Lemaître, who first proposed the expanding universe, was a Catholic priest.",
        "strength": 0.6,
        "teaching_note": "The parallel is interesting historically but the mechanisms are completely different. Lemaître himself insisted his physics and his theology were separate questions. The bridge is historical and inspirational, not a claim of equivalence.",
    },
    {
        "term_a": "karma", "term_b": "newtons_third_law",
        "domain_a": "religion_dharmic", "domain_b": "physics",
        "bridge_type": "religious",
        "reason": "Karma: every action has consequences that return to the actor. Newton's third law: every action has an equal and opposite reaction. Structural parallel — both assert reciprocity as a fundamental principle. One is moral, one is physical.",
        "strength": 0.4,
        "teaching_note": "CAUTION: This is a structural parallel, not an equivalence. Karma operates across lifetimes through moral causation; Newton's third law operates instantaneously through physical force. The bridge is in the principle of reciprocity, not the mechanism.",
    },
    {
        "term_a": "first_mover", "term_b": "causal_chain",
        "domain_a": "theology", "domain_b": "philosophy",
        "bridge_type": "religious",
        "reason": "Aquinas's cosmological argument (everything caused needs a cause, so there must be an uncaused first cause) is formally identical to the philosophical problem of infinite regress. The theological argument IS a logical argument regardless of whether the conclusion is accepted.",
        "strength": 0.8,
        "teaching_note": "The logical structure is valid even if you reject the conclusion. This bridge connects medieval theology to modern philosophy of causation and to the question of why anything exists at all.",
    },
    {
        "term_a": "sabbath_rest", "term_b": "circadian_rhythm",
        "domain_a": "religion_abrahamic", "domain_b": "biology",
        "bridge_type": "religious",
        "reason": "Religious mandate for periodic rest (Sabbath, prayer schedules, monastic hours) maps to biological evidence that humans require rest cycles for cognitive function. Whether or not the mandate is divine, the biology supports it.",
        "strength": 0.5,
        "teaching_note": "Religious practices often encode practical wisdom in doctrinal form. The question of whether the wisdom preceded or followed the doctrine is itself interesting.",
    },
    
    # ── ETHICAL ──
    {
        "term_a": "nuclear_fission", "term_b": "weapons_dilemma",
        "domain_a": "physics", "domain_b": "ethics",
        "bridge_type": "ethical",
        "reason": "Splitting the atom is physics. Whether to build nuclear weapons, and whether to use them, is ethics. The knowledge cannot be unknown once discovered. Oppenheimer: 'Now I am become Death, the destroyer of worlds.' Knowledge created irreversible moral obligation.",
        "strength": 0.95,
        "teaching_note": "The defining ethical bridge of the 20th century. Physics gave humanity the power to destroy itself. The ethical question of what to do with dangerous knowledge has no technical answer.",
    },
    {
        "term_a": "crispr", "term_b": "designer_babies",
        "domain_a": "biology", "domain_b": "ethics",
        "bridge_type": "ethical",
        "reason": "CRISPR gene editing is biology/chemistry. Whether to edit human germline cells — changes that pass to all descendants — is ethics. He Jiankui edited human embryos in 2018; the scientific community condemned it. Capability preceded ethical consensus.",
        "strength": 0.9,
        "teaching_note": "Technology moves faster than ethics. By the time society decides what's acceptable, the thing has already been done. This bridge is about the gap between 'can' and 'should.'",
    },
    {
        "term_a": "artificial_intelligence", "term_b": "alignment_problem",
        "domain_a": "computer_science", "domain_b": "ethics",
        "bridge_type": "ethical",
        "reason": "Building AI systems is engineering. Ensuring they do what we actually want — and defining what 'want' means — is ethics and philosophy. The alignment problem is fundamentally about values, not code.",
        "strength": 0.9,
        "teaching_note": "You cannot solve alignment with more compute. It requires answering 'what is good?' — a question philosophy has been working on for 3,000 years without consensus.",
    },
    {
        "term_a": "climate_science", "term_b": "intergenerational_justice",
        "domain_a": "earth_science", "domain_b": "ethics",
        "bridge_type": "ethical",
        "reason": "Climate models are science. Who bears the cost — current generations who benefit from fossil fuels or future generations who suffer the consequences — is ethics. The science creates a moral obligation that spans time.",
        "strength": 0.85,
    },
    {
        "term_a": "surveillance_technology", "term_b": "privacy_rights",
        "domain_a": "computer_science", "domain_b": "ethics",
        "bridge_type": "ethical",
        "reason": "Building surveillance systems is engineering. Whether to deploy them, who watches the watchers, and where the line falls between security and freedom is ethics. Every capability creates a choice.",
        "strength": 0.8,
    },
    
    # ── AESTHETIC ──
    {
        "term_a": "golden_ratio", "term_b": "visual_beauty",
        "domain_a": "mathematics", "domain_b": "art",
        "bridge_type": "aesthetic",
        "reason": "The ratio 1:1.618 appears in Greek architecture (Parthenon), Renaissance painting (da Vinci), modern design, and natural forms (sunflowers, shells). Whether humans find it inherently beautiful or it's cultural conditioning is debated, but the correlation is documented.",
        "strength": 0.7,
        "teaching_note": "The golden ratio's aesthetic power is sometimes overstated (not everything beautiful uses it, not everything using it is beautiful). But its appearance across art and nature is a legitimate mathematical pattern.",
    },
    {
        "term_a": "minor_key", "term_b": "sadness",
        "domain_a": "music", "domain_b": "psychology",
        "bridge_type": "aesthetic",
        "reason": "Minor keys are widely perceived as 'sad' across Western cultures and many others. The interval relationships create acoustic beating patterns that the auditory system processes differently from major keys. Physics of sound maps to emotional response.",
        "strength": 0.75,
        "teaching_note": "Not fully universal — some cultures don't have the same association. But the acoustic basis (rougher beating patterns in minor intervals) is objective physics.",
    },
    {
        "term_a": "color_wavelength", "term_b": "emotional_response",
        "domain_a": "physics", "domain_b": "psychology",
        "bridge_type": "aesthetic",
        "reason": "Red (long wavelength) increases heart rate and arousal. Blue (short wavelength) promotes calm. These responses are partially biological (rod/cone sensitivity curves) and partially cultural, but the physics-to-emotion pipeline is measurable.",
        "strength": 0.6,
        "teaching_note": "Color psychology is real but often oversimplified. The biological component (warm colors excite, cool colors calm) interacts with cultural meaning (red = luck in China, danger in the West).",
    },
    {
        "term_a": "symmetry", "term_b": "facial_attractiveness",
        "domain_a": "mathematics", "domain_b": "biology",
        "bridge_type": "aesthetic",
        "reason": "Mathematical symmetry maps to perceived beauty across cultures. Facial symmetry correlates with perceived attractiveness (cross-culturally documented). Evolutionary hypothesis: symmetry signals developmental stability and genetic fitness.",
        "strength": 0.75,
        "teaching_note": "Bridges math, biology, and psychology in a single connection. Beauty may be partially a fitness signal processed through mathematical pattern detection.",
    },
    {
        "term_a": "counterpoint", "term_b": "parallel_processing",
        "domain_a": "music", "domain_b": "computer_science",
        "bridge_type": "aesthetic",
        "reason": "Musical counterpoint (Bach's fugues) runs multiple independent melodic lines simultaneously, following rules about when they can converge and diverge. Parallel computing runs multiple independent processes with rules about synchronization. Same structural problem, different substrate.",
        "strength": 0.65,
        "teaching_note": "Bach was essentially a parallel processing architect working in sound. The fugue is a program with multiple threads.",
    },
    
    # ── RUMOR ──
    {
        "term_a": "ten_percent_brain", "term_b": "brain_function",
        "domain_a": "popular_belief", "domain_b": "neuroscience",
        "bridge_type": "rumor",
        "reason": "WIDELY BELIEVED: Humans only use 10% of their brains. FACT: fMRI shows virtually all brain regions are active; damage to any area causes deficits. The myth persists because it implies untapped potential. Its popularity reveals more about human psychology than neuroscience.",
        "strength": 0.9,
        "teaching_note": "The rumor is false but the fact that people believe it is true and important. It drives purchase of 'brain training' products and shapes public understanding of neuroscience. A knowledge system needs to model what people believe, not just what's true.",
    },
    {
        "term_a": "flat_earth", "term_b": "spherical_earth",
        "domain_a": "popular_belief", "domain_b": "earth_science",
        "bridge_type": "rumor",
        "reason": "WIDELY BELIEVED (by a small but vocal group): The Earth is flat. FACT: Spherical, confirmed by literally every measurement method available. The rumor's persistence reveals how distrust of institutions can override direct evidence.",
        "strength": 0.95,
        "teaching_note": "The interesting question isn't 'is Earth flat' (no) but 'why do people believe it is.' The rumor bridge captures the social/psychological dimension that a pure fact misses.",
    },
    {
        "term_a": "sugar_hyperactivity", "term_b": "child_behavior",
        "domain_a": "popular_belief", "domain_b": "medicine",
        "bridge_type": "rumor",
        "reason": "WIDELY BELIEVED: Sugar makes children hyperactive. FACT: Double-blind studies consistently show no link. Parents who THINK their child had sugar rate the child as more hyperactive (expectation bias). The rumor reveals confirmation bias in real time.",
        "strength": 0.85,
        "teaching_note": "A perfect teaching tool for confirmation bias. The belief is so strong that telling parents the studies doesn't change their minds — they trust their own observation over controlled experiments.",
    },
    {
        "term_a": "goldfish_memory", "term_b": "fish_cognition",
        "domain_a": "popular_belief", "domain_b": "biology",
        "bridge_type": "rumor",
        "reason": "WIDELY BELIEVED: Goldfish have 3-second memories. FACT: Goldfish can be trained to navigate mazes, remember feeding times for months, and recognize their owners. The myth persists because it's a useful metaphor for forgetfulness.",
        "strength": 0.8,
    },
    {
        "term_a": "lightning_same_place", "term_b": "lightning_physics",
        "domain_a": "popular_belief", "domain_b": "physics",
        "bridge_type": "rumor",
        "reason": "WIDELY BELIEVED: Lightning never strikes the same place twice. FACT: Tall structures get struck repeatedly (Empire State Building: ~20 times/year). Lightning follows the path of least resistance, which is often the same path. The saying is folk wisdom about luck, not physics.",
        "strength": 0.8,
    },
    
    # ── MYTHOLOGICAL ──
    {
        "term_a": "flood_myth", "term_b": "sea_level_rise",
        "domain_a": "mythology", "domain_b": "earth_science",
        "bridge_type": "mythological",
        "reason": "Nearly every culture has a great flood narrative (Noah, Gilgamesh, Manu, Deucalion, Nu Wa). Post-glacial sea level rose ~120m between 20,000-6,000 BCE, flooding coastlines globally. Myths may encode collective memory of real geological events across thousands of years.",
        "strength": 0.7,
        "teaching_note": "Myths aren't random — they often preserve real observations in narrative form. The challenge is separating the signal (real flooding happened) from the narrative elaboration (divine punishment).",
    },
    {
        "term_a": "heros_journey", "term_b": "psychological_development",
        "domain_a": "mythology", "domain_b": "psychology",
        "bridge_type": "mythological",
        "reason": "Campbell's monomyth (departure, initiation, return) appears across unrelated cultures. Jung argued it maps to psychological individuation — the ego confronting the unconscious. The narrative structure may encode the universal pattern of human maturation.",
        "strength": 0.7,
        "teaching_note": "Whether the hero's journey is truly universal or a Western lens imposed on other cultures is debated. But the structural recurrence is documented across many traditions.",
    },
    {
        "term_a": "trickster_archetype", "term_b": "creative_destruction",
        "domain_a": "mythology", "domain_b": "economics",
        "bridge_type": "mythological",
        "reason": "Trickster figures (Loki, Coyote, Anansi, Hermes) break rules to create new possibilities. Schumpeter's 'creative destruction' describes how innovation destroys old systems to create new ones. Both encode the principle that disruption is generative.",
        "strength": 0.6,
        "teaching_note": "The trickster isn't evil — it's the force that prevents systems from stagnating. Silicon Valley's 'move fast and break things' is Coyote in a hoodie.",
    },
    {
        "term_a": "ouroboros", "term_b": "feedback_loop",
        "domain_a": "mythology", "domain_b": "systems_theory",
        "bridge_type": "mythological",
        "reason": "The ouroboros (serpent eating its tail) appears in Egyptian, Greek, Norse, and Hindu mythology. It depicts self-reference and cyclical processes — exactly what systems theory calls a feedback loop. Kekulé reportedly dreamed of an ouroboros before discovering benzene's ring structure.",
        "strength": 0.75,
        "teaching_note": "One of the oldest symbols encoding a mathematical concept. Self-reference, recursion, and cyclical causation all live in this image.",
    },
    
    # ── LINGUISTIC ──
    {
        "term_a": "color_words", "term_b": "color_perception",
        "domain_a": "linguistics", "domain_b": "psychology",
        "bridge_type": "linguistic",
        "reason": "Russian speakers (who have separate words for light blue 'goluboy' and dark blue 'siniy') discriminate those shades faster than English speakers (who just have 'blue'). Language doesn't determine what you CAN see, but affects how quickly you categorize it.",
        "strength": 0.8,
        "teaching_note": "Weak Sapir-Whorf: language influences (doesn't determine) perception. The color boundary your language names becomes a perceptual boundary in your brain.",
    },
    {
        "term_a": "spatial_language", "term_b": "navigation_ability",
        "domain_a": "linguistics", "domain_b": "geography",
        "bridge_type": "linguistic",
        "reason": "Guugu Yimithirr speakers (Australia) use cardinal directions instead of left/right. They maintain absolute spatial orientation at all times. Speakers of relative-direction languages get lost more easily. Grammar shapes navigation.",
        "strength": 0.85,
        "teaching_note": "Your language's spatial system literally changes how your brain represents space. This is one of the strongest documented cases of language shaping cognition.",
    },
    {
        "term_a": "future_tense", "term_b": "saving_behavior",
        "domain_a": "linguistics", "domain_b": "economics",
        "bridge_type": "linguistic",
        "reason": "Keith Chen (2013): speakers of languages without obligatory future tense (Mandarin, Finnish) save more money and make more future-oriented decisions than speakers of strong-future-tense languages (English, Greek). How your language marks the future changes how distant tomorrow feels.",
        "strength": 0.65,
        "teaching_note": "Controversial but fascinating. If your language forces you to grammatically separate present and future, the future feels more distant and you discount it more. Language shapes economics.",
    },
    
    # ── TECHNOLOGICAL ──
    {
        "term_a": "electromagnetic_theory", "term_b": "radio",
        "domain_a": "physics", "domain_b": "engineering",
        "bridge_type": "technological",
        "reason": "Maxwell's equations (1865) predicted electromagnetic waves. Hertz proved them (1887). Marconi built a radio (1895). Pure mathematical physics became global communication technology in 30 years. Theory → proof → tool.",
        "strength": 0.95,
        "teaching_note": "One of the clearest theory-to-technology pipelines in history. Maxwell never imagined radio — he was solving equations. The application emerged from the understanding.",
    },
    {
        "term_a": "quantum_tunneling", "term_b": "flash_memory",
        "domain_a": "physics", "domain_b": "engineering",
        "bridge_type": "technological",
        "reason": "Quantum tunneling (particles passing through barriers they classically shouldn't) is the principle behind flash memory storage. Every USB drive, SSD, and smartphone uses a quantum effect that Einstein found disturbing. Weird physics became mundane technology.",
        "strength": 0.85,
        "teaching_note": "Students carry quantum mechanics in their pockets. Flash memory works because electrons tunnel through an insulating oxide layer. Everyday technology depends on physics that seems impossible.",
    },
    {
        "term_a": "germ_theory", "term_b": "handwashing",
        "domain_a": "biology", "domain_b": "medicine",
        "bridge_type": "technological",
        "reason": "Semmelweis proposed handwashing in 1847 but was ridiculed — germ theory wasn't established yet. Pasteur and Koch proved it in the 1860s-80s. The practice (technology) was right before the science. Sometimes the bridge goes backwards.",
        "strength": 0.8,
        "teaching_note": "Rare case where the application preceded the theory. Semmelweis knew handwashing worked but couldn't explain why. Sometimes you find the bridge before you see what it connects.",
    },
    
    # ── EMOTIONAL ──
    {
        "term_a": "sadness", "term_b": "blue",
        "domain_a": "psychology", "domain_b": "color",
        "bridge_type": "emotional",
        "reason": "English maps sadness to blue ('feeling blue', 'the blues'). Not universal — in Chinese, sadness is often mapped to white or gray, in Russian to dark tones. The mapping isn't arbitrary though: blue light suppresses melatonin and affects mood circuitry. There may be a weak biological basis amplified by cultural reinforcement.",
        "strength": 0.7,
        "teaching_note": "The mapping feels inevitable in English but isn't universal. Blues music, 'feeling blue', 'blue Monday' — the link is cultural and linguistic, not physical, yet it's so embedded that it shapes how English speakers actually experience sadness.",
    },
    {
        "term_a": "emotional_weight", "term_b": "physical_weight",
        "domain_a": "psychology", "domain_b": "physics",
        "bridge_type": "emotional",
        "reason": "Heavy news, light-hearted comedy, the weight of responsibility, burdened with guilt. Emotional gravity maps onto physical gravity across many languages. Studies show people holding heavy clipboards rate moral transgressions as more serious. The metaphor bleeds into actual cognition.",
        "strength": 0.8,
        "teaching_note": "Lakoff and Johnson's 'embodied cognition' — abstract concepts are understood through bodily metaphor. This isn't decorative language, it's how the brain actually processes abstract meaning through sensorimotor circuits.",
    },
    {
        "term_a": "anger", "term_b": "heat",
        "domain_a": "psychology", "domain_b": "physics",
        "bridge_type": "emotional",
        "reason": "Hot-headed, boiling with rage, heated argument, cool down, simmering resentment. Near-universal across languages. Anger literally raises skin temperature and blood pressure — the metaphor maps to a real physiological response. The body IS hotter when angry.",
        "strength": 0.85,
        "teaching_note": "One of the most embodied emotional bridges. Unlike sadness=blue (mostly cultural), anger=heat has direct physiological grounding. The metaphor is literally true at the body level.",
    },
    {
        "term_a": "understanding", "term_b": "seeing",
        "domain_a": "cognition", "domain_b": "vision",
        "bridge_type": "emotional",
        "reason": "I see what you mean. Clear explanation. Illuminating idea. Brilliant insight. Dim understanding. Blind to the truth. Vision is the dominant metaphor for comprehension across most languages. We treat knowledge as light and ignorance as darkness.",
        "strength": 0.9,
        "teaching_note": "So deeply embedded that we don't notice it. 'Enlightenment' — the entire Western intellectual tradition named itself after this bridge. Also present in Sanskrit (vidya = seeing/knowing), suggesting deep cognitive roots.",
    },
    {
        "term_a": "sharpness", "term_b": "intelligence",
        "domain_a": "physical", "domain_b": "cognition",
        "bridge_type": "emotional",
        "reason": "Sharp mind, sharp wit, cutting remark, dull student, pointed question, blunt response, edgy humor. We map cognitive precision onto physical sharpness. A sharp person penetrates problems the way a blade penetrates material — fast, precise, at a point.",
        "strength": 0.75,
        "teaching_note": "The mapping is about precision and penetration. Sharp tools work at a focused point; sharp minds focus on the key issue. The abstract concept borrows the geometry of the physical one.",
    },
    {
        "term_a": "warmth", "term_b": "social_closeness",
        "domain_a": "physics", "domain_b": "psychology",
        "bridge_type": "emotional",
        "reason": "Warm personality, cold shoulder, warm welcome, icy stare, frosty reception. Holding a warm drink makes people rate strangers as more trustworthy (Williams & Bargh, 2008). Physical temperature literally affects social judgment. The bridge runs through the insula cortex, which processes both.",
        "strength": 0.85,
        "teaching_note": "One of the best-documented embodied cognition findings. Same brain region (insula) processes physical temperature and social warmth. The metaphor isn't just language — it's neuroscience.",
    },
    
    # ── PATTERN ──
    {
        "term_a": "spiral", "term_b": "growth",
        "domain_a": "geometry", "domain_b": "biology",
        "bridge_type": "pattern",
        "reason": "The spiral appears wherever growth happens with rotation: nautilus shells (biological growth + rotation), sunflower seeds (packing efficiency), hurricanes (thermal expansion + Coriolis), galaxies (gravitational rotation), DNA (molecular stacking + twist). The pattern IS the signature of growth-under-rotation.",
        "strength": 0.9,
        "teaching_note": "The spiral isn't a metaphor — it's a mathematical attractor. Anything that grows while turning will produce it. The pattern is the explanation.",
        "examples": ["Fibonacci spirals in sunflowers", "Golden spiral in nautilus", "Spiral arms in galaxies", "DNA double helix"],
    },
    {
        "term_a": "pulse", "term_b": "cycle",
        "domain_a": "biology", "domain_b": "physics",
        "bridge_type": "pattern",
        "reason": "Heartbeat, breath, circadian rhythm, tidal cycle, seasonal cycle, boom/bust economics, predator/prey oscillation, AC electricity, orbital period, stellar pulsation. The pulse pattern — regular oscillation between states — appears at every scale in every domain. It may be the most universal pattern in nature.",
        "strength": 0.95,
        "teaching_note": "Students already know this pattern from their own heartbeat. Everything pulses. The question is always: what are the two states, and what drives the oscillation between them?",
    },
    {
        "term_a": "branching", "term_b": "distribution",
        "domain_a": "geometry", "domain_b": "engineering",
        "bridge_type": "pattern",
        "reason": "Trees, rivers, blood vessels, lungs, lightning, road networks, organizational hierarchies, file directory structures. Anything that distributes something from one source to many destinations (or collects from many sources) converges on branching. The pattern solves the distribution problem regardless of scale or substrate.",
        "strength": 0.9,
        "teaching_note": "Ask students: what do tree roots, river deltas, and your lungs have in common? They're all solving the same geometry problem — efficient distribution with minimal material.",
    },
    {
        "term_a": "wave", "term_b": "information_propagation",
        "domain_a": "physics", "domain_b": "communication",
        "bridge_type": "pattern",
        "reason": "Ocean waves, sound waves, electromagnetic radiation, crowd waves in a stadium, rumor spreading through a population, fashion trends, epidemic curves. The wave pattern — disturbance propagating through a medium without the medium itself traveling — describes how information moves through any substrate.",
        "strength": 0.85,
        "teaching_note": "The wave is about propagation without transport. The water doesn't travel with an ocean wave. The air molecules don't travel with sound. The people don't travel in a stadium wave. The PATTERN travels. That's what information is.",
    },
    {
        "term_a": "symmetry_breaking", "term_b": "decision",
        "domain_a": "physics", "domain_b": "psychology",
        "bridge_type": "pattern",
        "reason": "A ball balanced on a hill peak is symmetric — it could fall any direction. The moment it falls, symmetry breaks. Phase transitions (water→ice), crystal formation, the Big Bang, choosing a career, a group splitting into factions — all involve a state of equal possibility collapsing into a specific outcome. The pattern of decision is symmetry breaking.",
        "strength": 0.8,
        "teaching_note": "Every decision — physical or human — is a symmetry breaking event. Before: many options equally available. After: one path taken. The mathematics of the transition is the same in all cases.",
    },
    {
        "term_a": "emergence", "term_b": "more_than_sum",
        "domain_a": "systems_theory", "domain_b": "philosophy",
        "bridge_type": "pattern",
        "reason": "Wetness isn't a property of individual water molecules. Consciousness isn't a property of individual neurons. Traffic jams aren't a property of individual cars. Markets aren't a property of individual traders. The pattern: simple rules at one level produce complex behavior at the next level that cannot be predicted from the components alone.",
        "strength": 0.95,
        "teaching_note": "This is arguably the master pattern. It's what makes reductionism incomplete — knowing everything about the parts doesn't tell you about the whole. It's also exactly what the Idea layer of this system is trying to detect.",
    },
    
    # ── PHILOSOPHICAL ──
    {
        "term_a": "map", "term_b": "territory",
        "domain_a": "philosophy", "domain_b": "epistemology",
        "bridge_type": "philosophical",
        "reason": "Korzybski: 'The map is not the territory.' Every model, theory, equation, and database is a map. Reality is the territory. All knowledge systems — including this one — are maps. The gap between map and territory is where all errors live.",
        "strength": 0.95,
        "teaching_note": "The most important philosophical bridge for any knowledge system. Our database of subject-predicate-object triples is a map. The actual relationships in reality are the territory. The map is useful precisely because it's simpler than the territory, but that simplification is also where it's wrong.",
    },
    {
        "term_a": "free_will", "term_b": "determinism",
        "domain_a": "philosophy", "domain_b": "physics",
        "bridge_type": "philosophical",
        "reason": "If physics determines all particle interactions, and brains are made of particles, do we have free will? Compatibilism argues both can be true — determinism at the physical level doesn't eliminate meaningful choice at the experiential level. Different levels of description, both valid.",
        "strength": 0.85,
        "teaching_note": "This is the contradictory bridge's philosophical cousin. The contradiction dissolves when you recognize that 'cause' means different things at different scales.",
    },
    {
        "term_a": "ship_of_theseus", "term_b": "identity",
        "domain_a": "philosophy", "domain_b": "biology",
        "bridge_type": "philosophical",
        "reason": "If you replace every plank of a ship, is it the same ship? Your body replaces most cells over 7-10 years. Are you the same person? The question applies to companies, nations, rivers, and software codebases. Identity is pattern continuity, not material continuity.",
        "strength": 0.9,
        "teaching_note": "Directly relevant to this knowledge system — if we replace every entry over time through verification and correction, is it the same knowledge base? Yes, because the structure persists. Identity IS the pattern.",
    },
    {
        "term_a": "is_ought_gap", "term_b": "science_and_policy",
        "domain_a": "philosophy", "domain_b": "politics",
        "bridge_type": "philosophical",
        "reason": "Hume's guillotine: you cannot derive 'ought' from 'is.' Science tells you the climate is warming. It cannot tell you what to do about it. Every time someone says 'science says we should...' they've jumped the is-ought gap. The bridge between fact and value is always a human choice, never a logical necessity.",
        "strength": 0.9,
        "teaching_note": "Critical for understanding why political disagreements persist despite scientific consensus. People can agree on all the facts and still disagree on what to do — because values aren't facts.",
    },
    {
        "term_a": "observer_effect", "term_b": "measurement_problem",
        "domain_a": "philosophy", "domain_b": "physics",
        "bridge_type": "philosophical",
        "reason": "In quantum mechanics, observation affects the system being observed. In social science, being studied changes behavior (Hawthorne effect). In journalism, coverage changes events. The philosophical question: can any system fully know itself? Gödel proved the answer is no for formal systems.",
        "strength": 0.85,
        "teaching_note": "Connects quantum physics, social science, and mathematical logic through one philosophical principle: observation is participation, not passive recording.",
    },
    
    # ── SENSORY ──
    {
        "term_a": "loud", "term_b": "bright",
        "domain_a": "hearing", "domain_b": "vision",
        "bridge_type": "sensory",
        "reason": "Loud colors. Bright sounds. High-pitched = light/small, low-pitched = dark/large. These cross-modal mappings are near-universal and appear in infants before language. The brain maps intensity across senses onto a shared scale. A loud shirt and a loud noise activate overlapping evaluation circuits.",
        "strength": 0.8,
        "teaching_note": "Synaesthesia (literally 'joined perception') is the extreme version, but everyone does it mildly. The bouba/kiki effect proves it: round shapes 'sound' soft, spiky shapes 'sound' sharp. Senses aren't as separate as we think.",
    },
    {
        "term_a": "high_pitch", "term_b": "smallness",
        "domain_a": "hearing", "domain_b": "spatial",
        "bridge_type": "sensory",
        "reason": "High pitch = small. Low pitch = big. Universal across languages and cultures. A mouse 'should' squeak (high, small), an elephant 'should' rumble (low, big). This isn't arbitrary — it's physics. Smaller vibrating objects produce higher frequencies. The sensory mapping reflects physical reality.",
        "strength": 0.85,
        "teaching_note": "The bridge has a real physical basis — vocal cord length determines pitch, so larger animals genuinely produce lower sounds. Our cross-modal intuition is calibrated to reality.",
    },
    {
        "term_a": "sweet", "term_b": "pleasant",
        "domain_a": "taste", "domain_b": "emotion",
        "bridge_type": "sensory",
        "reason": "Sweet person, sweet deal, sweet music, sweet victory. Bitterness of defeat, sour mood, salty language. Taste maps onto emotional valence across many languages. Sweet = positive, bitter/sour = negative. Likely rooted in evolution: sweet = ripe fruit = calories = good.",
        "strength": 0.75,
        "teaching_note": "The evolutionary basis makes this bridge deep. Our ancestors who found sweet things pleasant survived better. The taste-to-emotion mapping is millions of years old.",
    },
    {
        "term_a": "texture", "term_b": "personality",
        "domain_a": "touch", "domain_b": "psychology",
        "bridge_type": "sensory",
        "reason": "Rough day, smooth talker, abrasive personality, soft-spoken, hard-nosed, gritty determination, slippery character, prickly attitude. Touch vocabulary maps comprehensively onto personality. Studies show rough textures make people rate social interactions as more difficult.",
        "strength": 0.8,
        "teaching_note": "Like warmth→social_closeness, this is embodied cognition. The tactile system provides the vocabulary for abstract social evaluation. We literally 'feel out' a situation.",
    },
    {
        "term_a": "smell", "term_b": "memory",
        "domain_a": "olfaction", "domain_b": "neuroscience",
        "bridge_type": "sensory",
        "reason": "The olfactory bulb connects directly to the hippocampus (memory) and amygdala (emotion) — no thalamic relay, unlike all other senses. This is why a smell can instantly transport you to a childhood memory. Proust's madeleine is neuroscience, not just literature.",
        "strength": 0.9,
        "teaching_note": "The only sense with a direct neural pathway to memory formation. This is why smell triggers the most vivid, emotional memories. The bridge is anatomical, not metaphorical.",
    },
    
    # ── HUMOR ──
    {
        "term_a": "expectation", "term_b": "surprise",
        "domain_a": "psychology", "domain_b": "cognition",
        "bridge_type": "humor",
        "reason": "Most humor theories agree: comedy is set-up (create expectation) + punchline (violate it in a safe way). The joke 'I told my wife she was drawing her eyebrows too high. She looked surprised.' works because 'surprised' is both the emotion AND the physical result of high eyebrows. The brain gets two valid interpretations where it expected one.",
        "strength": 0.85,
        "teaching_note": "Humor reveals how the brain handles ambiguity. Every joke is a miniature test of the knowledge system — it creates an inference, then breaks it. What we find funny tells us about the structure of our expectations.",
    },
    {
        "term_a": "pun", "term_b": "semantic_ambiguity",
        "domain_a": "humor", "domain_b": "linguistics",
        "bridge_type": "humor",
        "reason": "Puns exploit words with multiple meanings. 'Time flies like an arrow; fruit flies like a banana.' The parser resolves 'flies' and 'like' differently in each half. Humor arises from the collision of two valid parse trees. This is literally a demonstration of linguistic ambiguity made entertaining.",
        "strength": 0.8,
        "teaching_note": "Puns are unintentional linguistics tutorials. Every pun proves that language is ambiguous and that context resolves meaning. The groan is the brain acknowledging it got tricked.",
    },
    {
        "term_a": "irony", "term_b": "theory_of_mind",
        "domain_a": "humor", "domain_b": "psychology",
        "bridge_type": "humor",
        "reason": "Understanding irony requires modeling what someone means versus what they said — which requires theory of mind (modeling another person's mental state). Children develop irony comprehension around age 6-8, exactly when theory of mind matures. Sarcasm is a test of cognitive development.",
        "strength": 0.8,
        "teaching_note": "If a knowledge system can't handle irony, it can't model intent. This is why AI chatbots fail at sarcasm — they process the literal meaning, not the intended meaning. The gap IS the joke.",
    },
    {
        "term_a": "absurdity", "term_b": "category_violation",
        "domain_a": "humor", "domain_b": "logic",
        "bridge_type": "humor",
        "reason": "Absurdist humor (Monty Python, Hitchhiker's Guide) works by violating categories. A dead parrot in a pet shop. The answer to everything is 42. A ministry of silly walks. The humor comes from applying formal institutional logic to things that don't deserve it, or applying nonsense to things that demand seriousness.",
        "strength": 0.75,
        "teaching_note": "Absurdity is the domain scope system breaking on purpose for entertainment. Our system blocks 'orange trains_via backpropagation.' Comedy keeps it.",
    },
    {
        "term_a": "tickling", "term_b": "self_other_distinction",
        "domain_a": "humor", "domain_b": "neuroscience",
        "bridge_type": "humor",
        "reason": "You cannot tickle yourself. The cerebellum predicts self-generated touch and cancels the response. Tickling requires unpredictability, which requires another agent. This is one of the most basic demonstrations that the brain models self versus other — and that humor/surprise requires input you didn't generate.",
        "strength": 0.7,
        "teaching_note": "The inability to tickle yourself is a hardware demo of the prediction engine. Your brain cancels expected stimuli. Humor, like tickling, requires the unexpected.",
    },
]


# ═══════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════

def display_bridges(bridges_list):
    """Pretty print a list of bridges."""
    if not bridges_list:
        print("  No bridges found")
        return
    
    type_icons = {
        "historical": "📜",
        "inspirational": "💡",
        "structural": "🔷",
        "metaphorical": "🎭",
        "emergent": "✨",
        "causal": "⚡",
        "cultural": "🌍",
        "etymological": "📖",
        "scale": "🔬",
        "pedagogical": "🎓",
        "contradictory": "⚔️",
        "experiential": "🖐️",
        "spiritual": "🕉️",
        "religious": "⛪",
        "ethical": "⚖️",
        "aesthetic": "🎨",
        "rumor": "💭",
        "mythological": "🐉",
        "linguistic": "🗣️",
        "technological": "🔧",
        "emotional": "💙",
        "pattern": "🌀",
        "philosophical": "🏛️",
        "sensory": "👁️",
        "humor": "😄",
    }
    
    for b in bridges_list:
        icon = type_icons.get(b.get("bridge_type", ""), "🔗")
        strength = b.get("strength", 0.5)
        bar = "█" * int(strength * 10) + "░" * (10 - int(strength * 10))
        
        print(f"\n  {icon} {b['term_a']} ↔ {b['term_b']}")
        print(f"    Type: {b.get('bridge_type', '?'):15s} Strength: {bar} {strength:.1f}")
        if b.get("domain_a") or b.get("domain_b"):
            print(f"    Domains: {b.get('domain_a', '?')} ↔ {b.get('domain_b', '?')}")
        print(f"    Why: {b.get('reason', 'no reason given')}")
        if b.get("teaching_note"):
            print(f"    📝 {b['teaching_note']}")


def display_candidates(candidates):
    """Display bridge candidates from discovery."""
    if not candidates:
        print("  No candidates found")
        return
    
    for c in candidates:
        conf = c["confidence"]
        bar = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))
        print(f"\n  ? {c['term_a']} ↔ {c['term_b']}")
        print(f"    Shared: {c['shared_property']}")
        print(f"    Confidence: {bar} {conf:.1f}")
        print(f"    Suggested type: {c['suggested_type']}")
        print(f"    Reason: {c['reason']}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Knowledge Bridge System")
    parser.add_argument("--db", default="curriculum.db")
    parser.add_argument("--show", action="store_true", help="Show all bridges")
    parser.add_argument("--find", help="Find bridges involving a term")
    parser.add_argument("--between", nargs=2, help="Bridges between two domains")
    parser.add_argument("--type", help="Show bridges of a specific type")
    parser.add_argument("--analyze", action="store_true", help="Find potential bridges in KB")
    parser.add_argument("--seed", action="store_true", help="Load seed bridges")
    parser.add_argument("--stats", action="store_true", help="Bridge statistics")
    args = parser.parse_args()
    
    print(f"{'═'*55}")
    print(f"  🌉 KNOWLEDGE BRIDGE SYSTEM")
    print(f"{'═'*55}")
    
    bs = BridgeSystem(args.db)
    
    if args.seed:
        print(f"\n  Loading seed bridges...")
        for bridge in SEED_BRIDGES:
            result = bs.add_bridge(**bridge)
            if result:
                print(f"    + {bridge['term_a']} ↔ {bridge['term_b']} ({bridge['bridge_type']})")
            else:
                print(f"    ○ {bridge['term_a']} ↔ {bridge['term_b']} (already exists)")
        print(f"\n  Total bridges: {bs.count()}")
    
    elif args.show:
        bridges = bs.all_bridges()
        print(f"\n  All bridges ({len(bridges)}):")
        display_bridges(bridges)
    
    elif args.find:
        bridges = bs.find_bridges(args.find)
        print(f"\n  Bridges involving '{args.find}' ({len(bridges)}):")
        display_bridges(bridges)
    
    elif args.between:
        bridges = bs.bridges_between_domains(args.between[0], args.between[1])
        print(f"\n  Bridges between {args.between[0]} ↔ {args.between[1]} ({len(bridges)}):")
        display_bridges(bridges)
    
    elif args.type:
        bridges = bs.bridges_by_type(args.type)
        print(f"\n  Bridges of type '{args.type}' ({len(bridges)}):")
        display_bridges(bridges)
    
    elif args.analyze:
        print(f"\n  Scanning knowledge base for potential bridges...")
        discovery = BridgeDiscovery(args.db)
        candidates = discovery.find_potential_bridges()
        print(f"\n  Found {len(candidates)} candidates:")
        display_candidates(candidates[:20])
        if len(candidates) > 20:
            print(f"\n  ... +{len(candidates)-20} more")
    
    elif args.stats:
        total = bs.count()
        print(f"\n  Total bridges: {total}")
        if total > 0:
            for bt in BridgeSystem.VALID_TYPES:
                bridges = bs.bridges_by_type(bt)
                if bridges:
                    print(f"    {bt:15s}: {len(bridges)}")
    
    else:
        print(f"\n  Use --show, --find, --between, --analyze, --seed, or --stats")
        print(f"  Example: python3 bridges.py --seed          (load famous bridges)")
        print(f"           python3 bridges.py --show           (display all)")
        print(f"           python3 bridges.py --find gravity   (find bridges for term)")
        print(f"           python3 bridges.py --analyze        (discover new bridges)")

if __name__ == "__main__":
    main()
