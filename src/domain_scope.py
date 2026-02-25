#!/usr/bin/env python3
"""
DOMAIN AWARENESS & SCOPE CONTROL
═════════════════════════════════

Prevents the knowledge base from generating garbage entries when
mixing knowledge from different domains. Without this, pouring in
ConceptNet + Wikidata + textbooks produces things like:
  "orange and neural_network both have layers"
  "cake is structurally similar to sedimentary_rock" (both have layers)
  "dog → uses → backpropagation" (transitive chain gone wrong)

THREE MECHANISMS:
  1. Type Gates    — block invalid transitive chains
  2. Domain Affinity — control which domains can cross-pollinate  
  3. Nonsense Filter — catch weird entries before they're committed

USAGE:
  # Import and wrap your database:
  from domain_scope import ScopedDB
  
  sdb = ScopedDB("curriculum.db")
  sdb.add(subject, predicate, object, ...)  # checks before adding
  
  # Or run standalone to analyze/fix existing database:
  python3 domain_scope.py --analyze          # show domain health
  python3 domain_scope.py --clean            # remove nonsense entries
  python3 domain_scope.py --build            # build all scope tables
"""

import sqlite3
import json
import re
from collections import defaultdict


# ═══════════════════════════════════════════════════════════
# DOMAIN DEFINITIONS
# ═══════════════════════════════════════════════════════════
# Each domain has:
#   - keywords: terms that indicate membership
#   - root_types: high-level categories (from hierarchy)
#   - compatible: which other domains it can cross-pollinate with
# ═══════════════════════════════════════════════════════════

DOMAINS = {
    "mathematics": {
        "keywords": ["number", "equation", "function", "variable", "theorem",
                     "proof", "calculus", "algebra", "geometry", "fraction",
                     "decimal", "derivative", "integral", "matrix", "vector",
                     "probability", "statistics", "trigonometry", "logarithm",
                     "polynomial", "quadratic", "linear", "graph", "set",
                     "ratio", "proportion", "prime", "factorial", "infinity",
                     "dimension", "topology", "axiom", "conjecture"],
        "root_types": ["mathematical_concept", "number", "operation",
                       "geometric_shape", "mathematical_object"],
        "compatible": ["physics", "computer_science", "engineering",
                       "chemistry", "economics", "philosophy"],
    },
    "physics": {
        "keywords": ["force", "energy", "mass", "velocity", "acceleration",
                     "gravity", "momentum", "wave", "frequency", "photon",
                     "electron", "quantum", "relativity", "thermodynamic",
                     "entropy", "magnetic", "electric", "nuclear", "particle",
                     "newton", "kinetic", "potential", "pressure", "density",
                     "friction", "torque", "wavelength", "amplitude",
                     "oscillation", "field", "radiation", "plasma"],
        "root_types": ["physical_quantity", "physical_law", "physical_force",
                       "particle", "wave", "field"],
        "compatible": ["mathematics", "chemistry", "engineering",
                       "astronomy", "earth_science", "computer_science"],
    },
    "chemistry": {
        "keywords": ["atom", "molecule", "element", "compound", "bond",
                     "reaction", "acid", "base", "ion", "orbital",
                     "periodic_table", "chemical", "ph", "oxidation",
                     "catalyst", "solution", "mole", "isotope", "valence",
                     "polymer", "enzyme", "protein", "organic", "inorganic",
                     "solvent", "precipitate", "titration", "redox"],
        "root_types": ["chemical_element", "chemical_compound", "chemical_reaction",
                       "chemical_bond", "molecule"],
        "compatible": ["physics", "biology", "earth_science",
                       "medicine", "engineering", "materials"],
    },
    "biology": {
        "keywords": ["cell", "dna", "rna", "protein", "gene", "organism",
                     "species", "evolution", "photosynthesis", "mitosis",
                     "meiosis", "enzyme", "bacteria", "virus", "ecology",
                     "habitat", "chromosome", "membrane", "organ", "tissue",
                     "vertebrate", "invertebrate", "fungi", "plant",
                     "animal", "metabolism", "respiration", "taxonomy",
                     "biodiversity", "mutation", "allele", "phenotype"],
        "root_types": ["living_thing", "organism", "cell", "biological_process",
                       "organ", "tissue", "species"],
        "compatible": ["chemistry", "medicine", "earth_science",
                       "ecology", "psychology", "agriculture"],
    },
    "computer_science": {
        "keywords": ["algorithm", "software", "hardware", "processor",
                     "neural_network", "transformer", "machine_learning",
                     "artificial_intelligence", "programming", "database",
                     "encryption", "compiler", "binary", "recursion",
                     "data_structure", "complexity", "optimization",
                     "llm", "gpu", "cpu", "memory", "cache", "token",
                     "embedding", "attention", "backpropagation", "gradient",
                     "training", "inference", "model", "parameter",
                     "deep_learning", "reinforcement_learning", "diffusion",
                     "gan", "vae", "rag", "mamba", "ssm", "bert", "gpt"],
        "root_types": ["algorithm", "data_structure", "programming_language",
                       "neural_network_architecture", "ml_paradigm",
                       "software", "hardware"],
        "compatible": ["mathematics", "physics", "engineering",
                       "linguistics", "philosophy", "economics"],
    },
    "earth_science": {
        "keywords": ["rock", "mineral", "volcano", "earthquake", "tectonic",
                     "atmosphere", "weather", "climate", "erosion",
                     "sediment", "fossil", "glacier", "ocean", "continent",
                     "crust", "mantle", "core", "magma", "lava",
                     "precipitation", "evaporation", "humidity",
                     "water_cycle", "groundwater", "soil"],
        "root_types": ["rock_type", "geological_process", "weather_phenomenon",
                       "landform", "mineral"],
        "compatible": ["physics", "chemistry", "biology",
                       "geography", "ecology"],
    },
    "geography": {
        "keywords": ["continent", "country", "city", "ocean", "mountain",
                     "river", "climate", "latitude", "longitude",
                     "population", "capital", "border", "island",
                     "desert", "forest", "hemisphere", "equator",
                     "arctic", "tropical", "tundra", "savanna"],
        "root_types": ["country", "city", "continent", "landform",
                       "body_of_water", "biome"],
        "compatible": ["earth_science", "history", "economics",
                       "ecology", "biology"],
    },
    "history": {
        "keywords": ["war", "revolution", "empire", "dynasty", "treaty",
                     "constitution", "civilization", "colonial", "medieval",
                     "ancient", "renaissance", "industrial", "monarchy",
                     "democracy", "republic", "pharaoh", "emperor",
                     "independence", "amendment", "abolition", "suffrage"],
        "root_types": ["historical_event", "civilization", "war",
                       "political_system", "historical_period"],
        "compatible": ["geography", "philosophy", "economics",
                       "literature", "politics", "religion"],
    },
    "everyday": {
        "keywords": ["food", "animal", "color", "shape", "body", "sense",
                     "season", "weather", "family", "house", "car",
                     "clothing", "tool", "game", "sport", "music",
                     "cooking", "fruit", "vegetable", "pet", "toy",
                     "furniture", "job", "money", "store", "school"],
        "root_types": ["food", "animal", "color", "shape", "tool",
                       "body_part", "clothing", "furniture"],
        "compatible": ["biology", "geography", "economics"],
    },
    "philosophy": {
        "keywords": ["epistemology", "ethics", "logic", "metaphysics",
                     "ontology", "consciousness", "free_will", "morality",
                     "truth", "knowledge", "belief", "reason", "argument",
                     "fallacy", "syllogism", "axiom", "premise",
                     "deduction", "induction", "empiricism", "rationalism"],
        "root_types": ["philosophical_concept", "ethical_theory",
                       "epistemological_theory", "logical_system"],
        "compatible": ["mathematics", "computer_science", "history",
                       "psychology", "religion", "linguistics"],
    },
    # ── Extended domains (referenced in compatibility lists) ──
    "engineering": {
        "keywords": ["bridge", "circuit", "design", "engine", "material",
                     "structure", "mechanical", "electrical", "civil",
                     "manufacturing", "construction", "hydraulic"],
        "root_types": ["engineering_discipline", "structure", "machine"],
        "compatible": ["physics", "mathematics", "chemistry", "materials",
                       "computer_science"],
    },
    "astronomy": {
        "keywords": ["star", "planet", "galaxy", "orbit", "comet",
                     "asteroid", "nebula", "black_hole", "supernova",
                     "telescope", "constellation", "solar", "lunar",
                     "cosmic", "light_year", "red_shift"],
        "root_types": ["celestial_body", "astronomical_phenomenon"],
        "compatible": ["physics", "mathematics", "earth_science"],
    },
    "medicine": {
        "keywords": ["disease", "symptom", "diagnosis", "treatment",
                     "vaccine", "surgery", "anatomy", "pathology",
                     "pharmaceutical", "therapy", "infection", "immune",
                     "blood", "heart", "lung", "brain", "nerve"],
        "root_types": ["disease", "treatment", "organ", "body_system"],
        "compatible": ["biology", "chemistry", "psychology"],
    },
    "ecology": {
        "keywords": ["ecosystem", "food_chain", "predator", "prey",
                     "symbiosis", "niche", "biome", "conservation",
                     "endangered", "pollution", "deforestation",
                     "carbon_cycle", "nitrogen_cycle", "biodiversity"],
        "root_types": ["ecosystem", "ecological_process", "biome"],
        "compatible": ["biology", "earth_science", "geography", "chemistry"],
    },
    "psychology": {
        "keywords": ["cognition", "emotion", "behavior", "perception",
                     "memory", "learning", "motivation", "personality",
                     "disorder", "therapy", "stimulus", "response",
                     "conditioning", "neuroscience", "anxiety", "depression"],
        "root_types": ["psychological_concept", "mental_process", "disorder"],
        "compatible": ["biology", "philosophy", "medicine", "linguistics"],
    },
    "economics": {
        "keywords": ["market", "supply", "demand", "inflation", "gdp",
                     "trade", "currency", "investment", "labor", "tax",
                     "monopoly", "recession", "fiscal", "monetary",
                     "capital", "commodity", "scarcity"],
        "root_types": ["economic_concept", "market_type", "policy"],
        "compatible": ["mathematics", "history", "geography", "politics"],
    },
    "agriculture": {
        "keywords": ["crop", "harvest", "soil", "irrigation", "fertilizer",
                     "livestock", "farming", "seed", "plantation",
                     "pesticide", "drought", "yield", "organic_farming"],
        "root_types": ["crop", "farming_practice", "agricultural_tool"],
        "compatible": ["biology", "ecology", "chemistry", "economics"],
    },
    "linguistics": {
        "keywords": ["phoneme", "morpheme", "syntax", "semantics",
                     "pragmatics", "phonology", "lexicon", "dialect",
                     "bilingual", "translation", "grammar", "vowel",
                     "consonant", "tense", "clause", "phrase"],
        "root_types": ["linguistic_unit", "language_family", "grammatical_concept"],
        "compatible": ["philosophy", "computer_science", "psychology", "history"],
    },
    "literature": {
        "keywords": ["novel", "poem", "narrative", "metaphor", "allegory",
                     "protagonist", "genre", "satire", "fiction",
                     "mythology", "folklore", "sonnet", "stanza"],
        "root_types": ["literary_work", "genre", "literary_device"],
        "compatible": ["history", "philosophy", "linguistics"],
    },
    "politics": {
        "keywords": ["government", "policy", "election", "democracy",
                     "legislature", "judiciary", "executive", "sovereignty",
                     "diplomacy", "ideology", "constitution", "parliament",
                     "republic", "authoritarian", "federalism"],
        "root_types": ["political_system", "political_concept", "institution"],
        "compatible": ["history", "economics", "philosophy", "geography"],
    },
    "religion": {
        "keywords": ["theology", "scripture", "ritual", "prayer",
                     "salvation", "deity", "prophet", "worship",
                     "monastery", "pilgrimage", "sacred", "doctrine",
                     "baptism", "meditation", "afterlife"],
        "root_types": ["religion", "religious_practice", "theological_concept"],
        "compatible": ["history", "philosophy", "literature", "psychology"],
    },
    "materials": {
        "keywords": ["alloy", "ceramic", "polymer", "composite",
                     "crystal", "tensile", "ductile", "brittle",
                     "conductivity", "semiconductor", "superconductor",
                     "corrosion", "hardness", "elasticity"],
        "root_types": ["material_type", "material_property"],
        "compatible": ["chemistry", "physics", "engineering"],
    },
}

# Pre-build compatibility matrix
DOMAIN_COMPAT = {}
for d, info in DOMAINS.items():
    DOMAIN_COMPAT[d] = set(info["compatible"]) | {d}  # always compatible with self


# ═══════════════════════════════════════════════════════════
# DOMAIN CLASSIFIER
# ═══════════════════════════════════════════════════════════

class DomainClassifier:
    """Determines what domain(s) a term belongs to."""
    
    def __init__(self, conn=None):
        self._cache = {}
        self._keyword_to_domain = {}
        
        # Build reverse keyword lookup
        for domain, info in DOMAINS.items():
            for kw in info["keywords"]:
                self._keyword_to_domain[kw] = domain
        
        # Baseline vocabulary — common terms that should always be classified
        # even before the database has broad data.
        # This prevents "orange trains_via backpropagation" type nonsense.
        self._baseline = {}
        for term, domain in self._BASELINE_TERMS.items():
            self._baseline[term] = {domain} if isinstance(domain, str) else set(domain)
        
        # If we have a database, also use hierarchy
        self.conn = conn
        self._hier_cache = {}
        if conn:
            self._load_hierarchy_domains()
    
    # Common terms with known domains — acts as a safety net
    # before the database has enough data to classify on its own
    _BASELINE_TERMS = {
        # Everyday / food
        "orange": "everyday", "apple": "everyday", "banana": "everyday",
        "cake": "everyday", "bread": "everyday", "pizza": "everyday",
        "chair": "everyday", "table": "everyday", "car": "everyday",
        "house": "everyday", "book": "everyday", "phone": "everyday",
        "shoe": "everyday", "shirt": "everyday", "hat": "everyday",
        "water": "chemistry", "fire": "everyday", "ice": "everyday",
        "food": "everyday", "fruit": "everyday", "vegetable": "everyday",
        "meat": "everyday", "milk": "everyday", "egg": "everyday",
        "tree": "biology", "flower": "biology", "grass": "biology",
        "rock": "earth_science", "sand": "earth_science",
        
        # Animals
        "dog": "biology", "cat": "biology", "fish": "biology",
        "bird": "biology", "horse": "biology", "cow": "biology",
        "snake": "biology", "frog": "biology", "whale": "biology",
        "elephant": "biology", "lion": "biology", "tiger": "biology",
        "mouse": "biology", "rabbit": "biology", "bear": "biology",
        "insect": "biology", "spider": "biology", "ant": "biology",
        "bee": "biology", "butterfly": "biology",
        
        # Countries / geography
        "france": "geography", "germany": "geography", "japan": "geography",
        "china": "geography", "india": "geography", "brazil": "geography",
        "australia": "geography", "canada": "geography", "mexico": "geography",
        "russia": "geography", "egypt": "geography", "italy": "geography",
        "spain": "geography", "england": "geography", "africa": "geography",
        "europe": "geography", "asia": "geography", "america": "geography",
        "paris": "geography", "london": "geography", "tokyo": "geography",
        "new_york": "geography", "pacific": "geography", "atlantic": "geography",
        "mountain": "geography", "river": "geography", "lake": "geography",
        
        # People roles (for history/biography)
        "king": "history", "queen": "history", "president": "history",
        "soldier": "history", "farmer": "everyday",
        
        # Body parts
        "heart": "biology", "brain": "biology", "lung": "biology",
        "liver": "biology", "bone": "biology", "muscle": "biology",
        "eye": "biology", "ear": "biology", "hand": "biology",
        "skin": "biology", "blood": "biology", "stomach": "biology",
        
        # Celestial
        "sun": "physics", "moon": "physics", "star": "physics",
        "planet": "physics", "mars": "physics", "venus": "physics",
        "jupiter": "physics", "saturn": "physics",
        "earth": ["physics", "earth_science", "geography"],
        "galaxy": "physics", "universe": "physics", "comet": "physics",
        
        # Elements
        "hydrogen": "chemistry", "oxygen": "chemistry", "carbon": "chemistry",
        "nitrogen": "chemistry", "iron": "chemistry", "gold": "chemistry",
        "silver": "chemistry", "copper": "chemistry", "helium": "chemistry",
        "uranium": "chemistry", "sodium": "chemistry", "calcium": "chemistry",
    }
    
    def _load_hierarchy_domains(self):
        """Use hierarchy to propagate domain labels upward."""
        try:
            cur = self.conn.cursor()
            # Get domain assignments from entry_domains table if it exists
            cur.execute("""
                SELECT DISTINCT e.subject, ed.domain 
                FROM entries e 
                JOIN entry_domains ed ON e.id = ed.entry_id
                WHERE e.subject NOT LIKE '%,%'
            """)
            for row in cur.fetchall():
                if row[0] not in self._hier_cache:
                    self._hier_cache[row[0]] = set()
                self._hier_cache[row[0]].add(row[1])
        except sqlite3.OperationalError:
            pass  # hierarchy table may not exist yet

    def classify(self, term):
        """Return set of domains a term belongs to."""
        if term in self._cache:
            return self._cache[term]
        
        domains = set()
        term_lower = term.lower().replace(" ", "_")
        
        # Baseline vocabulary (always checked first)
        if term_lower in self._baseline:
            domains |= self._baseline[term_lower]
        
        # Direct keyword match
        if term_lower in self._keyword_to_domain:
            domains.add(self._keyword_to_domain[term_lower])
        
        # Component match (catches compound terms like "neural_network_layer")
        # Split on underscores to avoid false hits ("ion" in "animation")
        term_parts = set(term_lower.split("_"))
        for kw, domain in self._keyword_to_domain.items():
            kw_parts = set(kw.split("_"))
            # Match if any keyword component is a full word in the term (min 4 chars)
            if kw_parts & term_parts and all(len(p) >= 4 for p in kw_parts & term_parts):
                domains.add(domain)
        
        # Hierarchy-based classification
        if term_lower in self._hier_cache:
            domains |= self._hier_cache[term_lower]
        
        # Cache and return
        self._cache[term] = domains
        return domains
    
    def are_compatible(self, term_a, term_b):
        """Check if two terms come from compatible domains."""
        domains_a = self.classify(term_a)
        domains_b = self.classify(term_b)
        
        # If either term is unclassified, allow it (don't block unknowns)
        if not domains_a or not domains_b:
            return True
        
        # Check if any domain of A is compatible with any domain of B
        for da in domains_a:
            for db in domains_b:
                if db in DOMAIN_COMPAT.get(da, set()):
                    return True
        
        return False
    
    def compatibility_score(self, term_a, term_b):
        """
        Return 0.0-1.0 score of how compatible two terms are.
        1.0 = same domain
        0.7 = compatible domains
        0.3 = distant but not blocked
        0.0 = incompatible
        """
        domains_a = self.classify(term_a)
        domains_b = self.classify(term_b)
        
        if not domains_a or not domains_b:
            return 0.5  # unknown = neutral
        
        # Same domain?
        overlap = domains_a & domains_b
        if overlap:
            return 1.0
        
        # Compatible domains?
        for da in domains_a:
            for db in domains_b:
                if db in DOMAIN_COMPAT.get(da, set()):
                    return 0.7
        
        # Distant but share a compatible neighbor?
        for da in domains_a:
            for db in domains_b:
                neighbors_a = DOMAIN_COMPAT.get(da, set())
                neighbors_b = DOMAIN_COMPAT.get(db, set())
                if neighbors_a & neighbors_b:
                    return 0.3
        
        return 0.0


# ═══════════════════════════════════════════════════════════
# TYPE GATES
# ═══════════════════════════════════════════════════════════
# Controls which transitive chains are valid.
# Uses hierarchy root types to prevent nonsense.
# ═══════════════════════════════════════════════════════════

class TypeGate:
    """
    Validates whether a proposed inference makes sense.
    
    Example valid chain:
      transformer → uses → attention → enables → parallel_processing
      (all in computer_science, types are compatible)
    
    Example blocked chain:
      orange → has → layers → has → weights
      (orange is food, weights is math/CS — gate closed)
    """
    
    def __init__(self, conn):
        self.conn = conn
        self.classifier = DomainClassifier(conn)
        self._predicate_profiles = {}
        self._build_predicate_profiles()
    
    def _build_predicate_profiles(self):
        """
        Learn what types of subjects typically use each predicate.
        If "trains_via" is only ever used by CS subjects, then
        "orange trains_via backpropagation" is suspicious.
        """
        try:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT predicate, subject FROM entries 
                WHERE truth_value='T' AND subject NOT LIKE '%,%'
            """)
            
            pred_domains = defaultdict(lambda: defaultdict(int))
            for row in cur.fetchall():
                pred, subj = row
                domains = self.classifier.classify(subj)
                for d in domains:
                    pred_domains[pred][d] += 1
            
            self._predicate_profiles = dict(pred_domains)
        except (sqlite3.OperationalError, KeyError, AttributeError):
            pass
    
    def check_transitive(self, subject, predicate, obj, via=None):
        """
        Should this transitive inference be allowed?
        Returns: (allowed: bool, reason: str, confidence: float)
        """
        compat = self.classifier.compatibility_score(subject, obj)
        
        # Hard block: completely incompatible domains
        if compat == 0.0:
            subj_domains = self.classifier.classify(subject)
            obj_domains = self.classifier.classify(obj)
            return False, f"incompatible domains: {subj_domains} vs {obj_domains}", 0.0
        
        # Check if this predicate is typically used in this domain
        # Only block if there's STRONG evidence (50+ uses) that this predicate
        # belongs elsewhere and ZERO uses in subject's domain.
        if predicate in self._predicate_profiles:
            subj_domains = self.classifier.classify(subject)
            pred_profile = self._predicate_profiles[predicate]
            total_uses = sum(pred_profile.values())
            
            if subj_domains and total_uses >= 50:
                subj_domain_uses = sum(pred_profile.get(d, 0) for d in subj_domains)
                if subj_domain_uses == 0:
                    return False, f"predicate '{predicate}' never used in {subj_domains} ({total_uses} uses elsewhere)", 0.1
        
        # Soft gate: allow but with reduced confidence
        if compat < 0.5:
            return True, f"distant domains (score={compat:.1f}), needs verification", compat
        
        return True, "compatible", compat
    
    def check_analogy(self, subject_a, subject_b):
        """
        Should these two subjects be compared as analogous?
        Blocks nonsensical analogies like "cake ~ sedimentary_rock"
        """
        compat = self.classifier.compatibility_score(subject_a, subject_b)
        
        if compat == 0.0:
            return False, "incompatible domains"
        
        if compat < 0.3:
            return False, "too distant for meaningful analogy"
        
        return True, f"compatible (score={compat:.1f})"
    
    def check_cross_domain(self, subject_a, subject_b, predicate, obj):
        """
        Should this cross-domain pattern be recorded?
        "transformer and mamba both use attention" = yes
        "orange and neural_net both have layers" = no
        """
        # Both subjects must be compatible with each other
        compat_ab = self.classifier.compatibility_score(subject_a, subject_b)
        if compat_ab == 0.0:
            return False, "subjects from incompatible domains"
        
        # The shared property must make sense for both subjects
        compat_a = self.classifier.compatibility_score(subject_a, obj)
        compat_b = self.classifier.compatibility_score(subject_b, obj)
        
        if compat_a == 0.0 or compat_b == 0.0:
            return False, "shared property not meaningful for both subjects"
        
        # Average compatibility must be reasonable
        avg = (compat_ab + compat_a + compat_b) / 3
        if avg < 0.3:
            return False, f"low overall compatibility ({avg:.1f})"
        
        return True, f"compatible ({avg:.1f})"


# ═══════════════════════════════════════════════════════════
# NONSENSE FILTER
# ═══════════════════════════════════════════════════════════
# Last line of defense. Quick heuristics to catch garbage.
# ═══════════════════════════════════════════════════════════

class NonsenseFilter:
    """
    Fast heuristic checks for obviously bad entries.
    Runs before committing anything to the database.
    """
    
    # Predicates that only make sense between specific types
    TYPED_PREDICATES = {
        # Biology / living things
        "eats":           {"subject": ["biology", "everyday"], "object": ["biology", "everyday"]},
        "lives_in":       {"subject": ["biology", "everyday"], "object": ["geography", "everyday", "earth_science"]},
        "breathes":       {"subject": ["biology", "everyday"], "object": ["chemistry", "biology"]},
        "reproduces_via": {"subject": ["biology"],             "object": ["biology"]},
        "has_organ":      {"subject": ["biology"],             "object": ["biology"]},
        "migrates_to":    {"subject": ["biology"],             "object": ["geography"]},
        
        # Geography / political
        "has_capital":    {"subject": ["geography"],            "object": ["geography"]},
        "borders":        {"subject": ["geography"],            "object": ["geography"]},
        "has_population": {"subject": ["geography"],            "object": ["mathematics", "geography"]},
        "elected_in":     {"subject": ["history", "geography"], "object": ["history"]},
        "governed_by":    {"subject": ["geography", "history"], "object": ["history", "geography"]},
        
        # Chemistry
        "reacts_with":    {"subject": ["chemistry"],           "object": ["chemistry"]},
        "dissolves_in":   {"subject": ["chemistry"],           "object": ["chemistry"]},
        "has_ph":         {"subject": ["chemistry"],           "object": ["chemistry", "mathematics"]},
        "oxidizes":       {"subject": ["chemistry"],           "object": ["chemistry"]},
        
        # Physics / astronomy
        "orbits":         {"subject": ["physics", "earth_science"], "object": ["physics", "earth_science"]},
        "has_wavelength":  {"subject": ["physics"],            "object": ["physics", "mathematics"]},
        "has_mass":       {"subject": ["physics", "chemistry", "everyday"], "object": ["physics", "mathematics"]},
        
        # Computer science
        "trains_via":     {"subject": ["computer_science"],    "object": ["computer_science", "mathematics"]},
        "compiles_to":    {"subject": ["computer_science"],    "object": ["computer_science"]},
        "runs_on":        {"subject": ["computer_science"],    "object": ["computer_science"]},
        "has_complexity":  {"subject": ["computer_science", "mathematics"], "object": ["computer_science", "mathematics"]},
        "tokenizes":      {"subject": ["computer_science"],    "object": ["computer_science"]},
        
        # Everyday / sensory
        "tastes_like":    {"subject": ["everyday", "biology"], "object": ["everyday"]},
        "smells_like":    {"subject": ["everyday", "biology"], "object": ["everyday"]},
        "sounds_like":    {"subject": ["everyday", "physics"], "object": ["everyday"]},
        
        # General but typed
        "discovered_by":  {"subject": ["*"],                   "object": ["history"]},
    }
    
    # Known incompatible domain pairs — these should NEVER cross
    HARD_BLOCKS = {
        frozenset({"everyday", "computer_science"}),   # food ↔ code
        frozenset({"everyday", "mathematics"}),         # food ↔ abstract math
        frozenset({"geography", "computer_science"}),   # countries ↔ code
        frozenset({"geography", "chemistry"}),          # countries ↔ molecules
    }
    
    def __init__(self, classifier):
        self.classifier = classifier
        self.blocked_count = 0
        self.allowed_count = 0
    
    def check(self, subject, predicate, obj):
        """
        Returns (allowed: bool, reason: str)
        """
        # Rule 1: Self-reference
        if subject == obj:
            self.blocked_count += 1
            return False, "self-referential"
        
        # Rule 2: Empty or too short
        if not subject or not predicate or not obj:
            self.blocked_count += 1
            return False, "empty field"
        
        if len(subject) < 2 or len(obj) < 2:
            self.blocked_count += 1
            return False, "term too short"
        
        # Rule 3: Typed predicate check
        if predicate in self.TYPED_PREDICATES:
            constraints = self.TYPED_PREDICATES[predicate]
            
            subj_domains = self.classifier.classify(subject)
            obj_domains = self.classifier.classify(obj)
            
            # Check subject type (if we know the domain)
            if subj_domains and constraints.get("subject"):
                allowed_subj = set(constraints["subject"])
                if "*" not in allowed_subj and not (subj_domains & allowed_subj):
                    self.blocked_count += 1
                    return False, f"'{predicate}' invalid for subject domain {subj_domains}"
            
            # Check object type
            if obj_domains and constraints.get("object"):
                allowed_obj = set(constraints["object"])
                if "*" not in allowed_obj and not (obj_domains & allowed_obj):
                    self.blocked_count += 1
                    return False, f"'{predicate}' invalid for object domain {obj_domains}"
        
        # Rule 4: Hard domain blocks (even for untyped predicates)
        subj_domains = self.classifier.classify(subject)
        obj_domains = self.classifier.classify(obj)
        if subj_domains and obj_domains:
            for ds in subj_domains:
                for do in obj_domains:
                    pair = frozenset({ds, do})
                    if pair in self.HARD_BLOCKS:
                        self.blocked_count += 1
                        return False, f"hard block: {ds} ↔ {do}"
        
        # Rule 5: Suspiciously long compound terms (usually parsing errors)
        if len(subject) > 100 or len(obj) > 100:
            self.blocked_count += 1
            return False, "term exceeds 100 chars"
        
        self.allowed_count += 1
        return True, "ok"
    
    def stats(self):
        total = self.blocked_count + self.allowed_count
        if total == 0:
            return "No entries checked"
        pct = self.blocked_count / total * 100
        return f"Checked {total}: {self.allowed_count} allowed, {self.blocked_count} blocked ({pct:.1f}%)"


# ═══════════════════════════════════════════════════════════
# SCOPED DATABASE WRAPPER
# ═══════════════════════════════════════════════════════════
# Wraps KnowledgeDB and applies all scope checks before
# any entry is added. Drop-in replacement.
# ═══════════════════════════════════════════════════════════

class ScopedDB:
    """
    Drop-in wrapper around the knowledge database that applies
    domain awareness and nonsense filtering to all additions.
    
    Usage:
        # Instead of:
        db.add(subject, predicate, object, ...)
        
        # Use:
        sdb = ScopedDB("curriculum.db")
        sdb.add(subject, predicate, object, ...)
        # Now it checks domains before adding!
    """
    
    def __init__(self, db_path="curriculum.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        
        self.classifier = DomainClassifier(self.conn)
        self.type_gate = TypeGate(self.conn)
        self.nonsense_filter = NonsenseFilter(self.classifier)
        
        # Track what gets blocked and why
        self.blocked_log = []
    
    def add(self, subject, predicate, obj, truth_value="T",
            evidence_for=None, source="unknown", generation=0,
            grade_level=0, needs_verification=False,
            # Scope control options:
            skip_scope_check=False,
            scope_source=None):
        """
        Add an entry with scope checking.
        
        skip_scope_check: bypass all filters (for seed data)
        scope_source: what operation is trying to add this
                      ("transitive", "cross_domain", "analogy", "seed", etc.)
        """
        # Normalize
        subject = self._normalize(subject)
        predicate = self._normalize(predicate)
        obj = self._normalize(obj)
        
        if not skip_scope_check:
            # Nonsense filter (fast, always runs)
            allowed, reason = self.nonsense_filter.check(subject, predicate, obj)
            if not allowed:
                self._log_block(subject, predicate, obj, reason, scope_source)
                return None
            
            # Type gate (for inferred entries)
            if scope_source == "transitive":
                allowed, reason, conf = self.type_gate.check_transitive(
                    subject, predicate, obj)
                if not allowed:
                    self._log_block(subject, predicate, obj, reason, scope_source)
                    return None
                # Low confidence → mark as M
                if conf < 0.5:
                    truth_value = "M"
                    needs_verification = True
            
            elif scope_source == "analogy":
                # For analogies, check the two subjects being compared
                # (subject is term_a, obj is term_b in analogy entries)
                allowed, reason = self.type_gate.check_analogy(subject, obj)
                if not allowed:
                    self._log_block(subject, predicate, obj, reason, scope_source)
                    return None
            
            elif scope_source == "cross_domain":
                # Cross-domain entries have compound subjects like "a,b"
                if "," in subject:
                    parts = subject.split(",")
                    if len(parts) == 2:
                        allowed, reason = self.type_gate.check_cross_domain(
                            parts[0], parts[1], predicate, obj)
                        if not allowed:
                            self._log_block(subject, predicate, obj, reason, scope_source)
                            return None
        
        # Check for duplicate
        self.cursor.execute(
            "SELECT id FROM entries WHERE subject=? AND predicate=? AND object=?",
            (subject, predicate, obj))
        if self.cursor.fetchone():
            return None
        
        # Add it
        self.cursor.execute("""
            INSERT INTO entries (subject, predicate, object, truth_value,
                evidence_for, evidence_against, source, generation,
                grade_level, needs_verification)
            VALUES (?, ?, ?, ?, ?, '[]', ?, ?, ?, ?)
        """, (subject, predicate, obj, truth_value,
              json.dumps(evidence_for or []),
              source, generation, grade_level, int(needs_verification)))
        self.conn.commit()
        return self.cursor.lastrowid
    
    def _normalize(self, text):
        if not text: return ""
        text = text.strip().lower()
        text = re.sub(r'[^a-z0-9_\s-]', '', text)
        text = re.sub(r'[\s-]+', '_', text)
        return text.strip('_')[:100]
    
    def _log_block(self, subject, predicate, obj, reason, source):
        self.blocked_log.append({
            "subject": subject, "predicate": predicate, "object": obj,
            "reason": reason, "source": source
        })
    
    def blocked_report(self):
        """Show what got blocked and why."""
        if not self.blocked_log:
            print("  No entries blocked")
            return
        
        by_reason = defaultdict(int)
        for entry in self.blocked_log:
            by_reason[entry["reason"]] += 1
        
        print(f"\n  Blocked entries: {len(self.blocked_log)}")
        print(f"  By reason:")
        for reason, count in sorted(by_reason.items(), key=lambda x: -x[1]):
            print(f"    {count:4d} — {reason}")
        
        # Show some examples
        print(f"\n  Examples:")
        for entry in self.blocked_log[:10]:
            print(f"    ✗ {entry['subject']} → {entry['predicate']}({entry['object']})")
            print(f"      reason: {entry['reason']} | from: {entry['source']}")


# ═══════════════════════════════════════════════════════════
# STANDALONE ANALYSIS
# ═══════════════════════════════════════════════════════════

def analyze_database(db_path):
    """Analyze existing database for domain health."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    classifier = DomainClassifier(conn)
    
    print(f"\n  {'═'*55}")
    print(f"  DOMAIN ANALYSIS")
    print(f"  {'═'*55}")
    
    # Classify all subjects
    cur.execute("SELECT DISTINCT subject FROM entries WHERE subject NOT LIKE '%,%'")
    subjects = [r[0] for r in cur.fetchall()]
    
    domain_subjects = defaultdict(list)
    unclassified = []
    multi_domain = []
    
    for subj in subjects:
        domains = classifier.classify(subj)
        if not domains:
            unclassified.append(subj)
        elif len(domains) > 1:
            multi_domain.append((subj, domains))
        for d in domains:
            domain_subjects[d].append(subj)
    
    print(f"\n  Subjects: {len(subjects)} total")
    print(f"\n  By domain:")
    for domain in sorted(domain_subjects, key=lambda d: -len(domain_subjects[d])):
        bar = "█" * min(len(domain_subjects[domain]), 30)
        print(f"    {domain:20s} {len(domain_subjects[domain]):4d} {bar}")
    
    if unclassified:
        print(f"\n  Unclassified ({len(unclassified)}):")
        for s in unclassified[:20]:
            print(f"    ? {s}")
        if len(unclassified) > 20:
            print(f"    ... +{len(unclassified)-20} more")
    
    if multi_domain:
        print(f"\n  Multi-domain ({len(multi_domain)}):")
        for s, domains in multi_domain[:10]:
            print(f"    {s:30s} → {', '.join(sorted(domains))}")
    
    # Check for suspicious cross-domain entries
    print(f"\n  ── Suspicious Entries ──")
    cur.execute("""
        SELECT * FROM entries 
        WHERE truth_value='T' AND subject NOT LIKE '%,%'
        AND source LIKE 'idea:%'
    """)
    
    suspicious = []
    gate = TypeGate(conn)
    
    for row in cur.fetchall():
        entry = dict(row)
        compat = classifier.compatibility_score(entry["subject"], entry["object"])
        if compat < 0.3 and compat > 0.0:
            suspicious.append((entry, compat))
    
    if suspicious:
        print(f"  Found {len(suspicious)} low-compatibility inferred entries:")
        for entry, score in suspicious[:15]:
            print(f"    [{score:.1f}] {entry['subject']} → {entry['predicate']}({entry['object']})")
            print(f"         source: {entry['source']}")
    else:
        print(f"  No suspicious entries found")
    
    conn.close()

def clean_database(db_path):
    """Remove nonsense entries from existing database."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    classifier = DomainClassifier(conn)
    nsf = NonsenseFilter(classifier)
    
    cur.execute("SELECT * FROM entries")
    entries = [dict(sqlite3.Row(cur, r)) for r in cur.fetchall()]
    
    removed = 0
    for entry in entries:
        allowed, reason = nsf.check(entry["subject"], entry["predicate"], entry["object"])
        if not allowed:
            cur.execute("DELETE FROM entries WHERE id=?", (entry["id"],))
            removed += 1
    
    conn.commit()
    conn.close()
    print(f"  Removed {removed} nonsense entries")
    print(f"  Filter stats: {nsf.stats()}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Domain Scope Control")
    parser.add_argument("--db", default="curriculum.db")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--build", action="store_true", help="Build scope tables")
    parser.add_argument("--test", action="store_true", help="Run test cases")
    args = parser.parse_args()
    
    print(f"{'═'*55}")
    print(f"  🔬 DOMAIN SCOPE CONTROL")
    print(f"{'═'*55}")
    
    if args.analyze:
        analyze_database(args.db)
    elif args.clean:
        clean_database(args.db)
    elif args.test:
        # Run test cases to show what gets blocked
        print(f"\n  Running test cases...")
        conn = sqlite3.connect(args.db)
        classifier = DomainClassifier(conn)
        gate = TypeGate(conn)
        nsf = NonsenseFilter(classifier)
        
        tests = [
            # (subject, predicate, object, expected_result, description)
            ("transformer", "uses", "attention", True, "same domain — should pass"),
            ("transformer", "is_a", "neural_network", True, "same domain — should pass"),
            ("dog", "eats", "bone", True, "same domain (everyday) — should pass"),
            ("photosynthesis", "produces", "glucose", True, "same domain (biology) — should pass"),
            ("gravity", "causes", "acceleration", True, "same domain (physics) — should pass"),
            ("orange", "trains_via", "backpropagation", False, "food+CS — should BLOCK"),
            ("cake", "compiles_to", "assembly", False, "food+CS — should BLOCK"),
            ("dog", "orbits", "sun", False, "biology+astronomy — should BLOCK"),
            ("france", "reacts_with", "oxygen", False, "geography+chemistry — should BLOCK"),
            ("neural_network", "eats", "gradient", False, "CS using bio predicate — should BLOCK"),
        ]
        
        print(f"\n  {'Subject':20s} {'Pred':15s} {'Object':15s} {'Expected':8s} Result")
        print(f"  {'─'*80}")
        
        for subj, pred, obj, expected, desc in tests:
            allowed_ns, reason_ns = nsf.check(subj, pred, obj)
            
            if allowed_ns:
                allowed_tg, reason_tg, conf = gate.check_transitive(subj, pred, obj)
                actual = allowed_tg
                reason = reason_tg
            else:
                actual = False
                reason = reason_ns
            
            match = "✓" if actual == expected else "✗ WRONG"
            symbol = "pass" if actual else "BLOCK"
            
            print(f"  {subj:20s} {pred:15s} {obj:15s} {'pass' if expected else 'block':8s} {symbol:5s} {match}")
            if actual != expected:
                print(f"    reason: {reason}")
        
        conn.close()
    else:
        print(f"  Use --analyze, --clean, --build, or --test")
        print(f"  Example: python3 domain_scope.py --analyze")

if __name__ == "__main__":
    main()
