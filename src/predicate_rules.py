#!/usr/bin/env python3
"""
PREDICATE TRANSPARENCY
═══════════════════════

Controls which transitive chains are valid based on predicate types.

THE PROBLEM:
  baptism → uses → water → boils_at → 100°C
  ∴ baptism boils_at 100°C  ← GARBAGE

  ice → is_a → solid_water → is_a → crystal
  ∴ ice is_a crystal  ← CORRECT

THE RULE:
  "is_a" is TRANSPARENT — properties flow through it.
  "uses" is OPAQUE — properties DON'T flow through it.
  
  But even through opaque links, SOME properties make sense:
    baptism uses water, water essential_for life
    → "baptism relates to something essential for life" = reasonable
    
  So: opaque links block INTRINSIC properties but allow RELATIONAL ones.

USAGE:
  from predicate_rules import TransitiveFilter
  
  tf = TransitiveFilter()
  
  # Check if a chain is valid:
  allowed, reason = tf.check_chain(
      link_predicate="uses",        # how A connects to B
      inherited_predicate="boils_at" # B's property we'd inherit
  )
  # → False, "opaque link blocks intrinsic property"
"""


class TransitiveFilter:
    """
    Determines whether a transitive chain should propagate.
    
    A chain has two predicates:
      A --[link]--> B --[inherited]--> C
      
    The question: should A inherit C through B?
    Answer depends on both predicates.
    """
    
    # ── Link Predicates ──────────────────────────────────
    # How A connects to B. Determines whether B's properties
    # flow through to A.
    
    # TRANSPARENT: A "is" B in some sense. Properties pass through.
    # If dog is_a mammal and mammal is warm_blooded, dog is warm_blooded.
    TRANSPARENT_LINKS = {
        "is_a", "subclass_of", "instance_of", "is_type_of",
        "type_of", "kind_of", "classified_as",
        "is_mostly",       # "blood is_mostly water" → blood inherits some water properties
    }
    
    # OPAQUE: A relates to B but ISN'T B. Properties don't pass.
    # If baptism uses water and water boils, baptism doesn't boil.
    OPAQUE_LINKS = {
        "uses", "employs", "utilizes", "relies_on",
        "causes", "caused_by", "leads_to", "results_in",
        "enables", "prevents", "blocks",
        "created_by", "invented_by", "built_by",
        "symbolizes", "represents", "associated_with",
        "located_in", "found_in",
        "part_of",
        "produces", "generates", "outputs",
        "preceded_by", "followed_by",
        "similar_to", "analogous_to",
        "compares", "says", "proposed",
        "absorbs", "regulates", "converts",
        "predecessor_of", "successor_of",
    }
    
    # SEMI-TRANSPARENT: some properties pass, some don't.
    # "has_part" is interesting — if a car has_part engine and engine uses gasoline,
    # it's reasonable to say car uses gasoline. But if engine weighs 200kg,
    # car doesn't weigh 200kg.
    SEMI_TRANSPARENT_LINKS = {
        "has_part", "contains", "includes",
        "made_of", "composed_of",
    }
    
    # ── Inherited Predicates ─────────────────────────────
    # What property of B we're trying to give to A.
    
    # INTRINSIC: physical/chemical properties inherent to the thing itself.
    # These should NOT pass through opaque links.
    INTRINSIC_PROPERTIES = {
        "has_formula", "boils_at", "freezes_at", "melts_at",
        "has_state", "has_density", "has_mass", "has_weight",
        "has_color", "has_ph", "has_temperature",
        "has_property",  # generic physical property
        "has_depth", "has_height", "has_length", "has_width",
        "covers",        # "water covers 71% of earth" shouldn't propagate
        "has_population", "has_area",
        "weighs", "measures", "costs",
        "tastes_like", "smells_like", "looks_like",
        "has_wavelength", "has_frequency",
    }
    
    # RELATIONAL: how something connects to other things.
    # These CAN pass through opaque links (with reduced confidence).
    RELATIONAL_PROPERTIES = {
        "is_a", "subclass_of",
        "essential_for", "used_for", "enables", "prevents",
        "causes", "caused_by",
        "found_in", "located_in",
        "related_to", "connected_to",
    }
    
    # IDENTITY: what something IS at its core.
    # Only passes through transparent AND semi-transparent links.
    IDENTITY_PROPERTIES = {
        "has_formula", "is_a", "defined_as",
        "has_part", "part_of", "made_of",
    }
    
    def __init__(self):
        # Pre-compute sets for fast lookup
        self._all_opaque = self.OPAQUE_LINKS | self.SEMI_TRANSPARENT_LINKS
        self._all_transparent = self.TRANSPARENT_LINKS
    
    def check_chain(self, link_predicate, inherited_predicate):
        """
        Should this transitive chain be allowed?
        
        Chain: A --[link_predicate]--> B --[inherited_predicate]--> C
        Question: Should we create A --[inherited_predicate]--> C?
        
        Returns: (allowed: bool, reason: str)
        """
        link = link_predicate.lower()
        inherited = inherited_predicate.lower()
        
        # Rule 1: Transparent link — almost everything passes
        if link in self._all_transparent:
            return True, "transparent link"
        
        # Rule 2: Semi-transparent + identity — ALLOW
        # Must come BEFORE opaque+intrinsic check because semi-transparent
        # links are a subset of opaque, but identity properties (has_formula,
        # is_a) should pass through made_of/composed_of even though they're
        # intrinsic. "ice made_of water, water has_formula H2O" → ice IS H2O.
        if link in self.SEMI_TRANSPARENT_LINKS and inherited in self.IDENTITY_PROPERTIES:
            return True, f"semi-transparent '{link}' passes identity '{inherited}'"
        
        # Rule 3: Opaque link + intrinsic property — BLOCK
        if link in self._all_opaque and inherited in self.INTRINSIC_PROPERTIES:
            return False, f"opaque link '{link}' blocks intrinsic '{inherited}'"
        
        # Rule 4: Opaque link + relational property — ALLOW with note
        if link in self._all_opaque and inherited in self.RELATIONAL_PROPERTIES:
            return True, f"opaque link '{link}' allows relational '{inherited}'"
        
        # Rule 5: Profile/meta predicates — skip (not real knowledge)
        if inherited in ("profile_size",) or link in ("profile_size",):
            return False, "meta-predicate, skip"
        
        # Rule 6: Unknown predicates — default based on link type
        if link in self._all_opaque:
            # Unknown property through opaque link:
            # Allow but flag for verification
            return True, f"opaque link '{link}' + unknown '{inherited}' — needs verification"
        
        # Default: allow (don't block things we don't understand)
        return True, "default allow"
    
    def check_chain_confidence(self, link_predicate, inherited_predicate):
        """
        Returns (allowed, reason, confidence) for the chain.
        Confidence ranges 0.0-1.0.
        """
        allowed, reason = self.check_chain(link_predicate, inherited_predicate)
        
        link = link_predicate.lower()
        inherited = inherited_predicate.lower()
        
        if not allowed:
            return False, reason, 0.0
        
        # Transparent + anything = high confidence
        if link in self._all_transparent:
            return True, reason, 0.9
        
        # Semi-transparent = medium
        if link in self.SEMI_TRANSPARENT_LINKS:
            return True, reason, 0.6
        
        # Opaque + relational = lower
        if link in self._all_opaque and inherited in self.RELATIONAL_PROPERTIES:
            return True, reason, 0.4
        
        # Unknown = low
        return True, reason, 0.3


# ═══════════════════════════════════════════════════════════
# PREDICATE RULE LEARNER
# ═══════════════════════════════════════════════════════════

class PredicateRuleLearner:
    """
    Auto-learns predicate transparency rules from LLM verification outcomes.

    When a transitive entry (A -[link]-> B -[inherited]-> C) is verified
    True or False by the LLM, we record the (link, inherited) predicate pair
    and which way it went. Over time, patterns emerge:

      - If "uses + boils_at" consistently gets rejected → mark opaque+intrinsic
      - If "has_part + uses" consistently gets approved → mark semi-transparent+relational

    Usage:
        learner = PredicateRuleLearner(db)
        learner.record_outcome("uses", "boils_at", approved=False)
        learner.record_outcome("uses", "boils_at", approved=False)
        ...
        new_rules = learner.suggest_rules(min_samples=3, threshold=0.75)
        learner.apply_rules(transitive_filter, new_rules)
    """

    def __init__(self, db):
        self.db = db
        self._ensure_table()

    def _ensure_table(self):
        """Create the predicate_pair_stats table if missing."""
        self.db.cursor.executescript("""
            CREATE TABLE IF NOT EXISTS predicate_pair_stats (
                link_pred TEXT NOT NULL,
                inherited_pred TEXT NOT NULL,
                approved_count INTEGER DEFAULT 0,
                rejected_count INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (link_pred, inherited_pred)
            );
        """)
        self.db.conn.commit()

    def record_outcome(self, link_pred, inherited_pred, approved):
        """Record a verification outcome for a predicate pair."""
        link = link_pred.lower()
        inherited = inherited_pred.lower()
        col = "approved_count" if approved else "rejected_count"

        self.db.cursor.execute(f"""
            INSERT INTO predicate_pair_stats (link_pred, inherited_pred, {col})
            VALUES (?, ?, 1)
            ON CONFLICT(link_pred, inherited_pred) DO UPDATE SET
                {col} = {col} + 1,
                last_updated = CURRENT_TIMESTAMP
        """, (link, inherited))
        self.db.conn.commit()

    def get_stats(self):
        """Return all collected stats as list of dicts."""
        self.db.cursor.execute(
            "SELECT * FROM predicate_pair_stats ORDER BY approved_count + rejected_count DESC")
        return [dict(r) for r in self.db.cursor.fetchall()]

    def suggest_rules(self, min_samples=5, threshold=0.75):
        """
        Analyze verification patterns and suggest new transparency rules.

        Args:
            min_samples: Minimum total verifications before we trust the pattern
            threshold: Minimum rejection or approval rate to trigger a rule

        Returns:
            dict with keys:
                'block': [(link, inherited, reject_rate, total)] — pairs to block
                'allow': [(link, inherited, approve_rate, total)] — pairs to explicitly allow
                'uncertain': [(link, inherited, approve_rate, total)] — not enough data or mixed
        """
        stats = self.get_stats()
        block = []
        allow = []
        uncertain = []

        for s in stats:
            total = s["approved_count"] + s["rejected_count"]
            if total < min_samples:
                if total > 0:
                    approve_rate = s["approved_count"] / total
                    uncertain.append((s["link_pred"], s["inherited_pred"], approve_rate, total))
                continue

            approve_rate = s["approved_count"] / total
            reject_rate = s["rejected_count"] / total

            if reject_rate >= threshold:
                block.append((s["link_pred"], s["inherited_pred"], reject_rate, total))
            elif approve_rate >= threshold:
                allow.append((s["link_pred"], s["inherited_pred"], approve_rate, total))
            else:
                uncertain.append((s["link_pred"], s["inherited_pred"], approve_rate, total))

        return {"block": block, "allow": allow, "uncertain": uncertain}

    def apply_rules(self, tf, suggestions=None):
        """
        Apply learned rules to a TransitiveFilter instance.

        For blocking: adds the inherited predicate to INTRINSIC if the link
        is already opaque, or adds the link to OPAQUE if unknown.

        For allowing: adds the inherited predicate to RELATIONAL if the link
        is opaque, or adds the link to SEMI_TRANSPARENT if unknown.

        Returns count of rules applied.
        """
        if suggestions is None:
            suggestions = self.suggest_rules()

        applied = 0

        for link, inherited, rate, total in suggestions["block"]:
            # If we're currently allowing this pair, we need to block it
            currently_allowed, _ = tf.check_chain(link, inherited)
            if currently_allowed:
                # Add inherited to intrinsic set so it gets blocked through opaque links
                if link in tf.OPAQUE_LINKS or link in tf.SEMI_TRANSPARENT_LINKS:
                    if inherited not in tf.INTRINSIC_PROPERTIES:
                        tf.INTRINSIC_PROPERTIES.add(inherited)
                        applied += 1
                else:
                    # Unknown link type — add it to opaque
                    if link not in tf.OPAQUE_LINKS:
                        tf.OPAQUE_LINKS.add(link)
                        tf._all_opaque.add(link)
                        applied += 1
                    if inherited not in tf.INTRINSIC_PROPERTIES:
                        tf.INTRINSIC_PROPERTIES.add(inherited)
                        applied += 1

        for link, inherited, rate, total in suggestions["allow"]:
            currently_allowed, _ = tf.check_chain(link, inherited)
            if not currently_allowed:
                # Currently blocked — add inherited to relational so it passes through
                if inherited not in tf.RELATIONAL_PROPERTIES:
                    tf.RELATIONAL_PROPERTIES.add(inherited)
                    applied += 1

        return applied

    def report(self):
        """Print a human-readable report of learned patterns."""
        suggestions = self.suggest_rules()
        stats = self.get_stats()

        if not stats:
            print("  No predicate pair data collected yet.")
            return

        print(f"\n  ═══ PREDICATE RULE LEARNING REPORT ═══")
        print(f"  Total predicate pairs tracked: {len(stats)}")

        total_verifications = sum(s["approved_count"] + s["rejected_count"] for s in stats)
        print(f"  Total verifications recorded: {total_verifications}")

        if suggestions["block"]:
            print(f"\n  SUGGESTED BLOCKS ({len(suggestions['block'])} pairs):")
            for link, inherited, rate, total in suggestions["block"]:
                print(f"    {link} + {inherited}: {rate:.0%} rejected ({total} samples)")

        if suggestions["allow"]:
            print(f"\n  SUGGESTED ALLOWS ({len(suggestions['allow'])} pairs):")
            for link, inherited, rate, total in suggestions["allow"]:
                print(f"    {link} + {inherited}: {rate:.0%} approved ({total} samples)")

        if suggestions["uncertain"]:
            print(f"\n  UNCERTAIN ({len(suggestions['uncertain'])} pairs):")
            for link, inherited, rate, total in suggestions["uncertain"]:
                print(f"    {link} + {inherited}: {rate:.0%} approved ({total} samples, need more)")


def parse_transitive_evidence(evidence_str):
    """
    Extract (link_pred, inherited_pred) from transitive evidence string.

    Supports two formats:
        New: "transitive[uses→boils_at]: water→compound→100C"
        Old: "transitive: water→compound→100C"  (returns None — no pred info)

    Returns (link_pred, inherited_pred) or None if not parseable.
    """
    import re
    # New format: transitive[link→inherited]: ...
    m = re.match(r'transitive\[(\w+)→(\w+)\]:', evidence_str)
    if m:
        return m.group(1), m.group(2)
    return None


def run_tests():
    """Test suite for predicate transparency rules."""
    tf = TransitiveFilter()
    
    tests = [
        # (link, inherited, expected_allow, description)
        # Transparent links — should all pass
        ("is_a", "is_a", True,        "is_a chain: dog→mammal→animal"),
        ("is_a", "has_property", True, "inherit property through is_a"),
        ("is_a", "has_formula", True,  "inherit formula through is_a"),
        ("is_a", "enables", True,     "inherit capability through is_a"),
        
        # Opaque + intrinsic — should all BLOCK
        ("uses", "boils_at", False,    "baptism uses water, water boils → NO"),
        ("uses", "has_formula", False, "baptism uses water, water=H2O → NO"),
        ("uses", "has_state", False,   "baptism uses water, water is liquid → NO"),
        ("uses", "freezes_at", False,  "engine uses water, water freezes → NO"),
        ("uses", "has_property", False,"hydroelectric uses water, water=solvent → NO"),
        ("uses", "covers", False,      "baptism uses water, water covers earth → NO"),
        ("causes", "has_depth", False, "earthquake causes tsunami, tsunami deep → NO"),
        ("symbolizes", "boils_at", False, "baptism symbolizes purification, purif boils → NO"),
        
        # Opaque + relational — should ALLOW
        ("uses", "essential_for", True,  "baptism uses water, water essential for life → YES"),
        ("uses", "is_a", True,           "engine uses steam, steam is_a gas → YES (engine uses a gas)"),
        ("causes", "enables", True,      "heat causes evap, evap enables rain → YES"),
        ("causes", "caused_by", True,    "erosion causes canyon, canyon caused_by river → YES"),
        
        # Semi-transparent
        ("has_part", "is_a", True,       "water_cycle has_part evaporation, evap is_a process → YES"),
        ("made_of", "has_formula", True, "ice made_of water, water=H2O → YES"),
        
        # Edge cases
        ("predecessor_of", "has_state", False, "waterwheel predecessor hydroelectric, hydro is liquid → NO"),
        ("associated_with", "boils_at", False, "tears associated sadness, sadness boils → NO"),
    ]
    
    print(f"\n  {'Link pred':20s} {'Inherited pred':20s} {'Expect':6s} {'Got':6s} {'✓/✗'}")
    print(f"  {'─'*75}")
    
    passed = 0
    failed = 0
    for link, inherited, expected, desc in tests:
        allowed, reason = tf.check_chain(link, inherited)
        match = allowed == expected
        
        if match:
            passed += 1
        else:
            failed += 1
        
        symbol = "✓" if match else "✗"
        got = "pass" if allowed else "BLOCK"
        exp = "pass" if expected else "BLOCK"
        
        print(f"  {link:20s} {inherited:20s} {exp:6s} {got:6s} {symbol} {desc}")
        if not match:
            print(f"    reason: {reason}")
    
    print(f"\n  Results: {passed}/{passed+failed} passed")
    return failed == 0


if __name__ == "__main__":
    print(f"{'═'*60}")
    print(f"  🔗 PREDICATE TRANSPARENCY RULES")
    print(f"{'═'*60}")
    
    success = run_tests()
    
    if success:
        print(f"\n  ✓ All tests passed!")
    else:
        print(f"\n  ✗ Some tests failed")
