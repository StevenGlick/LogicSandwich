"""
SHARED LLM MODULE
=================

Supports two backends:
  - OllamaLLM  — local model via Ollama (free, offline, GPU-bound)
  - ClaudeLLM  — Anthropic API (fast, high quality, requires API key)

Both share the same interface via LLMBase. All higher-level methods
(verify_entry, generate_facts, etc.) live in the base class and call
ask() which each backend implements.

Usage:
    from llm import create_llm
    llm = create_llm("claude", model="claude-sonnet-4-6")
    llm = create_llm("ollama", model="llama3.2")
    # or the old way still works:
    from llm import OllamaLLM
    llm = OllamaLLM("llama3.2")
"""

import json
import os
import re

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ═══════════════════════════════════════════════════════════
# BASE CLASS — shared prompt logic + response parsing
# ═══════════════════════════════════════════════════════════

class LLMBase:
    """
    Base class for all LLM backends.
    Subclasses must implement ask(prompt, system, temperature).
    Everything else is shared.
    """

    def __init__(self, model, backend_name):
        self.model = model
        self.backend_name = backend_name
        self.available = False
        self.call_count = 0

    def ask(self, prompt, system=None, temperature=0.3):
        """Core chat interface. Returns response text or None. Override in subclass."""
        raise NotImplementedError

    # ── Verification ──────────────────────────────────────

    def verify_entry(self, subject, predicate, obj, evidence):
        """Ask LLM to verify a single claim. Returns (verdict, reason, new_facts)."""
        system = """You evaluate knowledge claims. Respond with EXACTLY this format, no markdown:

VERDICT: T
REASON: Water does boil at 100 celsius at standard pressure.

Or if the claim is wrong:

VERDICT: F
REASON: RAG is a retrieval technique, not a failure mode.

Rules:
- VERDICT must be T (true), F (false), or M (genuinely uncertain)
- Default to T for well-known facts. Default to F for clear nonsense.
- Only use M if the claim is genuinely debatable.
- Optionally add NEW_FACTS on separate lines: subject|predicate|object|T"""

        # Format evidence cleanly
        if isinstance(evidence, list):
            ev_str = "; ".join(str(e) for e in evidence)
        else:
            ev_str = str(evidence)

        prompt = f"Is this true? {subject} {predicate} {obj}\nEvidence: {ev_str}"
        response = self.ask(prompt, system)
        if not response:
            return "M", "unavailable", []

        verdict = "M"
        reason = ""
        new_facts = []
        for line in response.strip().split("\n"):
            # Strip markdown formatting
            clean = line.strip().strip("*_`#")
            upper = clean.upper()
            if upper.startswith("VERDICT"):
                # Handle "VERDICT: T", "VERDICT: True", "**VERDICT:** T", etc.
                after = clean.split(":", 1)[1].strip().upper() if ":" in clean else ""
                if after.startswith("T"):
                    verdict = "T"
                elif after.startswith("F"):
                    verdict = "F"
                elif after.startswith("M"):
                    verdict = "M"
            elif upper.startswith("REASON"):
                reason = clean.split(":", 1)[1].strip() if ":" in clean else ""
            elif "|" in line and line.count("|") >= 2:
                parts = [p.strip().lower().replace(" ", "_") for p in line.split("|")]
                if len(parts) >= 3 and parts[0] and parts[1] and parts[2]:
                    tv = parts[3].upper()[:1] if len(parts) > 3 and parts[3].strip().upper()[:1] in "TFM" else "T"
                    new_facts.append({
                        "subject": parts[0], "predicate": parts[1],
                        "object": parts[2], "truth_value": tv
                    })

        return verdict, reason, new_facts

    # ── Curriculum ────────────────────────────────────────

    def generate_facts(self, topic, grade_level, existing_subjects=None, curriculum=None):
        """Generate atomic facts about a topic at a grade level."""
        if curriculum:
            grade_desc = curriculum.get(grade_level, {}).get("name", f"Grade {grade_level}")
        else:
            grade_desc = f"Grade {grade_level}"

        context = ""
        if existing_subjects:
            sample = sorted(existing_subjects)[:30]
            context = f"\nThe student already knows about: {', '.join(sample)}"

        system = f"""You are building a knowledge base for a student at the {grade_desc} level.
Generate 15-25 atomic facts about the topic. Each fact is ONE relationship.

Output format -- one fact per line, no numbering, no bullets:
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
        if not response:
            return []

        facts = []
        for line in response.strip().split("\n"):
            line = line.strip().strip("-*0123456789. ")
            if "|" not in line:
                continue
            parts = [p.strip().lower().replace(" ", "_") for p in line.split("|")]
            if len(parts) >= 3 and parts[0] and parts[1] and parts[2]:
                facts.append({
                    "subject": parts[0], "predicate": parts[1], "object": parts[2]
                })
        return facts

    def ask_question(self, question, known_facts=None):
        """Ask a question, get back (facts, confusions, reasoning)."""
        context = ""
        if known_facts:
            fact_lines = [
                f"  {f['subject']} -> {f['predicate']}({f['object']})"
                for f in known_facts[:20]
            ]
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
        if not response:
            return [], [], ""

        facts = []
        confusions = []
        reasoning = ""
        section = None
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()
                section = "reasoning"
                continue
            elif line.startswith("FACTS:"):
                section = "facts"
                continue
            elif line.startswith("CONFUSED:"):
                section = "confused"
                line = line.split(":", 1)[1].strip()

            if section == "reasoning" and not line.startswith(("FACTS", "CONFUSED")):
                reasoning += " " + line

            if "|" in line:
                parts = [p.strip().lower().replace(" ", "_") for p in line.split("|")]
                if section == "confused" and len(parts) >= 3:
                    confusions.append({
                        "a": parts[0], "b": parts[1],
                        "reason": parts[2] if len(parts) > 2 else ""
                    })
                elif len(parts) >= 3:
                    tv = parts[3].upper() if len(parts) > 3 and parts[3].upper() in ("T", "F", "M") else "T"
                    facts.append({
                        "subject": parts[0], "predicate": parts[1],
                        "object": parts[2], "truth_value": tv
                    })

        return facts, confusions, reasoning

    # ── Ingestion ─────────────────────────────────────────

    def decompose_text(self, text, context=""):
        """Extract atomic facts from a prose passage."""
        system = """You are a knowledge extraction system.
Given a passage of text, extract ALL factual relationships as atomic triples.

Output format -- one fact per line, nothing else:
subject|predicate|object

Rules:
- lowercase_with_underscores for all terms
- Each fact = ONE relationship (no compound sentences)
- Be thorough -- extract every fact, even obvious ones
- Predicates: is_a, has_part, made_of, causes, requires, produces,
  located_in, used_for, defined_as, measures, contains, enables,
  discovered_by, invented_in, larger_than, etc.
- No commentary, just facts
- Maximum 30 facts per chunk"""

        prompt = f"Extract atomic facts from this text:\n\n{text}"
        if context:
            prompt += f"\n\nContext: {context}"

        response = self.ask(prompt, system)
        if not response:
            return []

        facts = []
        for line in response.strip().split("\n"):
            line = line.strip().strip("-*0123456789.) ")
            if "|" not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3 and parts[0] and parts[1] and parts[2]:
                facts.append({
                    "subject": parts[0], "predicate": parts[1], "object": parts[2]
                })
        return facts

    # ── Query ─────────────────────────────────────────────

    def parse_question(self, question):
        """Extract search terms from a natural language question."""
        system = """Extract the key concepts from this question for database lookup.
Output ONLY a JSON object with these fields:
{
  "subjects": ["term1", "term2"],
  "predicates": ["relationship1"],
  "keywords": ["other", "important", "words"]
}
Use lowercase_with_underscores. Be thorough -- include synonyms."""

        response = self.ask(question, system)
        if not response:
            return None

        try:
            cleaned = re.sub(r'```json?\s*', '', response)
            cleaned = re.sub(r'```', '', cleaned).strip()
            return json.loads(cleaned)
        except Exception:
            words = [w.lower().replace("?", "").replace("'s", "")
                     for w in question.split() if len(w) > 3]
            return {"subjects": words, "predicates": [], "keywords": words}

    def synthesize_answer(self, question, facts, path=None, bridges=None):
        """Synthesize a natural language answer from KB facts and bridges."""
        facts_text = "\n".join([
            f"  [{e['id']}] {e['subject']} -> {e['predicate']}({e['object']}) "
            f"[{e['truth_value']}] (source: {e['source']})"
            for e in facts[:40]
        ])

        path_text = ""
        if path:
            path_text = f"\nConnection path: {' '.join(path)}\n"

        bridge_text = ""
        if bridges:
            lines = []
            for b in bridges[:10]:
                lines.append(f"  {b['term_a']} <-> {b['term_b']} "
                             f"({b['bridge_type']}, strength={b.get('strength',0):.1f}): "
                             f"{b.get('reason','')[:100]}")
            bridge_text = "\nCross-domain bridges:\n" + "\n".join(lines) + "\n"

        system = """You are answering questions using ONLY the provided knowledge base facts.

Rules:
- ONLY use information from the provided facts and bridges
- When you state something, cite the entry ID in brackets like [42]
- If cross-domain bridges are relevant, mention the connection and its type
- If the facts don't contain enough info, say so honestly
- If facts conflict, note the conflict
- Keep the answer clear and concise
- If a fact has truth_value M, mention it's unverified"""

        prompt = f"""Question: {question}
{path_text}{bridge_text}
Known facts:
{facts_text}

Answer the question using only these facts. Cite entry IDs."""

        return self.ask(prompt, system)


# ═══════════════════════════════════════════════════════════
# OLLAMA BACKEND — local LLM via Ollama
# ═══════════════════════════════════════════════════════════

class OllamaLLM(LLMBase):
    """Talks to a local Ollama instance over HTTP."""

    def __init__(self, model="llama3.2", base_url="http://localhost:11434"):
        super().__init__(model, "ollama")
        self.base_url = base_url
        self.available = self._check()

    def _check(self):
        if not HAS_REQUESTS:
            return False
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def ask(self, prompt, system=None, temperature=0.3):
        if not self.available:
            return None
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            r = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": temperature}
                },
                timeout=180)
            if r.status_code == 200:
                self.call_count += 1
                return r.json()["message"]["content"]
        except Exception as e:
            print(f"    LLM error: {e}")
        return None


# ═══════════════════════════════════════════════════════════
# CLAUDE BACKEND — Anthropic API
# ═══════════════════════════════════════════════════════════

CLAUDE_MODELS = [
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
    "claude-opus-4-6",
]

class ClaudeLLM(LLMBase):
    """Talks to the Anthropic Claude API over HTTPS."""

    API_URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(self, model="claude-sonnet-4-6", api_key=None):
        super().__init__(model, "claude")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.available = self._check()

    def _check(self):
        if not HAS_REQUESTS:
            return False
        if not self.api_key:
            print("    Claude: no API key (set ANTHROPIC_API_KEY or pass --api-key)")
            return False
        # Key exists and requests available — mark as ready.
        # Real errors (bad key, network) surface on first ask() call.
        return True

    def _headers(self):
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.API_VERSION,
            "content-type": "application/json",
        }

    def ask(self, prompt, system=None, temperature=0.3):
        if not self.available:
            return None
        body = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if system:
            body["system"] = system

        import time as _time
        for attempt in range(4):
            try:
                r = requests.post(
                    self.API_URL,
                    headers=self._headers(),
                    json=body,
                    timeout=30)
                if r.status_code == 200:
                    self.call_count += 1
                    data = r.json()
                    content = data.get("content", [])
                    texts = [b["text"] for b in content if b.get("type") == "text"]
                    return "\n".join(texts) if texts else None
                elif r.status_code in (429, 529):
                    # Rate limited or overloaded — wait and retry
                    wait = (attempt + 1) * 5
                    print(f" overloaded, retry in {wait}s...", end="", flush=True)
                    _time.sleep(wait)
                    continue
                else:
                    resp = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                    msg = resp.get("error", {}).get("message", r.text[:100])
                    print(f"    Claude error ({r.status_code}): {msg}")
                    return None
            except Exception as e:
                print(f"    Claude error: {e}")
                return None
        print("    Claude: gave up after 4 attempts")
        return None


# ═══════════════════════════════════════════════════════════
# FACTORY — create the right backend
# ═══════════════════════════════════════════════════════════

def create_llm(backend="ollama", model=None, api_key=None):
    """Create an LLM instance. backend='ollama' or 'claude'."""
    if backend == "claude":
        return ClaudeLLM(model=model or "claude-sonnet-4-6", api_key=api_key)
    else:
        return OllamaLLM(model=model or "llama3.2")
