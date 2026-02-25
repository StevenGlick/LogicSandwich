# Three-Layer Pulse System
## A Knowledge Engine Based on the Trivium

---

## What Is This?

This is a system that **builds knowledge the way understanding actually works** — not by memorizing facts, but by finding patterns across facts, testing those patterns, and correcting mistakes. It's inspired by the classical Trivium (Grammar, Logic, Rhetoric) and maps those three layers onto both cognitive operations and hardware.

The core insight: most AI systems try to do everything with one tool (matrix multiplication on GPUs). This system separates knowledge work into three types and routes each to what it's best at:

- **Word Layer (Grammar)** → GPU: Raw language processing via LLM
- **Logic Layer** → CPU: Symbolic pattern matching, graph traversal, set operations
- **Idea Layer (Rhetoric)** → CPU: Cross-domain synthesis, analogy detection, meaning

The result is a knowledge base that **grows, self-corrects, and discovers things that weren't in any of its sources**.

---

## The Architecture in Plain English

### The Three Layers

Think of it geometrically:

| Layer | Trivium | Geometry | What It Does | Hardware |
|-------|---------|----------|--------------|----------|
| Word | Grammar | **Point** (0D) | Atomic facts: "A is B" | GPU (LLM) |
| Logic | Logic | **Line** (1D) | Chains: "If A→B and B→C then A→C" | CPU |
| Idea | Rhetoric | **Plane** (2D) | Surfaces where lines intersect: meaning | CPU |

**Word Layer** — This is where raw information enters the system. An LLM reads a textbook paragraph and extracts atomic facts like `photosynthesis → produces → glucose`. Each fact is a point — zero-dimensional, no connections yet. This runs on GPU because LLM inference is matrix multiplication.

**Logic Layer** — This is where facts get connected. If `photosynthesis → produces → glucose` and `glucose → is_a → sugar`, the logic engine proposes `photosynthesis → produces → sugar` (a transitive inference). This is just dictionary lookups and set intersections — CPUs have been screaming fast at this since the 1990s.

**Idea Layer** — This is where meaning emerges. The system notices that `photosynthesis` and `nuclear_fusion` both `produces(energy)` even though they're in completely different domains (biology vs physics). That cross-domain pattern — "there are multiple ways to produce energy" — doesn't exist in either fact alone. It exists on the **surface** where the two lines intersect. That's the plane. That's meaning.

### Why This Matters

Most knowledge systems are either:
- **Just an LLM**: Fast at generating text, terrible at logical consistency, can't show its work
- **Just a database**: Great at storing facts, can't reason, can't find patterns

This is both. The LLM generates and verifies knowledge. The database stores it with evidence chains. The pulse engine finds patterns the LLM never would. And every single claim traces back to evidence — no hallucination, because the Idea layer works from verified facts, not training data.

---

## The Five Pulse Operations

Every "pulse" cycle runs these operations on the knowledge base:

### 1. Merge
**What:** Combines entries that say the same thing about the same subject.
**Example:** Three entries saying `transformer → uses → self_attention`, `transformer → uses → residual_connection`, `transformer → uses → layer_normalization` become one aggregate entry.
**Why:** Housekeeping. Reduces clutter, makes other operations faster.

### 2. Transitive
**What:** If A→B and B→C, propose A→C.
**Example:** `backpropagation → uses → gradient_descent` + `gradient_descent → is_a → optimization_algorithm` → proposes `backpropagation → is_a → optimization_algorithm`
**Why:** This is reasoning by chain. It's the Logic layer's core operation.
**Important:** Results come out as **M (Maybe)**, not T (True), because chains can be wrong. They go into the verification queue.

### 3. Cross-Domain
**What:** Find different subjects that share the same property.
**Example:** `lstm → mitigates → vanishing_gradient` AND `residual_connection → mitigates → vanishing_gradient` → creates `lstm,residual_connection → both_mitigates → vanishing_gradient`
**Why:** This is the Idea layer. The insight "there are multiple solutions to vanishing gradients" doesn't exist in either fact alone. **This operation produces ~40% of all generated knowledge.** It's the biggest source of genuinely new understanding.

### 4. Analogy
**What:** Find subjects with similar *shapes* of knowledge (share the same types of predicates).
**Example:** `transformer` and `mamba` both have predicates like `is_a`, `uses`, `enables`, `has_complexity` → they're structurally similar.
**Why:** Structural similarity suggests deep relationships even when specific facts differ. **This produces ~30% of generated knowledge.**

### 5. Verification (LLM)
**What:** Take M (Maybe) entries and ask the LLM: "Is this actually true?"
**Example:** The transitive engine proposes `diffusion_model → trains_via → adversarial_competition`. The LLM says: "FALSE — diffusion models use iterative denoising, not adversarial training. That's GANs."
**Why:** Self-correction. The system catches its own mistakes. The LLM also **decomposes** its reasoning into new atomic facts, which go back into the database for more pulsing.

---

## The Confusion Map

The most valuable output isn't what the system knows — it's **where knowledge goes wrong**.

When verification catches a false inference, it records WHY:
- `neural_network ≠ brain_simulation` (name implies biological accuracy that doesn't exist)
- `rag ≠ hallucination_cure` (marketed as solution, is only mitigation)
- `diffusion_model ≠ gan` (both generative, completely different training)

This confusion map grows organically. It's a map of **where human understanding commonly fails** — auto-generated from the structure of knowledge itself.

---

## Hardware Insight

The Logic and Idea layers are **CPU-bound**. Dictionary lookups, set intersections, graph traversal. No floating point, no tensors, no matrix multiplication. In our tests, 150→931 entries in under 2 seconds in unoptimized Python. In C/Rust it would run in microseconds.

The Word layer (LLM) is **GPU-bound**. Massive parallel matrix multiplications for attention computation.

This means:
```
CPU (fast, continuous):
  pulse → pulse → pulse → [queue: 47 entries need verification] → pulse

GPU (slower, batch):
  [picks up 47 entries] → LLM inference → [returns corrections]

CPU receives results → pulse → pulse → [new queue] → ...
```

The CPU never waits for the GPU. It keeps finding patterns. When the GPU finishes a batch, the CPU integrates results and has new material. **They feed each other asynchronously.**

---

## The Curriculum System

The system can learn like a student, progressing through grade levels:

- **Kindergarten**: Colors, shapes, counting, animal babies
- **Elementary**: States of matter, food chains, simple machines
- **Middle School**: Cell division, chemical reactions, Newton's laws
- **High School**: Quantum mechanics, calculus, philosophy
- **College**: Molecular biology, programming, formal logic

Each grade level introduces topics that **build on previous knowledge**. When 5th grade teaches photosynthesis, the transitive engine connects it to the plant knowledge from 1st grade and the food chain knowledge from 2nd grade. Later concepts resolve faster because the foundation is already there.

The questions at each grade are deliberately tricky:
- "Does evolution mean we came from monkeys?" (Grade 8 — tests confusion tracking)
- "Is calculus just advanced algebra?" (Grade 11 — forces decomposition of why it's wrong)
- "Can a computer think?" (Grade 12 — genuinely unresolvable, should stay M)

---

## Domain Awareness (The Immune System)

Without scope control, mixing ConceptNet ("an orange has layers") with AI knowledge ("a neural network has layers") produces garbage: `orange,neural_network → both_has → layers`.

Three defense layers prevent this:

### Type Gates
Certain predicates only make sense between certain domains. `eats` is biology/everyday. `compiles_to` is computer science. `orbits` is physics. The gate checks domain compatibility before allowing any inference.

### Domain Affinity
Some domain pairs can cross-pollinate meaningfully (physics+math, biology+chemistry). Others can't (food+computing). Cross-domain patterns are only generated between compatible pairs.

### Nonsense Filter
Fast heuristic checks: self-referential entries, empty fields, typed predicate violations, hard-blocked domain pairs. Runs before any entry is committed.

Test results (10/10 correct):
```
✓ transformer → uses → attention           (same domain — pass)
✓ photosynthesis → produces → glucose       (same domain — pass)
✓ orange → trains_via → backpropagation     (food+CS — BLOCKED)
✓ cake → compiles_to → assembly             (food+CS — BLOCKED)
✓ dog → orbits → sun                        (biology+physics typed — BLOCKED)
✓ france → reacts_with → oxygen             (geography+chemistry — BLOCKED)
```

---

## Database Optimization

The `optimize.py` script reorganizes the knowledge base for scale:

### Predicate Normalization
Maps synonyms to canonical forms: `uses/utilizes/employs → uses`. Prevents graph fragmentation.

### Hierarchy Index
Precomputes is_a ancestry chains. `dog → mammal → animal → living_thing`. Instant ancestor lookups instead of multi-hop queries.

### Centrality Scoring
Scores concepts by connectivity. High centrality = foundational (many things depend on it). Used to prioritize query results. In our AI database, `parallel_processing` scored 337 — the most foundational concept, discovered automatically from graph structure.

### Full-Text Search (FTS5)
SQLite's built-in search engine with stemming. "Weather" finds "weathering." "Run" finds "running." Boolean operators, phrase matching, prefix matching.

### Domain Clustering
Groups entries by knowledge domain so the pulse engine can process one domain at a time instead of everything at once. Critical for scaling to millions of entries.

---

## Knowledge Sources

### ConceptNet (NO LLM NEEDED — #1 recommended start)
Commonsense knowledge already in triple format. "Fire is hot." "Dogs have four legs."
~400MB download, hundreds of thousands of facts, zero LLM cost.
```
python3 ingest.py --conceptnet conceptnet-assertions-5.7.0.csv.gz
```

### Wikidata (NO LLM NEEDED)
Structured encyclopedia. "Earth → instance_of → planet." Millions of triples.
```
python3 ingest.py --wikidata latest-all.json.bz2
```

### Textbooks (requires LLM)
Drop .txt files in a folder. LLM decomposes each paragraph into atomic facts.
Free sources: OpenStax (https://openstax.org), CK-12 (https://www.ck12.org)
```
python3 ingest.py --textbook biology_101.txt
python3 ingest.py --textbook-dir my_textbooks/
```

### Wikipedia (requires LLM)
Full dump extracted to plaintext, then LLM-decomposed.
```
python3 ingest.py --wikipedia wiki_text/
```

### Agent Queue
External programs write tasks to the database. The agent loop processes them.
```
python3 ingest.py --add-task research "quantum computing"
python3 ingest.py --add-task question "What is entropy?"
python3 ingest.py --agent-loop
```

---

## Query System

### Natural Language
```
python3 query.py "What is the relationship between mamba and transformers?"
```
Parses question → searches database → walks knowledge graph → synthesizes cited answer.

### Direct Lookup
```
python3 query.py --raw transformer
```
Shows everything known about a term: outgoing connections, incoming connections, evidence.

### Graph Visualization
```
python3 query.py --graph transformer 2
```
Draws the knowledge web around a concept, N hops deep. Shows the structure of understanding.

### Path Finding
```
python3 query.py --path photosynthesis glucose
```
BFS through the knowledge graph. Finds how two concepts connect through intermediate nodes. Like six degrees of separation for knowledge.

### Interactive Mode
```
python3 query.py --interactive
```
Chat interface with commands: `/raw`, `/graph`, `/path`, `/stats`, `/confusions`, `/unresolved`.

---

## Bridge System (Cross-Domain Analogies)

The domain scope system keeps knowledge clean by blocking nonsense cross-domain connections. But some cross-domain connections are the most valuable knowledge there is. Newton's apple connecting everyday experience to gravitational physics. Neural networks borrowing architecture from biological neurons. Predator-prey cycles following the same equations as market supply and demand.

The **bridge table** is a separate space where these documented cross-domain connections live, with their justification, type, and strength attached. The main knowledge base stays lean and domain-pure. Bridges capture the nuance.

### Bridge Types

| Type | Icon | Meaning | Example |
|------|------|---------|---------|
| Historical | 📜 | Real event created the connection | Newton's apple → gravity |
| Inspirational | 💡 | One domain borrowed from another | Biology → neural networks |
| Structural | 🔷 | Same math in different domains | Lotka-Volterra in ecology AND economics |
| Metaphorical | 🎭 | Useful for teaching, not literally true | Atom = tiny solar system |
| Causal | ⚡ | One domain actually IS the other | Music harmony IS wave physics |
| Emergent | ✨ | System discovered it automatically | Pulse engine cross-domain pattern |

### Why This Matters

A flat knowledge base can only say "true" or "false." The bridge system can say: "This connection is metaphorical, strength 0.3, useful for initial teaching but fundamentally wrong — here's where it breaks down." That's the kind of nuanced understanding that makes the difference between a database and actual knowledge.

The bridge discovery engine also scans the existing KB for *potential* bridges — cross-domain patterns that might be meaningful rather than coincidental. These candidates go to the LLM for verification: "Is the fact that rivers and blood vessels both exhibit branching a coincidence, or is there real mathematical structure here?" (Answer: real structure — Murray's law and Horton's law are the same branching optimization.)

```bash
python3 bridges.py --seed                        # load famous bridges
python3 bridges.py --show                        # display all bridges  
python3 bridges.py --find gravity                # bridges involving a term
python3 bridges.py --between biology physics     # bridges between domains
python3 bridges.py --analyze                     # discover new bridges in KB
```

---

| File | Purpose | Depends On |
|------|---------|------------|
| `curriculum_pulse.py` | Main brain: curriculum + pulse engine + web dashboard | `requests`, `flask` (optional) |
| `ingest.py` | Knowledge ingestion from external sources | `requests` (optional) |
| `query.py` | Query engine: ask questions, walk graphs | `requests` (optional) |
| `optimize.py` | Database optimization and indexing | nothing (pure Python + SQLite) |
| `domain_scope.py` | Domain awareness and nonsense prevention | nothing (pure Python + SQLite) |
| `bridges.py` | Cross-domain analogy and bridge system | `domain_scope.py` (optional) |
| `pulse_production.py` | Simpler standalone pulse system (no curriculum) | `requests` (optional) |
| `ai_kb_big.json` | 150 seed entries about AI/ML | — |
| `curriculum.db` | SQLite database (generated) | — |

All files share the same SQLite database. Run any combination simultaneously.

---

## Setup Guide

### Prerequisites
```bash
# Python 3.8+ (probably already installed)
python3 --version

# Install Python packages
pip install requests flask

# Install Ollama (local LLM runner)
# Visit https://ollama.com and download for your OS

# Pull a model (pick one that fits your GPU VRAM)
ollama pull llama3.2        # ~2GB, fits most GPUs
# OR
ollama pull mistral          # ~4GB, better quality
# OR  
ollama pull phi3             # ~2GB, good for smaller GPUs
```

### Quick Start
```bash
# 1. Test that everything works (no LLM needed)
python3 pulse_production.py --seed-json ai_kb_big.json --cycles 10

# 2. Run the curriculum with dashboard
python3 curriculum_pulse.py --dashboard
# Open http://localhost:5000 in your browser

# 3. Download ConceptNet for massive instant knowledge
# Get: https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
python3 ingest.py --conceptnet conceptnet-assertions-5.7.0.csv.gz

# 4. Optimize after big ingests
python3 optimize.py

# 5. Query the knowledge base
python3 query.py --interactive
```

### Recommended First Run
```bash
# Terminal 1: Start the curriculum (runs through K-13)
python3 curriculum_pulse.py --dashboard --model llama3.2

# Terminal 2: Watch the dashboard
# Open http://localhost:5000

# Terminal 3: Feed it additional knowledge
python3 ingest.py --add-task research "photosynthesis"
python3 ingest.py --add-task research "plate tectonics"
python3 ingest.py --agent-loop
```

---

## What Makes This Special

1. **Self-correction**: The system catches its own mistakes. False transitive inferences get flagged and corrected by the LLM. The confusion map grows organically.

2. **Emergent knowledge**: Cross-domain patterns and analogies produce understanding that wasn't in any single source. The Idea layer finds meaning at the intersection of facts.

3. **Full auditability**: Every fact traces to evidence. Every inference traces to the chain that produced it. Every correction traces to what was wrong and why.

4. **Hardware-optimal**: CPU does what CPUs are good at (symbolic logic). GPU does what GPUs are good at (matrix math). Neither waits for the other.

5. **Compositional growth**: New knowledge doesn't just add to the pile — it multiplies. Post-injection pulses chain new facts through existing knowledge, producing bursts of derived understanding. In testing, 15 LLM additions produced 50+ new entries through pulsing.

6. **Domain-aware**: The immune system prevents combinatorial nonsense when mixing knowledge from different fields while still allowing legitimate cross-domain insights.

7. **Curriculum-structured**: Knowledge builds on itself the way understanding actually develops. Later concepts depend on and connect to earlier foundations.

---

## Growth Data From Testing

Starting from 150 seed entries about AI/ML:

| Stage | Entries | Source |
|-------|---------|--------|
| Seed | 150 | Manual / web research |
| After 3 pulses | 343 | Merge, transitive, cross-domain, analogy |
| After LLM verification | 376 | Corrections + decompositions |
| After question injection | 396 | 6 questions answered |
| After web search injection | 410 | 14 new concepts |
| After 3 more pulses | 530 | Chaining new knowledge through existing |
| Final (11 generations) | 819 | Full convergence |

Key observation: **113 new entries** appeared in one burst after injecting 34 new facts (questions + web search). The system doesn't just store what you give it — it discovers what the new facts imply when combined with everything it already knows.

---

## The Trivium Connection

This project started from an observation about how the classical Trivium maps to AI architecture:

| Trivium | AI Layer | Operation | Geometry |
|---------|----------|-----------|----------|
| Grammar | Word/Token | Parse, tokenize, embed | Point (0D) |
| Logic | Inference | Chain, deduce, verify | Line (1D) |
| Rhetoric | Synthesis | Analogize, cross-reference, mean | Plane (2D) |

The Grammar stage gives you vocabulary — isolated facts, no connections. Logic gives you reasoning — chains of implication, one-dimensional paths through knowledge. Rhetoric gives you meaning — the surface that emerges when multiple lines of reasoning intersect.

What's remarkable is that this isn't just a metaphor. **It's a hardware allocation strategy.** Grammar/Word is GPU work (LLM inference = matrix multiplication). Logic is CPU work (set operations, graph traversal). The Trivium predicted the optimal compute split 2000 years before GPUs existed.

---

## Future Directions

- **Multiple LLMs**: Run different models (Llama, Mistral, Phi) simultaneously. They catch different errors and inject different knowledge.
- **Continuous operation**: Run for days/weeks. The system converges — every M either resolves or decomposes into sub-questions that themselves resolve.
- **Training data export**: The verified knowledge base becomes training data for specialized models. Bake the Logic and Idea layers into smaller, faster models.
- **Agent integration**: Hook up an autonomous agent that reads the confusion map, formulates research questions, searches for answers, and feeds them back in.
- **Entry aging**: Facts that keep getting re-confirmed become more trusted. Facts that get contradicted decay.
- **Source credibility**: Wikidata triples > textbook extractions > LLM guesses > transitive inferences.
- **Contradiction detection**: If entry A says "X is Y" and entry B says "X is not Y," flag for resolution.

---

*Built from conversations between a human game developer with deep knowledge of esoteric pattern systems and an AI, exploring whether ancient frameworks for organizing knowledge could improve how machines learn.*
