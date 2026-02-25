#!/usr/bin/env python3
"""
KNOWLEDGE INGESTION SYSTEM
═══════════════════════════

Feeds knowledge into the pulse system from multiple sources:

1. TEXTBOOKS / TEXT FILES
   - Drop .txt files in a folder
   - LLM decomposes each chunk into atomic facts
   - Facts go into the database

2. WIKIPEDIA DUMP
   - Download: https://dumps.wikimedia.org/enwiki/latest/
   - Get: enwiki-latest-pages-articles.xml.bz2 (about 22GB compressed)
   - This script extracts articles and decomposes them
   - OR use the pre-extracted plaintext version

3. WIKIDATA (BEST SOURCE — already structured!)
   - Download: https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2
   - Already in subject-predicate-object format
   - Millions of triples, no LLM needed for basic import

4. CONCEPTNET
   - Download: https://s3.amazonaws.com/conceptnet/downloads/2019/edges/
   - conceptnet-assertions-5.7.0.csv.gz
   - Commonsense knowledge, already structured

5. AGENT QUEUE
   - Other systems can write tasks to the agent_queue table
   - Tasks: "verify X", "research topic Y", "decompose text Z"
   - The agent loop picks them up and processes them

USAGE:
  python3 ingest.py --textbook path/to/file.txt      # ingest a text file
  python3 ingest.py --textbook-dir path/to/folder/    # ingest all .txt files
  python3 ingest.py --wikipedia path/to/dump.xml.bz2  # ingest wikipedia
  python3 ingest.py --wikidata path/to/wikidata.json  # ingest wikidata
  python3 ingest.py --conceptnet path/to/conceptnet.csv  # ingest conceptnet
  python3 ingest.py --agent-loop                       # run agent queue processor
  python3 ingest.py --list-sources                     # show available sources
"""

import sqlite3
import json
import time
import argparse
import os
import sys
import re
from collections import defaultdict


# ═══════════════════════════════════════════════════════════
# SHARED MODULES
# ═══════════════════════════════════════════════════════════
from db import KnowledgeDB
from llm import create_llm

# ═══════════════════════════════════════════════════════════
# TEXT CHUNKER
# ═══════════════════════════════════════════════════════════

def chunk_text(text, chunk_size=1500, overlap=200):
    """
    Break text into overlapping chunks for LLM processing.
    
    chunk_size: approximate characters per chunk
    overlap: characters of overlap between chunks
    
    We split on paragraph boundaries when possible,
    falling back to sentence boundaries.
    """
    # Split into paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    
    chunks = []
    current = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # If adding this paragraph would exceed chunk_size, save current and start new
        if len(current) + len(para) > chunk_size and current:
            chunks.append(current.strip())
            # Keep overlap from end of current chunk
            if overlap > 0 and len(current) > overlap:
                current = current[-overlap:]
            else:
                current = ""
        
        current += para + "\n\n"
    
    # Don't forget the last chunk
    if current.strip():
        chunks.append(current.strip())
    
    return chunks


# ═══════════════════════════════════════════════════════════
# TEXTBOOK INGESTION
# ═══════════════════════════════════════════════════════════

def ingest_textbook(db, llm, filepath, grade_level=0):
    """
    Ingest a text file:
    1. Read the file
    2. Chunk it into ~1500 char pieces
    3. Send each chunk to LLM for decomposition
    4. Add resulting facts to database
    """
    print(f"\n  📖 Ingesting: {filepath}")
    
    if not llm.available:
        print("    ⚠ No LLM available. Cannot decompose text.")
        return 0
    
    # Read file
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
    except Exception as e:
        print(f"    ⚠ Error reading file: {e}")
        return 0
    
    print(f"    Size: {len(text):,} characters")
    
    # Chunk
    chunks = chunk_text(text)
    print(f"    Chunks: {len(chunks)}")
    
    # Process each chunk
    total_facts = 0
    errors = 0
    filename = os.path.basename(filepath)
    
    for i, chunk in enumerate(chunks):
        # Skip very short chunks
        if len(chunk) < 50:
            continue
        
        print(f"    Chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...", end=" ", flush=True)
        
        try:
            facts = llm.decompose_text(chunk, context=f"From: {filename}")
            added = 0
            for fact in facts:
                result = db.add(
                    fact["subject"], fact["predicate"], fact["object"], "T",
                    evidence_for=[f"textbook: {filename}, chunk {i+1}"],
                    source=f"textbook:{filename}",
                    generation=0, grade_level=grade_level,
                    normalize=True)
                if result:
                    added += 1
            total_facts += added
            print(f"+{added} facts ({len(facts)} extracted, {added} new)")
        except Exception as e:
            print(f"ERROR: {e}")
            errors += 1
        
        # Brief pause to not overwhelm Ollama
        time.sleep(0.5)
    
    db.log_ingest("textbook", filepath, len(chunks), total_facts, errors)
    print(f"\n    ✓ Complete: {total_facts} facts added from {len(chunks)} chunks")
    return total_facts


def ingest_textbook_dir(db, llm, dirpath, grade_level=0):
    """Ingest all .txt files in a directory."""
    print(f"\n  📚 Ingesting directory: {dirpath}")
    
    files = sorted([f for f in os.listdir(dirpath) 
                   if f.endswith(('.txt', '.md', '.text'))])
    
    if not files:
        print(f"    No .txt files found")
        return 0
    
    print(f"    Found {len(files)} files")
    total = 0
    for f in files:
        total += ingest_textbook(db, llm, os.path.join(dirpath, f), grade_level)
    
    return total


# ═══════════════════════════════════════════════════════════
# WIKIPEDIA INGESTION
# ═══════════════════════════════════════════════════════════

def ingest_wikipedia(db, llm, path, max_articles=1000):
    """
    Ingest Wikipedia dump.
    
    Supports two formats:
    1. Pre-extracted plaintext (one article per file in a directory)
       → Use WikiExtractor first: https://github.com/attardi/wikiextractor
       → python -m wikiextractor.WikiExtractor dump.xml.bz2 -o output_dir
    
    2. Simple text file with articles separated by blank lines
       (for testing with smaller extracts)
    
    For the full dump, you want WikiExtractor output — it produces
    clean text files organized in folders.
    """
    print(f"\n  📰 Ingesting Wikipedia: {path}")
    
    if not llm.available:
        print("    ⚠ No LLM available. Cannot decompose articles.")
        print("    TIP: For Wikipedia, consider using Wikidata instead (already structured)")
        return 0
    
    total_facts = 0
    articles_processed = 0
    
    if os.path.isdir(path):
        # WikiExtractor output: directories of text files
        for root, dirs, files in os.walk(path):
            for fname in sorted(files):
                if articles_processed >= max_articles:
                    break
                
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
                except IOError:
                    continue
                
                # WikiExtractor format: <doc id="..." url="..." title="...">content</doc>
                # Extract articles
                articles = re.findall(
                    r'<doc[^>]*title="([^"]*)"[^>]*>(.*?)</doc>', 
                    content, re.DOTALL)
                
                if not articles:
                    # Plain text file, treat as single article
                    articles = [(fname, content)]
                
                for title, text in articles:
                    if articles_processed >= max_articles:
                        break
                    
                    text = text.strip()
                    if len(text) < 100:
                        continue
                    
                    print(f"    [{articles_processed+1}] {title[:50]}...", end=" ", flush=True)
                    
                    # Take first ~3000 chars (summary + intro usually)
                    excerpt = text[:3000]
                    chunks = chunk_text(excerpt, chunk_size=1500)
                    
                    article_facts = 0
                    for chunk in chunks[:3]:  # Max 3 chunks per article
                        facts = llm.decompose_text(chunk, context=f"Wikipedia: {title}")
                        for fact in facts:
                            result = db.add(
                                fact["subject"], fact["predicate"], fact["object"], "T",
                                evidence_for=[f"wikipedia: {title}"],
                                source="wikipedia",
                                generation=0,
                    normalize=True)
                            if result:
                                article_facts += 1
                        time.sleep(0.3)
                    
                    total_facts += article_facts
                    articles_processed += 1
                    print(f"+{article_facts} facts")
    
    elif os.path.isfile(path):
        # Single text file — split on double newlines or headers
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Try to split into sections
        sections = re.split(r'\n={2,}\s*\n|\n\n\n+', content)
        
        for section in sections:
            if articles_processed >= max_articles:
                break
            section = section.strip()
            if len(section) < 100:
                continue
            
            title = section[:50].split('\n')[0]
            print(f"    [{articles_processed+1}] {title}...", end=" ", flush=True)
            
            chunks = chunk_text(section, chunk_size=1500)
            article_facts = 0
            for chunk in chunks[:3]:
                facts = llm.decompose_text(chunk, context="Wikipedia article")
                for fact in facts:
                    result = db.add(
                        fact["subject"], fact["predicate"], fact["object"], "T",
                        evidence_for=["wikipedia"],
                        source="wikipedia", generation=0,
                    normalize=True)
                    if result:
                        article_facts += 1
                time.sleep(0.3)
            
            total_facts += article_facts
            articles_processed += 1
            print(f"+{article_facts} facts")
    
    db.log_ingest("wikipedia", path, articles_processed, total_facts, 0)
    print(f"\n    ✓ {articles_processed} articles → {total_facts} facts")
    return total_facts


# ═══════════════════════════════════════════════════════════
# WIKIDATA INGESTION (NO LLM NEEDED!)
# ═══════════════════════════════════════════════════════════
# This is the gold mine. Wikidata is already structured as
# subject-predicate-object triples. We just need to parse
# the JSON dump and normalize the terms.
# ═══════════════════════════════════════════════════════════

# Common Wikidata property IDs → human-readable predicates
WIKIDATA_PROPERTIES = {
    "P31": "is_a",              # instance of
    "P279": "subclass_of",      # subclass of
    "P361": "part_of",          # part of
    "P527": "has_part",         # has part
    "P1542": "has_effect",      # has effect
    "P1552": "has_quality",     # has quality
    "P186": "made_of",          # material used
    "P176": "made_by",          # manufacturer
    "P17": "located_in_country",# country
    "P131": "located_in",       # located in administrative entity
    "P36": "has_capital",       # capital
    "P30": "located_on_continent", # continent
    "P1376": "capital_of",      # capital of
    "P37": "official_language",
    "P47": "shares_border_with",
    "P138": "named_after",
    "P61": "discovered_by",
    "P170": "created_by",
    "P178": "developed_by",
    "P1071": "location_of_creation",
    "P127": "owned_by",
    "P155": "preceded_by",      # follows
    "P156": "followed_by",     # preceded by
    "P366": "used_for",        # use
    "P1542": "causes",
    "P1478": "has_immediate_cause",
    "P828": "has_cause",
    "P2283": "uses",
    "P101": "field_of_work",
    "P106": "occupation",
    "P39": "position_held",
    "P108": "employer",
    "P69": "educated_at",
    "P800": "notable_work",
    "P135": "movement",         # art/cultural movement
    "P136": "genre",
    "P495": "country_of_origin",
    "P1376": "capital_of",
    "P6": "head_of_government",
    "P35": "head_of_state",
}

def ingest_wikidata(db, path, max_entities=50000, language="en"):
    """
    Ingest Wikidata JSON dump.
    
    NO LLM NEEDED — data is already structured!
    
    Download from:
      https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2
      (WARNING: ~100GB compressed, ~1.4TB uncompressed)
    
    For testing, use a subset. You can create one with:
      bzcat latest-all.json.bz2 | head -100000 > wikidata_sample.json
    
    Or download a pre-filtered subset from:
      https://www.wikidata.org/wiki/Wikidata:Database_download
    """
    import bz2
    
    print(f"\n  🌐 Ingesting Wikidata: {path}")
    print(f"    Max entities: {max_entities:,}")
    print(f"    NO LLM needed — data is pre-structured!")
    
    total_facts = 0
    entities_processed = 0
    
    # Detect format
    is_bz2 = path.endswith('.bz2')
    opener = bz2.open if is_bz2 else open
    
    try:
        with opener(path, 'rt', encoding='utf-8') as f:
            for line in f:
                if entities_processed >= max_entities:
                    break
                
                line = line.strip().rstrip(',')
                if not line or line in ('[', ']'):
                    continue
                
                try:
                    entity = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                # Get English label
                labels = entity.get("labels", {})
                en_label = labels.get(language, {}).get("value", "")
                if not en_label:
                    continue
                
                subject = en_label.lower().replace(" ", "_")
                
                # Extract claims (the actual triples!)
                claims = entity.get("claims", {})
                entity_facts = 0
                
                for prop_id, claim_list in claims.items():
                    predicate = WIKIDATA_PROPERTIES.get(prop_id)
                    if not predicate:
                        continue  # skip unknown properties
                    
                    for claim in claim_list:
                        mainsnak = claim.get("mainsnak", {})
                        datavalue = mainsnak.get("datavalue", {})
                        
                        # Handle entity references (most common)
                        if datavalue.get("type") == "wikibase-entityid":
                            obj_id = datavalue["value"].get("id", "")
                            # We'd need to resolve this ID to a label
                            # For now, store the ID — a second pass resolves them
                            # OR use the pre-resolved dumps
                            obj = obj_id  # placeholder
                        elif datavalue.get("type") == "string":
                            obj = datavalue["value"]
                        elif datavalue.get("type") == "monolingualtext":
                            if datavalue["value"].get("language") == language:
                                obj = datavalue["value"]["text"]
                            else:
                                continue
                        else:
                            continue
                        
                        if obj:
                            result = db.add(
                                subject, predicate, 
                                obj.lower().replace(" ", "_"),
                                "T",
                                evidence_for=[f"wikidata:{entity.get('id','')}"],
                                source="wikidata",
                    normalize=True)
                            if result:
                                entity_facts += 1
                                total_facts += 1
                
                entities_processed += 1
                if entities_processed % 1000 == 0:
                    print(f"    [{entities_processed:,}] entities, {total_facts:,} facts...", flush=True)
    
    except ImportError:
        print("    ⚠ bz2 module needed for compressed files")
        print("    Decompress first: bunzip2 file.json.bz2")
    except Exception as e:
        print(f"    ⚠ Error: {e}")
    
    db.log_ingest("wikidata", path, entities_processed, total_facts, 0)
    print(f"\n    ✓ {entities_processed:,} entities → {total_facts:,} facts")
    return total_facts


# ═══════════════════════════════════════════════════════════
# CONCEPTNET INGESTION (NO LLM NEEDED!)
# ═══════════════════════════════════════════════════════════
# ConceptNet is commonsense knowledge — "fire is hot",
# "dogs have four legs", "a bicycle is used for transport"
# Already in edge format, just needs parsing.
# ═══════════════════════════════════════════════════════════

CONCEPTNET_RELATIONS = {
    "/r/IsA": "is_a",
    "/r/PartOf": "part_of",
    "/r/HasA": "has",
    "/r/UsedFor": "used_for",
    "/r/CapableOf": "can",
    "/r/AtLocation": "found_at",
    "/r/Causes": "causes",
    "/r/HasProperty": "has_property",
    "/r/MadeOf": "made_of",
    "/r/ReceivesAction": "receives_action",
    "/r/CreatedBy": "created_by",
    "/r/DefinedAs": "defined_as",
    "/r/HasPrerequisite": "requires",
    "/r/HasSubevent": "involves",
    "/r/HasFirstSubevent": "starts_with",
    "/r/HasLastSubevent": "ends_with",
    "/r/MotivatedByGoal": "motivated_by",
    "/r/CausesDesire": "causes_desire_for",
    "/r/Desires": "desires",
    "/r/SymbolOf": "symbolizes",
    "/r/LocatedNear": "near",
    "/r/SimilarTo": "similar_to",
    "/r/Antonym": "opposite_of",
    "/r/DerivedFrom": "derived_from",
    "/r/RelatedTo": "related_to",
}

def ingest_conceptnet(db, path, max_edges=100000, language="en", min_weight=1.0):
    """
    Ingest ConceptNet CSV dump.
    
    NO LLM NEEDED — data is pre-structured!
    
    Download from:
      https://s3.amazonaws.com/conceptnet/downloads/2019/edges/
      File: conceptnet-assertions-5.7.0.csv.gz
      (about 400MB compressed)
    
    Decompress: gunzip conceptnet-assertions-5.7.0.csv.gz
    
    Format: tab-separated
      URI  relation  subject  object  metadata_json
    
    Example line:
      /a/[/r/IsA/,/c/en/cat/,/c/en/animal/]  /r/IsA  /c/en/cat  /c/en/animal  {"weight":4.0}
    """
    import gzip
    
    print(f"\n  🧠 Ingesting ConceptNet: {path}")
    print(f"    Max edges: {max_edges:,}")
    print(f"    Min weight: {min_weight}")
    print(f"    NO LLM needed — commonsense knowledge, pre-structured!")
    
    total_facts = 0
    edges_processed = 0
    
    is_gz = path.endswith('.gz')
    opener = gzip.open if is_gz else open
    mode = 'rt' if is_gz else 'r'
    
    try:
        with opener(path, mode, encoding='utf-8') as f:
            for line in f:
                if edges_processed >= max_edges:
                    break
                
                parts = line.strip().split('\t')
                if len(parts) < 5:
                    continue
                
                relation = parts[1]
                subj_uri = parts[2]
                obj_uri = parts[3]
                
                # Filter to English only
                if f"/c/{language}/" not in subj_uri or f"/c/{language}/" not in obj_uri:
                    continue
                
                # Map relation
                predicate = CONCEPTNET_RELATIONS.get(relation)
                if not predicate:
                    continue
                
                # Extract terms from URIs: /c/en/dog → dog
                try:
                    subject = subj_uri.split(f"/c/{language}/")[1].split("/")[0]
                    obj = obj_uri.split(f"/c/{language}/")[1].split("/")[0]
                except (IndexError, ValueError):
                    continue
                
                # Check weight
                try:
                    meta = json.loads(parts[4])
                    weight = meta.get("weight", 0)
                    if weight < min_weight:
                        continue
                except (json.JSONDecodeError, IndexError, TypeError):
                    continue
                
                # Add to database
                result = db.add(
                    subject.replace("_", "_"), predicate, obj.replace("_", "_"),
                    "T",
                    evidence_for=[f"conceptnet:weight={weight:.1f}"],
                    source="conceptnet",
                    normalize=True)
                
                if result:
                    total_facts += 1
                
                edges_processed += 1
                if edges_processed % 10000 == 0:
                    print(f"    [{edges_processed:,}] edges, {total_facts:,} facts...", flush=True)
    
    except Exception as e:
        print(f"    ⚠ Error: {e}")
    
    db.log_ingest("conceptnet", path, edges_processed, total_facts, 0)
    print(f"\n    ✓ {edges_processed:,} edges → {total_facts:,} facts")
    return total_facts


# ═══════════════════════════════════════════════════════════
# AGENT QUEUE
# ═══════════════════════════════════════════════════════════
# Other programs can add tasks to the agent_queue table.
# This loop picks them up and processes them.
#
# Task types:
#   "verify"    — verify a specific claim
#   "research"  — generate facts about a topic
#   "question"  — answer a question, add facts
#   "decompose" — decompose a text passage into facts
#   "search"    — search the web for info on a topic (future)
# ═══════════════════════════════════════════════════════════

def run_agent_loop(db, llm):
    """
    Process tasks from the agent queue.
    Another program (or the dashboard) can add tasks.
    """
    print(f"\n  🤖 Agent loop started")
    print(f"    Watching agent_queue table in {db.db_path}")
    print(f"    Add tasks from another script or the dashboard")
    print(f"    Ctrl+C to stop\n")
    
    if not llm.available:
        print("    ⚠ No LLM. Can only process 'decompose' tasks with text files.")
    
    while True:
        task = db.get_next_task()
        
        if not task:
            time.sleep(2)  # poll every 2 seconds
            continue
        
        task_type = task["task_type"]
        payload = task["payload"]
        
        try:
            payload_data = json.loads(payload) if payload.startswith('{') else {"text": payload}
        except (json.JSONDecodeError, AttributeError):
            payload_data = {"text": payload}
        
        print(f"  📋 Task #{task['id']}: {task_type} — {str(payload_data)[:60]}")
        
        if task_type == "research" and llm.available:
            # Generate facts about a topic
            topic = payload_data.get("topic", payload_data.get("text", ""))
            system = """Generate 15-25 atomic knowledge facts.
Output: subject|predicate|object (one per line, nothing else)
Use lowercase_with_underscores."""
            response = llm.ask(f"Generate facts about: {topic}", system)
            if response:
                count = 0
                for line in response.strip().split("\n"):
                    line = line.strip().strip("-•* ")
                    if "|" not in line: continue
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) >= 3:
                        if db.add(parts[0],parts[1],parts[2],"T",
                            evidence_for=[f"agent research: {topic}"],
                            source="agent:research",
                    normalize=True):
                            count += 1
                db.complete_task(task["id"], f"added {count} facts")
                print(f"    → +{count} facts about '{topic}'")
            else:
                db.complete_task(task["id"], "llm unavailable")
        
        elif task_type == "question" and llm.available:
            question = payload_data.get("question", payload_data.get("text", ""))
            system = """Answer this question by producing atomic facts.
Output: subject|predicate|object|T_or_M (one per line)
Also: CONFUSED: thing_a|thing_b|reason (if applicable)"""
            response = llm.ask(f"Question: {question}", system)
            if response:
                count = 0
                for line in response.strip().split("\n"):
                    if "|" not in line: continue
                    parts = [p.strip() for p in line.split("|")]
                    if line.strip().startswith("CONFUSED") and len(parts) >= 3:
                        db.cursor.execute(
                            "INSERT OR IGNORE INTO confusion_map (subject,confused_with,reason,grade_discovered) VALUES (?,?,?,0)",
                            (parts[0].replace("CONFUSED:","").strip(), parts[1], parts[2] if len(parts)>2 else ""))
                        db.conn.commit()
                    elif len(parts) >= 3:
                        tv = parts[3].upper() if len(parts)>3 and parts[3].upper() in "TFM" else "T"
                        if db.add(parts[0],parts[1],parts[2],tv,
                            evidence_for=[f"agent Q&A: {question}"],
                            source="agent:question",
                    normalize=True):
                            count += 1
                db.complete_task(task["id"], f"added {count} facts")
                print(f"    → +{count} facts from question")
        
        elif task_type == "decompose":
            text = payload_data.get("text", "")
            if llm.available and text:
                facts = llm.decompose_text(text)
                count = 0
                for f in facts:
                    if db.add(f["subject"],f["predicate"],f["object"],"T",
                        evidence_for=["agent decomposition"],
                        source="agent:decompose",
                    normalize=True):
                        count += 1
                db.complete_task(task["id"], f"added {count} facts")
                print(f"    → +{count} facts from decomposition")
            else:
                db.complete_task(task["id"], "llm unavailable or no text")
        
        elif task_type == "verify":
            # Verify a specific entry by ID
            entry_id = payload_data.get("entry_id")
            if entry_id and llm.available:
                entries = db.find(id=int(entry_id))
                if entries:
                    e = entries[0]
                    verdict, reason, new_facts = llm.verify_entry(
                        e["subject"],e["predicate"],e["object"],
                        json.loads(e["evidence_for"]))
                    db.update_truth(e["id"], verdict, f"agent: {reason[:100]}")
                    db.complete_task(task["id"], f"verdict: {verdict}")
                    print(f"    → Verified as {verdict}: {reason[:60]}")
        
        else:
            db.complete_task(task["id"], f"unknown task type: {task_type}")
            print(f"    → Unknown task type: {task_type}")


# ═══════════════════════════════════════════════════════════
# HELPER: Add tasks from command line
# ═══════════════════════════════════════════════════════════

def add_task_cli(db, task_type, text, priority=5):
    """Quick way to add a task from the command line."""
    task_id = db.add_queue_task(task_type, text, priority)
    print(f"  Added task #{task_id}: {task_type} — {text[:60]}")
    return task_id


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Knowledge Ingestion System")
    parser.add_argument("--db", default="curriculum.db")
    parser.add_argument("--model", default=None, help="LLM model name")
    parser.add_argument("--backend", choices=["ollama", "claude"], default="ollama")
    parser.add_argument("--api-key", default=None, help="Claude API key (or set ANTHROPIC_API_KEY)")

    # Ingestion sources
    parser.add_argument("--textbook", help="Ingest a text file")
    parser.add_argument("--textbook-dir", help="Ingest all .txt files in directory")
    parser.add_argument("--wikipedia", help="Ingest Wikipedia dump")
    parser.add_argument("--wiki-max", type=int, default=1000, help="Max Wikipedia articles")
    parser.add_argument("--wikidata", help="Ingest Wikidata JSON dump")
    parser.add_argument("--wikidata-max", type=int, default=50000, help="Max Wikidata entities")
    parser.add_argument("--conceptnet", help="Ingest ConceptNet CSV")
    parser.add_argument("--conceptnet-max", type=int, default=100000, help="Max ConceptNet edges")
    parser.add_argument("--conceptnet-weight", type=float, default=1.0, help="Min edge weight")
    parser.add_argument("--grade", type=int, default=0, help="Grade level for textbook content")
    
    # Agent
    parser.add_argument("--agent-loop", action="store_true", help="Run agent queue processor")
    parser.add_argument("--add-task", nargs=2, metavar=("TYPE","TEXT"), help="Add a task to queue")
    
    # Info
    parser.add_argument("--stats", action="store_true", help="Show database stats")
    parser.add_argument("--list-sources", action="store_true", help="Show available sources info")
    
    args = parser.parse_args()
    
    print(f"{'═'*60}")
    print(f"  📥 KNOWLEDGE INGESTION SYSTEM")
    print(f"{'═'*60}")
    
    db = KnowledgeDB(args.db)
    db.enable_agent_queue()
    db.enable_ingest_log()
    db.enable_confusion_map()
    llm = create_llm(backend=args.backend, model=args.model, api_key=args.api_key)

    print(f"  Database: {args.db} ({db.count()} entries)")
    print(f"  LLM: {'✓ '+llm.model if llm.available else '✗ offline'}")
    
    if args.list_sources:
        print(f"""
  ═══ AVAILABLE KNOWLEDGE SOURCES ═══
  
  📖 TEXTBOOKS (requires LLM)
     Any .txt file → LLM decomposes into atomic facts
     Usage: --textbook file.txt
            --textbook-dir folder/
     Get free textbooks: https://openstax.org
                         https://www.ck12.org
  
  📰 WIKIPEDIA (requires LLM)
     Full dump: https://dumps.wikimedia.org/enwiki/latest/
     Extract with: pip install wikiextractor
                   wikiextractor dump.xml.bz2 -o wiki_text/
     Usage: --wikipedia wiki_text/
  
  🌐 WIKIDATA (NO LLM needed! Already structured!)
     THE BEST SOURCE — millions of subject-predicate-object triples
     Download: https://dumps.wikimedia.org/wikidatawiki/entities/
     File: latest-all.json.bz2 (~100GB compressed)
     For testing: bzcat latest-all.json.bz2 | head -100000 > sample.json
     Usage: --wikidata sample.json
  
  🧠 CONCEPTNET (NO LLM needed! Already structured!)
     Commonsense knowledge: "fire is hot", "dogs have legs"
     Download: https://s3.amazonaws.com/conceptnet/downloads/2019/edges/
     File: conceptnet-assertions-5.7.0.csv.gz (~400MB)
     Usage: --conceptnet conceptnet-assertions-5.7.0.csv.gz
  
  🤖 AGENT QUEUE
     Other programs add tasks to the database
     Tasks: research, question, verify, decompose
     Usage: --agent-loop (to process)
            --add-task research "quantum computing"
            --add-task question "What is entropy?"
""")
        return
    
    if args.stats:
        total = db.count()
        t = db.count("T"); m = db.count("M"); f = db.count("F")
        print(f"\n  Total: {total} | T={t} F={f} M={m}")
        db.cursor.execute("SELECT source, COUNT(*) FROM entries GROUP BY source ORDER BY COUNT(*) DESC")
        print(f"\n  By source:")
        for src, cnt in db.cursor.fetchall():
            bar = "█" * min(cnt//3, 40)
            print(f"    {src:30s} {cnt:6d} {bar}")
        
        db.cursor.execute("SELECT COUNT(*) FROM agent_queue WHERE status='pending'")
        pending = db.cursor.fetchone()[0]
        print(f"\n  Agent queue: {pending} pending tasks")
        return
    
    # Process commands
    did_something = False
    
    if args.textbook:
        ingest_textbook(db, llm, args.textbook, args.grade)
        did_something = True
    
    if args.textbook_dir:
        ingest_textbook_dir(db, llm, args.textbook_dir, args.grade)
        did_something = True
    
    if args.wikipedia:
        ingest_wikipedia(db, llm, args.wikipedia, args.wiki_max)
        did_something = True
    
    if args.wikidata:
        ingest_wikidata(db, args.wikidata, args.wikidata_max)
        did_something = True
    
    if args.conceptnet:
        ingest_conceptnet(db, args.conceptnet, args.conceptnet_max, 
                         min_weight=args.conceptnet_weight)
        did_something = True
    
    if args.add_task:
        add_task_cli(db, args.add_task[0], args.add_task[1])
        did_something = True
    
    if args.agent_loop:
        try:
            run_agent_loop(db, llm)
        except KeyboardInterrupt:
            print("\n  Agent loop stopped")
        did_something = True
    
    if not did_something:
        print("\n  No action specified. Use --help for options.")
        print("  Quick start:")
        print("    python3 ingest.py --list-sources")
        print("    python3 ingest.py --textbook myfile.txt")
        print("    python3 ingest.py --add-task research 'solar system'")
    
    print(f"\n  Database: {db.count()} entries")

if __name__ == "__main__":
    main()
