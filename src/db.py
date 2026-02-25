"""
SHARED DATABASE MODULE
======================

Single canonical KnowledgeDB class used by all scripts.
Replaces 5 separate copies with incompatible schemas.

Usage:
    from db import KnowledgeDB
    db = KnowledgeDB("curriculum.db")

    # Optional: enable supporting tables as needed
    db.enable_logging()        # for curriculum_pulse.py
    db.enable_confusion_map()  # for curriculum_pulse.py, ingest.py
    db.enable_agent_queue()    # for ingest.py
    db.enable_ingest_log()     # for ingest.py
"""

import sqlite3
import json
import re
from collections import defaultdict, deque


# ── FNV-1a 64-bit hash ────────────────────────────────
def fnv1a_64(data):
    """FNV-1a 64-bit hash. Returns signed int for SQLite compatibility."""
    FNV_OFFSET = 14695981039346656037
    FNV_PRIME = 1099511628211
    h = FNV_OFFSET
    for byte in data.encode('utf-8'):
        h ^= byte
        h = (h * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    # Convert to signed 64-bit for SQLite INTEGER
    if h >= (1 << 63):
        h -= (1 << 64)
    return h

def triple_hash(subject, predicate, obj):
    """Hash a full triple for fast duplicate detection."""
    return fnv1a_64(f"{subject}|{predicate}|{obj}")


class KnowledgeDB:
    """
    Shared knowledge base backed by SQLite.

    Canonical schema includes all columns from every script.
    Supporting tables (log, confusion_map, agent_queue, ingest_log)
    are created on demand via enable_*() methods.
    """

    def __init__(self, db_path="curriculum.db", check_same_thread=False):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=check_same_thread)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        self._enabled_tables = set()
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

            CREATE INDEX IF NOT EXISTS idx_subject ON entries(subject);
            CREATE INDEX IF NOT EXISTS idx_predicate ON entries(predicate);
            CREATE INDEX IF NOT EXISTS idx_object ON entries(object);
            CREATE INDEX IF NOT EXISTS idx_truth ON entries(truth_value);
            CREATE INDEX IF NOT EXISTS idx_source ON entries(source);
            CREATE INDEX IF NOT EXISTS idx_sig ON entries(subject, predicate, object);
            CREATE INDEX IF NOT EXISTS idx_grade ON entries(grade_level);
        """)
        self.conn.commit()
        # Hash column migration — add if not present
        self._migrate_hash_columns()

    def _migrate_hash_columns(self):
        """Add FNV-1a hash columns for fast lookups. Safe to call repeatedly."""
        cols = {r[1] for r in self.cursor.execute("PRAGMA table_info(entries)").fetchall()}
        added = False
        for col in ("subject_hash", "predicate_hash", "object_hash", "triple_hash"):
            if col not in cols:
                self.cursor.execute(f"ALTER TABLE entries ADD COLUMN {col} INTEGER DEFAULT 0")
                added = True
        if added:
            self.conn.commit()
        # Always backfill — catches rows left unhashed from failed migrations
        self._backfill_hashes()
        # Create hash indexes (idempotent)
        self.cursor.executescript("""
            CREATE INDEX IF NOT EXISTS idx_subject_hash ON entries(subject_hash);
            CREATE INDEX IF NOT EXISTS idx_triple_hash ON entries(triple_hash);
            CREATE INDEX IF NOT EXISTS idx_subj_pred_hash ON entries(subject_hash, predicate_hash);
            CREATE INDEX IF NOT EXISTS idx_object_hash ON entries(object_hash);
        """)
        self.conn.commit()

    def _backfill_hashes(self):
        """Compute hashes for existing rows that have hash=0."""
        self.cursor.execute(
            "SELECT id, subject, predicate, object FROM entries WHERE triple_hash=0 OR triple_hash IS NULL")
        rows = self.cursor.fetchall()
        if not rows:
            return
        for r in rows:
            s, p, o = r[1], r[2], r[3]
            self.cursor.execute(
                "UPDATE entries SET subject_hash=?, predicate_hash=?, object_hash=?, triple_hash=? WHERE id=?",
                (fnv1a_64(s), fnv1a_64(p), fnv1a_64(o), triple_hash(s, p, o), r[0]))
        self.conn.commit()

    # ── Table Enablement ──────────────────────────────────

    def _has_table(self, name):
        if name in self._enabled_tables:
            return True
        self.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
        if self.cursor.fetchone():
            self._enabled_tables.add(name)
            return True
        return False

    def enable_logging(self):
        self.cursor.executescript("""
            CREATE TABLE IF NOT EXISTS log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT,
                grade_level INTEGER,
                message TEXT,
                details TEXT DEFAULT ''
            );
        """)
        self.conn.commit()
        self._enabled_tables.add("log")

    def enable_confusion_map(self):
        self.cursor.executescript("""
            CREATE TABLE IF NOT EXISTS confusion_map (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT,
                confused_with TEXT,
                reason TEXT,
                grade_discovered INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self.conn.commit()
        self._enabled_tables.add("confusion_map")

    def enable_agent_queue(self):
        self.cursor.executescript("""
            CREATE TABLE IF NOT EXISTS agent_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                priority INTEGER DEFAULT 5,
                status TEXT DEFAULT 'pending',
                result TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_queue_status ON agent_queue(status);
        """)
        self.conn.commit()
        self._enabled_tables.add("agent_queue")

    def enable_ingest_log(self):
        self.cursor.executescript("""
            CREATE TABLE IF NOT EXISTS ingest_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_type TEXT,
                source_path TEXT,
                chunks_processed INTEGER DEFAULT 0,
                facts_added INTEGER DEFAULT 0,
                errors INTEGER DEFAULT 0,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            );
        """)
        self.conn.commit()
        self._enabled_tables.add("ingest_log")

    # ── Core Methods ──────────────────────────────────────

    def add(self, subject, predicate, obj, truth_value="M",
            evidence_for=None, evidence_against=None,
            source="unknown", generation=0, grade_level=0,
            needs_verification=False, normalize=False):
        """Add a triple. Returns row ID or None if duplicate."""
        if normalize:
            subject = self._normalize(subject)
            predicate = self._normalize(predicate)
            obj = self._normalize(obj)
            if not subject or not predicate or not obj:
                return None
        # Hash-accelerated duplicate detection
        th = triple_hash(subject, predicate, obj)
        self.cursor.execute(
            "SELECT id FROM entries WHERE triple_hash=? AND subject=? AND predicate=? AND object=?",
            (th, subject, predicate, obj))
        if self.cursor.fetchone():
            return None
        sh = fnv1a_64(subject)
        ph = fnv1a_64(predicate)
        oh = fnv1a_64(obj)
        self.cursor.execute("""
            INSERT INTO entries (subject, predicate, object, truth_value,
                evidence_for, evidence_against, source, generation,
                grade_level, needs_verification,
                subject_hash, predicate_hash, object_hash, triple_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (subject, predicate, obj, truth_value,
              json.dumps(evidence_for or []), json.dumps(evidence_against or []),
              source, generation, grade_level, int(needs_verification),
              sh, ph, oh, th))
        self.conn.commit()
        return self.cursor.lastrowid

    def _normalize(self, text):
        """Normalize a term: lowercase, underscores, strip junk."""
        if not text:
            return ""
        text = text.strip().lower()
        text = re.sub(r'[^a-z0-9_\s-]', '', text)
        text = re.sub(r'[\s-]+', '_', text)
        text = text.strip('_')
        return text[:100]

    def find(self, **kwargs):
        """Find entries matching all keyword arguments exactly."""
        conditions = []
        params = []
        for k, v in kwargs.items():
            conditions.append(f"{k}=?")
            params.append(v)
        where = " AND ".join(conditions) if conditions else "1=1"
        self.cursor.execute(f"SELECT * FROM entries WHERE {where}", params)
        return [dict(r) for r in self.cursor.fetchall()]

    def all_entries(self, truth_value=None):
        """Get all entries, optionally filtered by truth value."""
        if truth_value:
            self.cursor.execute(
                "SELECT * FROM entries WHERE truth_value=?", (truth_value,))
        else:
            self.cursor.execute("SELECT * FROM entries")
        return [dict(r) for r in self.cursor.fetchall()]

    def update_truth(self, eid, new_truth, evidence=None):
        """Update an entry's truth value. Marks as verified."""
        self.cursor.execute(
            "UPDATE entries SET truth_value=?, verified=1, needs_verification=0 WHERE id=?",
            (new_truth, eid))
        if evidence:
            row = self.cursor.execute(
                "SELECT * FROM entries WHERE id=?", (eid,)).fetchone()
            if row:
                ev = json.loads(dict(row)["evidence_for"])
                ev.append(evidence)
                self.cursor.execute(
                    "UPDATE entries SET evidence_for=? WHERE id=?",
                    (json.dumps(ev), eid))
        self.conn.commit()

    def count(self, truth_value=None):
        """Count entries, optionally filtered by truth value."""
        if truth_value:
            self.cursor.execute(
                "SELECT COUNT(*) FROM entries WHERE truth_value=?", (truth_value,))
        else:
            self.cursor.execute("SELECT COUNT(*) FROM entries")
        return self.cursor.fetchone()[0]

    def current_generation(self):
        """Get the next generation number."""
        self.cursor.execute("SELECT MAX(generation) FROM entries")
        r = self.cursor.fetchone()[0]
        return (r or 0) + 1

    def signatures(self):
        """Get set of all (subject, predicate, object) tuples."""
        self.cursor.execute("SELECT subject, predicate, object FROM entries")
        return {(r[0], r[1], r[2]) for r in self.cursor.fetchall()}

    def stats(self):
        """Print a summary of the database to stdout."""
        total = self.count()
        t = self.count("T")
        f = self.count("F")
        m = self.count("M")
        self.cursor.execute(
            "SELECT source, COUNT(*) FROM entries GROUP BY source ORDER BY COUNT(*) DESC")
        sources = self.cursor.fetchall()

        print(f"\n  === KB STATS ===")
        print(f"  Total: {total} | T={t} F={f} M={m}")
        print(f"  Sources:")
        for src, cnt in sources:
            bar = "=" * min(cnt, 40)
            print(f"    {src:25s} {cnt:4d} {bar}")

    # ── Query Methods (from query.py) ─────────────────────

    def search(self, term):
        """Find entries where term appears in subject or object."""
        term_norm = term.strip().lower().replace(" ", "_")
        self.cursor.execute("""
            SELECT * FROM entries WHERE
                subject LIKE ? OR object LIKE ? OR subject=? OR object=?
            ORDER BY truth_value, generation
        """, (f"%{term_norm}%", f"%{term_norm}%", term_norm, term_norm))
        return [dict(r) for r in self.cursor.fetchall()]

    def search_exact(self, **kwargs):
        """Exact field match (thin wrapper around find)."""
        return self.find(**kwargs)

    def search_multi(self, terms):
        """Find entries matching ANY of the given terms."""
        results = {}
        for term in terms:
            for entry in self.search(term):
                results[entry["id"]] = entry
        return sorted(results.values(), key=lambda e: e["id"])

    def search_fts(self, terms):
        """Search using FTS5 index if available, with synonym expansion.
        Falls back to LIKE-based search_multi if FTS5 isn't built."""
        # Check if FTS5 table exists
        try:
            self.cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='entries_fts'")
            if not self.cursor.fetchone():
                return self.search_multi(terms)
        except sqlite3.OperationalError:
            return self.search_multi(terms)

        # Expand terms with synonyms
        expanded = set(terms)
        try:
            for term in terms:
                term_norm = term.strip().lower().replace(" ", "_")
                self.cursor.execute(
                    "SELECT synonym FROM synonyms WHERE term=?", (term_norm,))
                for row in self.cursor.fetchall():
                    expanded.add(row[0])
        except sqlite3.OperationalError:
            pass  # synonyms table may not exist

        # Build FTS5 query: term1 OR term2 OR term3
        fts_terms = []
        for t in expanded:
            # FTS5 uses spaces not underscores, escape quotes
            clean = t.strip().lower().replace("_", " ").replace('"', '')
            if clean:
                fts_terms.append(f'"{clean}"')
        if not fts_terms:
            return self.search_multi(terms)

        fts_query = " OR ".join(fts_terms)
        try:
            self.cursor.execute("""
                SELECT entry_id FROM entries_fts WHERE entries_fts MATCH ?
            """, (fts_query,))
            entry_ids = [int(row[0]) for row in self.cursor.fetchall()]
            if not entry_ids:
                return self.search_multi(terms)

            # Fetch the actual entries
            placeholders = ",".join("?" * len(entry_ids))
            self.cursor.execute(
                f"SELECT * FROM entries WHERE id IN ({placeholders}) ORDER BY truth_value, generation",
                entry_ids)
            return [dict(r) for r in self.cursor.fetchall()]
        except sqlite3.OperationalError:
            return self.search_multi(terms)

    def get_neighbors(self, subject, depth=1):
        """BFS graph walk from a subject, up to N hops. Returns unique entries."""
        visited = set()
        frontier = {subject.lower().replace(" ", "_")}
        all_entries = []

        for d in range(depth):
            next_frontier = set()
            for node in frontier:
                if node in visited:
                    continue
                visited.add(node)

                self.cursor.execute(
                    "SELECT * FROM entries WHERE subject=? AND truth_value='T'",
                    (node,))
                for r in self.cursor.fetchall():
                    entry = dict(r)
                    all_entries.append(entry)
                    obj = entry["object"]
                    if "," not in obj and obj not in visited:
                        next_frontier.add(obj)

                self.cursor.execute(
                    "SELECT * FROM entries WHERE object=? AND truth_value='T'",
                    (node,))
                for r in self.cursor.fetchall():
                    entry = dict(r)
                    all_entries.append(entry)
                    subj = entry["subject"]
                    if "," not in subj and subj not in visited:
                        next_frontier.add(subj)

            frontier = next_frontier

        seen = set()
        unique = []
        for e in all_entries:
            key = (e["subject"], e["predicate"], e["object"])
            if key not in seen:
                seen.add(key)
                unique.append(e)
        return unique

    def find_path(self, start, end, max_depth=5):
        """BFS shortest path between two concepts."""
        start = start.lower().replace(" ", "_")
        end = end.lower().replace(" ", "_")

        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            current, path = queue.popleft()
            if len(path) > max_depth:
                continue

            self.cursor.execute(
                "SELECT subject, predicate, object FROM entries WHERE "
                "(subject=? OR object=?) AND truth_value='T'",
                (current, current))

            for row in self.cursor.fetchall():
                s, p, o = row
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

        return None

    def get_about(self, subject):
        """Get direct facts FROM a subject. Hash-accelerated."""
        subject = subject.lower().replace(" ", "_")
        sh = fnv1a_64(subject)
        self.cursor.execute(
            "SELECT * FROM entries WHERE subject_hash=? AND subject=? AND truth_value='T' "
            "AND object NOT LIKE '%,%'",
            (sh, subject))
        return [dict(r) for r in self.cursor.fetchall()]

    def get_reverse(self, obj):
        """What entries point TO this object? Hash-accelerated."""
        obj = obj.lower().replace(" ", "_")
        oh = fnv1a_64(obj)
        self.cursor.execute(
            "SELECT * FROM entries WHERE object_hash=? AND object=? AND truth_value='T' "
            "AND subject NOT LIKE '%,%'",
            (oh, obj))
        return [dict(r) for r in self.cursor.fetchall()]

    def get_unresolved(self, limit=30):
        """Get Maybe-valued entries awaiting verification."""
        self.cursor.execute(
            "SELECT * FROM entries WHERE truth_value='M' LIMIT ?", (limit,))
        return [dict(r) for r in self.cursor.fetchall()]

    # ── Curriculum Methods (need enable_logging/enable_confusion_map) ──

    def log_event(self, event_type, grade, message, details=""):
        if not self._has_table("log"):
            return None
        self.cursor.execute(
            "INSERT INTO log (event_type, grade_level, message, details) VALUES (?,?,?,?)",
            (event_type, grade, message, details))
        self.conn.commit()

    def add_confusion(self, subject, confused_with, reason, grade):
        if not self._has_table("confusion_map"):
            return False
        self.cursor.execute(
            "SELECT id FROM confusion_map WHERE subject=? AND confused_with=?",
            (subject, confused_with))
        if not self.cursor.fetchone():
            self.cursor.execute(
                "INSERT INTO confusion_map (subject, confused_with, reason, grade_discovered) "
                "VALUES (?,?,?,?)",
                (subject, confused_with, reason, grade))
            self.conn.commit()
            return True
        return False

    def get_log(self, limit=50):
        if not self._has_table("log"):
            return []
        self.cursor.execute("SELECT * FROM log ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) for r in self.cursor.fetchall()]

    def get_confusions(self):
        if not self._has_table("confusion_map"):
            return []
        self.cursor.execute("SELECT * FROM confusion_map ORDER BY grade_discovered")
        return [dict(r) for r in self.cursor.fetchall()]

    def stats_by_grade(self):
        self.cursor.execute("""
            SELECT grade_level, truth_value, COUNT(*)
            FROM entries GROUP BY grade_level, truth_value
            ORDER BY grade_level
        """)
        result = defaultdict(lambda: {"T": 0, "F": 0, "M": 0})
        for row in self.cursor.fetchall():
            result[row[0]][row[1]] = row[2]
        return dict(result)

    def stats_by_source(self):
        self.cursor.execute("""
            SELECT source, COUNT(*) FROM entries
            GROUP BY source ORDER BY COUNT(*) DESC
        """)
        return [(r[0], r[1]) for r in self.cursor.fetchall()]

    # ── Ingest Methods (need enable_agent_queue/enable_ingest_log) ──

    def add_queue_task(self, task_type, payload, priority=5):
        if not self._has_table("agent_queue"):
            return None
        self.cursor.execute(
            "INSERT INTO agent_queue (task_type, payload, priority) VALUES (?,?,?)",
            (task_type, json.dumps(payload) if isinstance(payload, dict) else payload,
             priority))
        self.conn.commit()
        return self.cursor.lastrowid

    def get_next_task(self):
        if not self._has_table("agent_queue"):
            return None
        self.cursor.execute(
            "SELECT * FROM agent_queue WHERE status='pending' "
            "ORDER BY priority DESC, id ASC LIMIT 1")
        row = self.cursor.fetchone()
        if row:
            self.cursor.execute(
                "UPDATE agent_queue SET status='processing' WHERE id=?",
                (row['id'],))
            self.conn.commit()
            return dict(row)
        return None

    def complete_task(self, task_id, result=""):
        if not self._has_table("agent_queue"):
            return
        self.cursor.execute(
            "UPDATE agent_queue SET status='done', result=?, "
            "completed_at=CURRENT_TIMESTAMP WHERE id=?",
            (result, task_id))
        self.conn.commit()

    def log_ingest(self, source_type, source_path, chunks, facts, errors):
        if not self._has_table("ingest_log"):
            return
        self.cursor.execute("""
            INSERT INTO ingest_log (source_type, source_path, chunks_processed,
                facts_added, errors, completed_at)
            VALUES (?,?,?,?,?,CURRENT_TIMESTAMP)
        """, (source_type, source_path, chunks, facts, errors))
        self.conn.commit()
