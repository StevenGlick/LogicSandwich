#!/usr/bin/env python3
"""
LOGICSANDWICH LAUNCHER
======================
Tkinter GUI for the Three-Layer Pulse System.
Run with: py -X utf8 launcher.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import subprocess
import threading
import sys
import os
import json

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
SEED_DIR = os.path.join(os.path.dirname(SRC_DIR), "ClaudeInput", "three_layer_pulse_system")

# Known seed files
SEEDS = {
    "AI / ML (150 entries)": os.path.join(SEED_DIR, "ai_kb_big.json"),
    "Water (140 entries)": os.path.join(SEED_DIR, "water_seed.json"),
}


# ═══════════════════════════════════════════════════════════
# PROCESS RUNNER — subprocess with live output
# ═══════════════════════════════════════════════════════════

class ProcessRunner:
    def __init__(self, console, status_callback=None):
        self.console = console
        self.process = None
        self.running = False
        self.status_cb = status_callback

    def run(self, cmd):
        if self.running:
            return
        self.running = True
        if self.status_cb:
            self.status_cb(True)
        t = threading.Thread(target=self._run, args=(cmd,), daemon=True)
        t.start()

    def _run(self, cmd):
        try:
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            self.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1, cwd=SRC_DIR,
                encoding="utf-8", errors="replace",
                creationflags=flags)

            def reader(stream, tag):
                for line in stream:
                    self.console.after(0, self._append, line, tag)
                stream.close()

            t1 = threading.Thread(target=reader, args=(self.process.stdout, "out"), daemon=True)
            t2 = threading.Thread(target=reader, args=(self.process.stderr, "err"), daemon=True)
            t1.start(); t2.start()
            self.process.wait()
            t1.join(timeout=2); t2.join(timeout=2)

            code = self.process.returncode
            self.console.after(0, self._append,
                f"\n--- Finished (exit code {code}) ---\n", "info")
        except Exception as e:
            self.console.after(0, self._append, f"\nError: {e}\n", "err")
        finally:
            self.running = False
            self.process = None
            if self.status_cb:
                self.console.after(0, self.status_cb, False)

    def stop(self):
        if self.process and self.running:
            self.process.terminate()

    def _append(self, text, tag):
        self.console.config(state="normal")
        self.console.insert("end", text, tag)
        self.console.see("end")
        self.console.config(state="disabled")


# ═══════════════════════════════════════════════════════════
# LAUNCHER APP
# ═══════════════════════════════════════════════════════════

class LauncherApp:
    def __init__(self, root):
        self.root = root
        root.title("LogicSandwich -- Three-Layer Pulse System")
        root.minsize(920, 700)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        # Variables
        self.db_var = tk.StringVar(value="curriculum.db")
        self.backend_var = tk.StringVar(value="ollama")
        self.model_var = tk.StringVar(value="llama3.2")
        self.api_key_var = tk.StringVar(value=os.environ.get("ANTHROPIC_API_KEY", ""))
        self.status_var = tk.StringVar(value="Checking...")

        # Build layout
        self._build_global_bar()
        self._build_notebook()
        self._build_console()

        # Process runner
        self.runner = ProcessRunner(self.console, self._on_process_state)

        # Backend change handler
        self.backend_var.trace_add("write", self._on_backend_change)

        # Check status on startup
        self._check_status()

        # Close handler
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Global Bar ────────────────────────────────────────

    def _build_global_bar(self):
        bar = ttk.Frame(self.root, padding=6)
        bar.grid(row=0, column=0, sticky="ew")

        # Row 1: DB + Backend + Model + Status
        row1 = ttk.Frame(bar)
        row1.pack(fill="x")

        ttk.Label(row1, text="Database:").pack(side="left")
        ttk.Entry(row1, textvariable=self.db_var, width=18).pack(side="left", padx=(4, 0))
        ttk.Button(row1, text="Browse", width=6,
                   command=self._browse_db).pack(side="left", padx=(2, 2))
        ttk.Button(row1, text="Reset DB", width=8,
                   command=self._reset_db).pack(side="left", padx=(2, 10))

        ttk.Label(row1, text="Backend:").pack(side="left")
        self.backend_combo = ttk.Combobox(row1, textvariable=self.backend_var, width=8,
                                          values=["ollama", "claude"], state="readonly")
        self.backend_combo.pack(side="left", padx=(4, 10))

        ttk.Label(row1, text="Model:").pack(side="left")
        self.model_combo = ttk.Combobox(row1, textvariable=self.model_var, width=22,
                                        values=["llama3.2", "mistral", "phi3", "gemma2"])
        self.model_combo.pack(side="left", padx=(4, 10))

        self.status_label = ttk.Label(row1, textvariable=self.status_var)
        self.status_label.pack(side="left", padx=(4, 4))
        ttk.Button(row1, text="Check", width=6,
                   command=self._check_status).pack(side="left")

        # Row 2: API key (shown for Claude backend)
        self.api_row = ttk.Frame(bar)

        ttk.Label(self.api_row, text="API Key:").pack(side="left")
        self.api_entry = ttk.Entry(self.api_row, textvariable=self.api_key_var, width=50, show="*")
        self.api_entry.pack(side="left", padx=(4, 4))
        ttk.Button(self.api_row, text="Show/Hide", width=9,
                   command=self._toggle_key_visibility).pack(side="left")
        if self.api_key_var.get():
            ttk.Label(self.api_row, text="(from ANTHROPIC_API_KEY)",
                      foreground="gray").pack(side="left", padx=(8, 0))

    def _browse_db(self):
        path = filedialog.askopenfilename(
            initialdir=SRC_DIR, filetypes=[("SQLite DB", "*.db"), ("All", "*.*")])
        if path:
            # Use relative if inside SRC_DIR
            try:
                rel = os.path.relpath(path, SRC_DIR)
                if not rel.startswith(".."):
                    path = rel
            except ValueError:
                pass
            self.db_var.set(path)

    def _reset_db(self):
        """Delete the current DB and re-seed from a chosen template."""
        # Build choices
        choices = list(SEEDS.keys()) + ["Empty (no seed)"]
        win = tk.Toplevel(self.root)
        win.title("Reset Database")
        win.geometry("350x260")
        win.transient(self.root)
        win.grab_set()

        db_name = self.db_var.get()
        db_path = os.path.join(SRC_DIR, db_name) if not os.path.isabs(db_name) else db_name

        ttk.Label(win, text=f"This will delete '{db_name}' and create\n"
                  "a fresh database from a seed template.",
                  wraplength=300, justify="center").pack(pady=(16, 12))

        ttk.Label(win, text="Choose seed:").pack(anchor="w", padx=20)
        seed_var = tk.StringVar(value=choices[0])
        for c in choices:
            ttk.Radiobutton(win, text=c, variable=seed_var, value=c).pack(anchor="w", padx=40)

        def do_reset():
            choice = seed_var.get()
            # Confirm
            if not messagebox.askyesno("Confirm Reset",
                    f"Delete '{db_name}' and re-seed with:\n{choice}?",
                    parent=win):
                return
            win.destroy()
            # Delete the DB file
            if os.path.exists(db_path):
                try:
                    os.remove(db_path)
                    self._console_write(f"Deleted {db_name}\n")
                except OSError as e:
                    self._console_write(f"Error deleting DB: {e}\n")
                    return
            # Re-seed if a template was chosen
            if choice != "Empty (no seed)":
                seed_path = SEEDS[choice]
                cmd = self._base_cmd("pulse_production.py")
                cmd += ["--seed-json", seed_path, "--cycles", "0"]
                self._console_write(f"Seeding from {choice}...\n")
                self.runner.run(cmd)
            else:
                self._console_write("Database reset. Use Pulse or Brain tab to seed.\n")

        btn_frame = ttk.Frame(win)
        btn_frame.pack(pady=12)
        ttk.Button(btn_frame, text="Reset", command=do_reset).pack(side="left", padx=8)
        ttk.Button(btn_frame, text="Cancel", command=win.destroy).pack(side="left", padx=8)

    def _toggle_key_visibility(self):
        current = self.api_entry.cget("show")
        self.api_entry.config(show="" if current == "*" else "*")

    def _on_backend_change(self, *args):
        backend = self.backend_var.get()
        if backend == "claude":
            self.api_row.pack(fill="x", pady=(4, 0))
            self.model_combo["values"] = [
                "claude-sonnet-4-6", "claude-haiku-4-5-20251001", "claude-opus-4-6"]
            if not self.model_var.get().startswith("claude"):
                self.model_var.set("claude-sonnet-4-6")
        else:
            self.api_row.pack_forget()
            self.model_combo["values"] = ["llama3.2", "mistral", "phi3", "gemma2"]
            if self.model_var.get().startswith("claude"):
                self.model_var.set("llama3.2")
        self._check_status()

    def _check_status(self):
        backend = self.backend_var.get()
        self.status_var.set("Checking...")
        def _do():
            try:
                import requests as req
                if backend == "claude":
                    key = self.api_key_var.get()
                    if not key:
                        self.root.after(0, self._status_update, False, "No API key")
                        return
                    r = req.post("https://api.anthropic.com/v1/messages",
                        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                                 "content-type": "application/json"},
                        json={"model": self.model_var.get(), "max_tokens": 16,
                              "messages": [{"role": "user", "content": "ping"}]},
                        timeout=10)
                    if r.status_code == 200:
                        self.root.after(0, self._status_update, True, "Claude OK")
                    elif r.status_code == 401:
                        self.root.after(0, self._status_update, False, "Bad API key")
                    else:
                        self.root.after(0, self._status_update, False, f"Error {r.status_code}")
                else:
                    r = req.get("http://localhost:11434/api/tags", timeout=3)
                    if r.status_code == 200:
                        models = [m["name"] for m in r.json().get("models", [])]
                        self.root.after(0, self._status_update, True, "Ollama OK", models)
                    else:
                        self.root.after(0, self._status_update, False, "Not responding")
            except ImportError:
                self.root.after(0, self._status_update, False, "requests not installed")
            except Exception:
                self.root.after(0, self._status_update, False, "Offline")
        threading.Thread(target=_do, daemon=True).start()

    def _status_update(self, ok, msg, models=None):
        self.status_var.set(msg)
        self.status_label.config(foreground="green" if ok else "red")
        if models and self.backend_var.get() == "ollama":
            self.model_combo["values"] = models

    # ── Notebook ──────────────────────────────────────────

    def _build_notebook(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=1, column=0, sticky="nsew", padx=6)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_change)

        self.tabs = {}
        self._build_tab_start()
        self._build_tab_brain()
        self._build_tab_pulse()
        self._build_tab_ingest()
        self._build_tab_query()
        self._build_tab_optimize()
        self._build_tab_test()

    def _make_tab(self, name):
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text=f"  {name}  ")
        self.tabs[name] = frame
        return frame

    def _desc(self, parent, text):
        lbl = ttk.Label(parent, text=text, wraplength=700, justify="left",
                        font=("Segoe UI", 9, "italic"))
        lbl.pack(anchor="w", pady=(0, 10))

    # ── Tab: Getting Started ──────────────────────────────

    def _build_tab_start(self):
        f = self._make_tab("Getting Started")
        txt = (
            "LOGICSANDWICH  --  THREE-LAYER PULSE SYSTEM\n\n"
            "A knowledge engine that stores facts as subject-predicate-object triples,\n"
            "discovers patterns through symbolic logic, and verifies them with a local LLM.\n\n"
            "TYPICAL WORKFLOW:\n"
            "  1. SEED    -- Load initial knowledge (seed JSON or a topic)\n"
            "  2. PULSE   -- Run pulse cycles to discover patterns\n"
            "  3. OPTIMIZE -- Clean predicates, build search indexes\n"
            "  4. QUERY   -- Ask questions of the knowledge base\n\n"
            "QUICK START (first time):\n"
            "  1. Check that Ollama shows 'Connected' above\n"
            "  2. Go to the Pulse tab\n"
            "  3. Click 'Water seed' or 'AI/ML seed'\n"
            "  4. Set Cycles to 5 for a quick test\n"
            "  5. Click 'Preview' to see the command, then 'Run'\n"
            "  6. Watch it discover patterns in the output below\n"
            "  7. Then try the Query tab to explore what it found\n\n"
            "TABS:\n"
            "  Brain    -- Full K-12 curriculum (autonomous learning)\n"
            "  Pulse    -- Standalone pulse cycles (quick experiments)\n"
            "  Ingest   -- Feed knowledge from files or the web\n"
            "  Query    -- Ask questions, browse the graph\n"
            "  Optimize -- Database cleanup and indexing\n"
            "  Test     -- Run the integration test"
        )
        lbl = ttk.Label(f, text=txt, justify="left", font=("Consolas", 9))
        lbl.pack(anchor="w", fill="both", expand=True)

        bf = ttk.Frame(f)
        bf.pack(anchor="w", pady=(10, 0))
        ttk.Button(bf, text="Show DB Stats", command=self._show_stats).pack(side="left")

    def _show_stats(self):
        cmd = self._base_cmd("query.py") + ["--stats"]
        self._preview_and_run(cmd)

    # ── Tab: Brain (curriculum_pulse.py) ──────────────────

    def _build_tab_brain(self):
        f = self._make_tab("Brain")
        self._desc(f, "Learns K-12 curriculum progressively. Each grade builds on the last. "
                      "Use this for autonomous knowledge building from scratch.")

        grid = ttk.Frame(f)
        grid.pack(anchor="w")

        self.brain_start = tk.IntVar(value=0)
        self.brain_end = tk.IntVar(value=13)
        self.brain_ppg = tk.IntVar(value=5)
        self.brain_seed = tk.StringVar()
        self.brain_dash = tk.BooleanVar(value=False)
        self.brain_port = tk.IntVar(value=5000)

        r = 0
        for label, var, rng in [
            ("Start Grade:", self.brain_start, (0, 13)),
            ("End Grade:", self.brain_end, (0, 13)),
            ("Pulses / Grade:", self.brain_ppg, (1, 50)),
        ]:
            ttk.Label(grid, text=label).grid(row=r, column=0, sticky="w", padx=(0, 8), pady=2)
            ttk.Spinbox(grid, from_=rng[0], to=rng[1], textvariable=var, width=6).grid(
                row=r, column=1, sticky="w", pady=2)
            r += 1

        ttk.Label(grid, text="Seed JSON:").grid(row=r, column=0, sticky="w", padx=(0, 8), pady=2)
        ef = ttk.Frame(grid)
        ef.grid(row=r, column=1, columnspan=2, sticky="w", pady=2)
        ttk.Entry(ef, textvariable=self.brain_seed, width=40).pack(side="left")
        ttk.Button(ef, text="Browse", width=7,
                   command=lambda: self._browse_json(self.brain_seed)).pack(side="left", padx=4)
        r += 1

        ttk.Checkbutton(grid, text="Enable Web Dashboard", variable=self.brain_dash).grid(
            row=r, column=0, columnspan=2, sticky="w", pady=2)
        r += 1
        ttk.Label(grid, text="Dashboard Port:").grid(row=r, column=0, sticky="w", padx=(0, 8), pady=2)
        ttk.Spinbox(grid, from_=1024, to=65535, textvariable=self.brain_port, width=6).grid(
            row=r, column=1, sticky="w", pady=2)

    def _get_brain_cmd(self):
        cmd = self._base_cmd("curriculum_pulse.py")
        cmd += ["--start-grade", str(self.brain_start.get())]
        cmd += ["--end-grade", str(self.brain_end.get())]
        cmd += ["--pulses-per-grade", str(self.brain_ppg.get())]
        if self.brain_seed.get():
            cmd += ["--seed-json", self.brain_seed.get()]
        if self.brain_dash.get():
            cmd += ["--dashboard", "--dashboard-port", str(self.brain_port.get())]
        return cmd

    # ── Tab: Pulse (pulse_production.py) ──────────────────

    def _build_tab_pulse(self):
        f = self._make_tab("Pulse")
        self._desc(f, "Runs pulse cycles on existing knowledge. Good for quick experiments. "
                      "Pick a seed file or type a topic to get started.")

        # Quick picks
        qf = ttk.LabelFrame(f, text="Quick Start Seeds", padding=6)
        qf.pack(anchor="w", fill="x", pady=(0, 8))
        self.pulse_seed = tk.StringVar()
        self.pulse_topic = tk.StringVar()
        for label, path in SEEDS.items():
            ttk.Button(qf, text=label,
                       command=lambda p=path: self.pulse_seed.set(p)).pack(side="left", padx=4)

        grid = ttk.Frame(f)
        grid.pack(anchor="w")

        self.pulse_cycles = tk.IntVar(value=20)
        self.pulse_verify = tk.IntVar(value=10)

        r = 0
        ttk.Label(grid, text="Seed JSON:").grid(row=r, column=0, sticky="w", padx=(0, 8), pady=2)
        sf = ttk.Frame(grid)
        sf.grid(row=r, column=1, columnspan=2, sticky="w", pady=2)
        ttk.Entry(sf, textvariable=self.pulse_seed, width=45).pack(side="left")
        ttk.Button(sf, text="Browse", width=7,
                   command=lambda: self._browse_json(self.pulse_seed)).pack(side="left", padx=4)
        r += 1

        ttk.Label(grid, text="-- OR --").grid(row=r, column=0, columnspan=2, sticky="w", pady=4)
        r += 1

        ttk.Label(grid, text="Topic:").grid(row=r, column=0, sticky="w", padx=(0, 8), pady=2)
        ttk.Entry(grid, textvariable=self.pulse_topic, width=40).grid(
            row=r, column=1, sticky="w", pady=2)
        r += 1

        ttk.Separator(grid, orient="horizontal").grid(row=r, column=0, columnspan=3, sticky="ew", pady=8)
        r += 1

        for label, var, rng in [
            ("Cycles:", self.pulse_cycles, (1, 1000)),
            ("Verify Batch:", self.pulse_verify, (0, 100)),
        ]:
            ttk.Label(grid, text=label).grid(row=r, column=0, sticky="w", padx=(0, 8), pady=2)
            ttk.Spinbox(grid, from_=rng[0], to=rng[1], textvariable=var, width=6).grid(
                row=r, column=1, sticky="w", pady=2)
            r += 1

    def _get_pulse_cmd(self):
        cmd = self._base_cmd("pulse_production.py")
        cmd += ["--cycles", str(self.pulse_cycles.get())]
        cmd += ["--verify-batch", str(self.pulse_verify.get())]
        if self.pulse_seed.get():
            cmd += ["--seed-json", self.pulse_seed.get()]
        elif self.pulse_topic.get():
            cmd += ["--topic", self.pulse_topic.get()]
        return cmd

    # ── Tab: Ingest (ingest.py) ───────────────────────────

    def _build_tab_ingest(self):
        f = self._make_tab("Ingest")
        self._desc(f, "Feeds knowledge from external sources into the database. "
                      "Use seed files for quick starts, or large dumps for broad coverage.")

        # Source selection
        sf = ttk.LabelFrame(f, text="Source", padding=6)
        sf.pack(anchor="w", fill="x", pady=(0, 8))

        self.ingest_source = tk.StringVar(value="textbook")
        sources = [("Textbook File", "textbook"), ("Text Directory", "textbook-dir"),
                   ("Wikipedia", "wikipedia"), ("Wikidata", "wikidata"),
                   ("ConceptNet", "conceptnet"), ("Agent Loop", "agent-loop")]
        for label, val in sources:
            ttk.Radiobutton(sf, text=label, variable=self.ingest_source, value=val).pack(
                side="left", padx=4)

        # Path
        pf = ttk.Frame(f)
        pf.pack(anchor="w", fill="x", pady=(0, 8))
        ttk.Label(pf, text="Path:").pack(side="left")
        self.ingest_path = tk.StringVar()
        ttk.Entry(pf, textvariable=self.ingest_path, width=50).pack(side="left", padx=4)
        ttk.Button(pf, text="Browse", width=7, command=self._browse_ingest).pack(side="left")

        # Options
        of = ttk.LabelFrame(f, text="Options", padding=6)
        of.pack(anchor="w", fill="x", pady=(0, 8))

        og = ttk.Frame(of)
        og.pack(anchor="w")
        self.ingest_grade = tk.IntVar(value=0)
        self.ingest_max = tk.IntVar(value=50000)
        self.ingest_weight = tk.DoubleVar(value=1.0)

        ttk.Label(og, text="Grade level:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Spinbox(og, from_=0, to=13, textvariable=self.ingest_grade, width=6).grid(
            row=0, column=1, sticky="w")
        ttk.Label(og, text="Max items:").grid(row=0, column=2, sticky="w", padx=(16, 8))
        ttk.Spinbox(og, from_=100, to=1000000, textvariable=self.ingest_max, width=10).grid(
            row=0, column=3, sticky="w")
        ttk.Label(og, text="Min weight:").grid(row=0, column=4, sticky="w", padx=(16, 8))
        ttk.Spinbox(og, from_=0.0, to=10.0, increment=0.1,
                     textvariable=self.ingest_weight, width=6).grid(row=0, column=5, sticky="w")

        # Quick actions
        qf = ttk.Frame(f)
        qf.pack(anchor="w", pady=(4, 0))
        ttk.Button(qf, text="Show Stats",
                   command=lambda: self._preview_and_run(
                       self._base_cmd("ingest.py") + ["--stats"])).pack(side="left", padx=4)
        ttk.Button(qf, text="List Sources",
                   command=lambda: self._preview_and_run(
                       self._base_cmd("ingest.py") + ["--list-sources"])).pack(side="left", padx=4)

    def _browse_ingest(self):
        src = self.ingest_source.get()
        if src == "textbook-dir":
            path = filedialog.askdirectory(initialdir=SRC_DIR)
        elif src == "textbook":
            path = filedialog.askopenfilename(initialdir=SRC_DIR,
                filetypes=[("Text files", "*.txt"), ("All", "*.*")])
        elif src in ("wikipedia", "wikidata"):
            path = filedialog.askopenfilename(initialdir=SRC_DIR,
                filetypes=[("JSON/XML", "*.json *.xml *.bz2"), ("All", "*.*")])
        elif src == "conceptnet":
            path = filedialog.askopenfilename(initialdir=SRC_DIR,
                filetypes=[("CSV", "*.csv *.csv.gz *.tsv"), ("All", "*.*")])
        else:
            return
        if path:
            self.ingest_path.set(path)

    def _get_ingest_cmd(self):
        cmd = self._base_cmd("ingest.py")
        src = self.ingest_source.get()

        if src == "agent-loop":
            cmd += ["--agent-loop"]
            return cmd

        path = self.ingest_path.get()
        if not path:
            return cmd + ["--stats"]  # fallback: just show stats

        cmd += [f"--{src}", path]

        if src == "textbook":
            cmd += ["--grade", str(self.ingest_grade.get())]
        elif src == "wikipedia":
            cmd += ["--wiki-max", str(self.ingest_max.get())]
        elif src == "wikidata":
            cmd += ["--wikidata-max", str(self.ingest_max.get())]
        elif src == "conceptnet":
            cmd += ["--conceptnet-max", str(self.ingest_max.get())]
            cmd += ["--conceptnet-weight", str(self.ingest_weight.get())]

        return cmd

    # ── Tab: Query (query.py) ─────────────────────────────

    def _build_tab_query(self):
        f = self._make_tab("Query")
        self._desc(f, "Ask questions of the knowledge base. Every claim traces back to "
                      "a specific entry -- no hallucination, only verified knowledge.")

        # Mode selection
        mf = ttk.LabelFrame(f, text="Mode", padding=6)
        mf.pack(anchor="w", fill="x", pady=(0, 8))

        self.query_mode = tk.StringVar(value="question")
        modes = [("Ask Question", "question"), ("Interactive Chat", "interactive"),
                 ("Raw Lookup", "raw"), ("Show Graph", "graph"),
                 ("Find Path", "path"), ("Stats", "stats"),
                 ("Confusions", "confusions"), ("Unresolved", "unresolved")]
        for label, val in modes:
            ttk.Radiobutton(mf, text=label, variable=self.query_mode, value=val).pack(
                side="left", padx=4)

        # Input area
        inf = ttk.Frame(f)
        inf.pack(anchor="w", fill="x", pady=(0, 8))

        self.query_input = tk.StringVar()
        self.query_input2 = tk.StringVar()
        self.query_depth = tk.IntVar(value=2)

        ttk.Label(inf, text="Input:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(inf, textvariable=self.query_input, width=50).grid(
            row=0, column=1, sticky="w", pady=2)

        ttk.Label(inf, text="To (path):").grid(row=1, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(inf, textvariable=self.query_input2, width=30).grid(
            row=1, column=1, sticky="w", pady=2)

        ttk.Label(inf, text="Depth:").grid(row=2, column=0, sticky="w", padx=(0, 8))
        ttk.Spinbox(inf, from_=1, to=10, textvariable=self.query_depth, width=6).grid(
            row=2, column=1, sticky="w", pady=2)

    def _get_query_cmd(self):
        cmd = self._base_cmd("query.py")
        mode = self.query_mode.get()
        inp = self.query_input.get().strip()

        if mode == "question" and inp:
            cmd += [inp]
        elif mode == "interactive":
            # Special: open in new terminal
            return self._interactive_cmd()
        elif mode == "raw" and inp:
            cmd += ["--raw", inp]
        elif mode == "graph" and inp:
            cmd += ["--graph", inp, str(self.query_depth.get())]
        elif mode == "path" and inp:
            inp2 = self.query_input2.get().strip()
            if inp2:
                cmd += ["--path", inp, inp2]
        elif mode == "stats":
            cmd += ["--stats"]
        elif mode == "confusions":
            cmd += ["--confusions"]
        elif mode == "unresolved":
            cmd += ["--unresolved"]

        return cmd

    def _interactive_cmd(self):
        """Build command that opens query.py --interactive in a new console."""
        return None  # Handled specially in run

    # ── Tab: Optimize (optimize.py) ───────────────────────

    def _build_tab_optimize(self):
        f = self._make_tab("Optimize")
        self._desc(f, "Database cleanup and optimization. Run after big ingests, "
                      "or periodically. Like defragmenting -- makes everything faster.")

        cf = ttk.LabelFrame(f, text="Operations", padding=6)
        cf.pack(anchor="w", fill="x", pady=(0, 8))

        self.opt_vars = {}
        ops = ["normalize", "hierarchy", "centrality", "fts",
               "synonyms", "domains", "dedup", "integrity"]
        labels = ["Normalize Predicates", "Build Hierarchy", "Compute Centrality",
                  "Full-Text Search Index", "Synonym Table", "Domain Clustering",
                  "Deduplicate", "Integrity Check"]

        for i, (op, label) in enumerate(zip(ops, labels)):
            var = tk.BooleanVar(value=False)
            self.opt_vars[op] = var
            ttk.Checkbutton(cf, text=label, variable=var).grid(
                row=i // 2, column=i % 2, sticky="w", padx=8, pady=2)

        bf = ttk.Frame(f)
        bf.pack(anchor="w", pady=(4, 0))
        ttk.Button(bf, text="Select All", command=self._opt_select_all).pack(side="left", padx=4)
        ttk.Button(bf, text="Deselect All", command=self._opt_deselect_all).pack(side="left", padx=4)
        ttk.Button(bf, text="Show Report",
                   command=lambda: self._preview_and_run(
                       self._base_cmd("optimize.py") + ["--report"])).pack(side="left", padx=16)

    def _opt_select_all(self):
        for v in self.opt_vars.values():
            v.set(True)

    def _opt_deselect_all(self):
        for v in self.opt_vars.values():
            v.set(False)

    def _get_optimize_cmd(self):
        cmd = self._base_cmd("optimize.py")
        selected = [k for k, v in self.opt_vars.items() if v.get()]
        if not selected:
            cmd += ["--all"]
        else:
            for op in selected:
                cmd += [f"--{op}"]
        return cmd

    # ── Tab: Test (test_water.py) ─────────────────────────

    def _build_tab_test(self):
        f = self._make_tab("Test")
        self._desc(f, "Runs the full integration test on the water domain (140 seed entries). "
                      "No LLM needed -- tests CPU-side operations only. "
                      "Creates its own test_water.db, separate from the main database.")
        ttk.Label(f, text=(
            "Expected results:\n"
            "  Seed: 140 -> After pulse: 472 (+332)\n"
            "  Transitive blocked: 39\n"
            "  Bridge types: 14 of 25"
        ), font=("Consolas", 9), justify="left").pack(anchor="w", pady=(0, 10))

    def _get_test_cmd(self):
        return [sys.executable, "-X", "utf8", "test_water.py"]

    # ── Console ───────────────────────────────────────────

    def _build_console(self):
        cf = ttk.Frame(self.root, padding=6)
        cf.grid(row=2, column=0, sticky="nsew")
        self.root.rowconfigure(2, weight=0, minsize=220)

        # Button bar
        bf = ttk.Frame(cf)
        bf.pack(fill="x", pady=(0, 4))

        self.btn_preview = ttk.Button(bf, text="Preview Command", command=self._preview)
        self.btn_preview.pack(side="left", padx=2)
        self.btn_run = ttk.Button(bf, text="Run", command=self._run, width=8)
        self.btn_run.pack(side="left", padx=2)
        self.btn_stop = ttk.Button(bf, text="Stop", command=self._stop, width=8, state="disabled")
        self.btn_stop.pack(side="left", padx=2)
        ttk.Button(bf, text="Clear", command=self._clear, width=8).pack(side="left", padx=2)

        # Console output
        self.console = scrolledtext.ScrolledText(
            cf, height=12, state="disabled", wrap="word",
            font=("Consolas", 9), bg="#1a1a1a", fg="#e0e0e0",
            insertbackground="#e0e0e0", selectbackground="#336")
        self.console.pack(fill="both", expand=True)
        self.console.tag_config("cmd", foreground="#5599ff")
        self.console.tag_config("out", foreground="#e0e0e0")
        self.console.tag_config("err", foreground="#ff6666")
        self.console.tag_config("info", foreground="#888888")

    def _on_process_state(self, running):
        if running:
            self.btn_run.config(state="disabled", text="Running...")
            self.btn_stop.config(state="normal")
        else:
            self.btn_run.config(state="normal", text="Run")
            self.btn_stop.config(state="disabled")

    # ── Command Building ──────────────────────────────────

    def _base_cmd(self, script):
        cmd = [sys.executable, "-X", "utf8", script]
        cmd += ["--db", self.db_var.get()]
        # Only add LLM args if the script supports them
        if script not in ("optimize.py", "test_water.py"):
            model = self.model_var.get()
            if model:
                cmd += ["--model", model]
            backend = self.backend_var.get()
            cmd += ["--backend", backend]
            if backend == "claude":
                key = self.api_key_var.get()
                if key:
                    cmd += ["--api-key", key]
        return cmd

    def _current_tab_name(self):
        idx = self.notebook.index(self.notebook.select())
        return list(self.tabs.keys())[idx]

    def _build_command(self):
        tab = self._current_tab_name()
        builders = {
            "Brain": self._get_brain_cmd,
            "Pulse": self._get_pulse_cmd,
            "Ingest": self._get_ingest_cmd,
            "Query": self._get_query_cmd,
            "Optimize": self._get_optimize_cmd,
            "Test": self._get_test_cmd,
        }
        builder = builders.get(tab)
        if builder:
            return builder()
        return None

    def _on_tab_change(self, event=None):
        tab = self._current_tab_name()
        can_run = tab != "Getting Started"
        state = "normal" if can_run and not self.runner.running else "disabled"
        self.btn_run.config(state=state)
        self.btn_preview.config(state="normal" if can_run else "disabled")

    # ── Run / Preview / Stop ──────────────────────────────

    def _console_write(self, text, tag="info"):
        self.console.config(state="normal")
        self.console.insert("end", text, tag)
        self.console.see("end")
        self.console.config(state="disabled")

    def _preview(self):
        cmd = self._build_command()
        if not cmd:
            self._console_write("(No command for this tab)\n", "info")
            return
        display = " ".join(cmd)
        self._console_write(f"> {display}\n", "cmd")

    def _run(self):
        tab = self._current_tab_name()

        # Special case: interactive query opens a new console
        if tab == "Query" and self.query_mode.get() == "interactive":
            self._run_interactive()
            return

        cmd = self._build_command()
        if not cmd:
            self._console_write("(No command for this tab)\n", "info")
            return
        self._console_write(f"> {' '.join(cmd)}\n", "cmd")
        self.runner.run(cmd)

    def _run_interactive(self):
        """Launch query.py --interactive in a new terminal window."""
        backend = self.backend_var.get()
        model = self.model_var.get()
        cmd_str = (f'"{sys.executable}" -X utf8 query.py '
                   f'--db "{self.db_var.get()}" '
                   f'--backend {backend} ')
        if model:
            cmd_str += f'--model "{model}" '
        if backend == "claude" and self.api_key_var.get():
            cmd_str += f'--api-key "{self.api_key_var.get()}" '
        cmd_str += '--interactive'
        if sys.platform == "win32":
            os.system(f'start cmd /k "cd /d {SRC_DIR} && {cmd_str}"')
        else:
            os.system(f'xterm -e "cd {SRC_DIR} && {cmd_str}" &')
        self._console_write("Opened interactive query in new terminal window.\n", "info")

    def _preview_and_run(self, cmd):
        """Preview then immediately run a command."""
        self._console_write(f"> {' '.join(cmd)}\n", "cmd")
        self.runner.run(cmd)

    def _stop(self):
        self.runner.stop()
        self._console_write("\n--- Stopped ---\n", "info")

    def _clear(self):
        self.console.config(state="normal")
        self.console.delete("1.0", "end")
        self.console.config(state="disabled")

    def _on_close(self):
        if self.runner.running:
            self.runner.stop()
        self.root.destroy()

    # ── Helpers ───────────────────────────────────────────

    def _browse_json(self, var):
        start = SEED_DIR if os.path.isdir(SEED_DIR) else SRC_DIR
        path = filedialog.askopenfilename(
            initialdir=start, filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if path:
            var.set(path)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    root = tk.Tk()
    app = LauncherApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
