"""
runner.py — BatterySwapAI 2026 Pipeline Runner
Desktop GUI (tkinter) — runs natively, no browser required.

Usage:
    python runner.py
"""

import os
import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import scrolledtext

REPO_ROOT = Path(__file__).parent
BASE      = REPO_ROOT / "battery_swap_ai_2026"

STAGES = [
    {
        "id":    "s1",
        "label": "1.  Generate Data",
        "desc":  "500 sensors · 50 buildings · 6 Norwegian cities",
        "cmd":   [sys.executable, str(BASE / "data/raw/generate_dummy_data.py")],
    },
    {
        "id":    "s2",
        "label": "2.  Feature Engineering",
        "desc":  "30+ rolling voltage / temperature features",
        "cmd":   [sys.executable, str(BASE / "model/feature_pipeline.py")],
    },
    {
        "id":    "s3",
        "label": "3.  Baseline Model",
        "desc":  "Linear RUL extrapolation · reference MAE",
        "cmd":   [sys.executable, str(BASE / "model/baseline.py")],
    },
    {
        "id":    "s4",
        "label": "4.  Train LightGBM",
        "desc":  "RUL regression + isotonic calibration by building type",
        "cmd":   [sys.executable, str(BASE / "model/train.py")],
    },
    {
        "id":    "s5",
        "label": "5.  Uncertainty Quantification",
        "desc":  "Quantile intervals · p_fail_3d / 7d / 14d ∈ [0, 1]",
        "cmd":   [sys.executable, str(BASE / "model/uncertainty.py")],
    },
    {
        "id":    "s6",
        "label": "6.  Risk & Priority Scoring",
        "desc":  "DEAD / CRITICAL / WARNING / SAFE · building-type weights",
        "cmd":   [sys.executable, str(BASE / "optimization/priority.py")],
    },
    {
        "id":    "s7",
        "label": "7.  VRP Scheduler",
        "desc":  "OR-Tools · 3 workers · 480-min shift · unreachable detection",
        "cmd":   [sys.executable, str(BASE / "optimization/scheduler.py")],
    },
    {
        "id":    "s8",
        "label": "8.  Cost Simulation",
        "desc":  "AGGRESSIVE / NORMAL / CONSERVATIVE · labor + downtime (NOK)",
        "cmd":   [sys.executable, str(BASE / "optimization/simulator.py")],
    },
    {
        "id":    "s9",
        "label": "9.  Build Norway Map",
        "desc":  "Folium interactive map · risk markers · worker routes",
        "cmd":   [sys.executable, str(BASE / "demo/map_builder.py")],
    },
    {
        "id":    "s10",
        "label": "10. Run Test Suite",
        "desc":  "46-check end-to-end pipeline validation · PASS / FAIL report",
        "cmd":   [sys.executable, str(REPO_ROOT / "test_full_pipeline.py")],
    },
]

# ── Colors ────────────────────────────────────────────────────────────────────
BG     = "#0d1117"
BG2    = "#161b22"
BORDER = "#30363d"
TEXT   = "#e6edf3"
GRAY   = "#484f58"
GREEN  = "#3fb950"
ORANGE = "#f0883e"
RED    = "#f85149"
BLUE   = "#388bfd"


class PipelineRunner(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BatterySwapAI 2026 — Pipeline Runner")
        self.configure(bg=BG)
        self.geometry("860x740")
        self.minsize(700, 500)

        self.states      = {s["id"]: "idle" for s in STAGES}
        self.row_widgets = {}

        self._build_ui()
        self._refresh()

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        # Title
        hdr = tk.Frame(self, bg=BG, padx=20, pady=14)
        hdr.pack(fill="x")
        tk.Label(hdr, text="⚙  BatterySwapAI 2026 — Pipeline Runner",
                 bg=BG, fg=TEXT, font=("Segoe UI", 15, "bold")).pack(side="left")

        # Action buttons row
        btn_bar = tk.Frame(self, bg=BG, padx=20, pady=2)
        btn_bar.pack(fill="x")

        self.run_all_btn = tk.Button(
            btn_bar, text="▶▶  Run All Stages",
            bg=BLUE, fg="white", font=("Segoe UI", 10, "bold"),
            relief="flat", padx=14, pady=6, cursor="hand2",
            activebackground="#1f6feb", activeforeground="white",
            command=self._run_all,
        )
        self.run_all_btn.pack(side="left", padx=(0, 8))

        self.reset_btn = tk.Button(
            btn_bar, text="↺  Reset All",
            bg=BG2, fg=TEXT, font=("Segoe UI", 10),
            relief="flat", padx=14, pady=6, cursor="hand2",
            activebackground=BORDER, activeforeground=TEXT,
            command=self._reset_all,
        )
        self.reset_btn.pack(side="left")

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", pady=8)

        # Scrollable stage list
        outer = tk.Frame(self, bg=BG)
        outer.pack(fill="both", expand=True, padx=20)

        self._canvas = tk.Canvas(outer, bg=BG, highlightthickness=0)
        vsb = tk.Scrollbar(outer, orient="vertical", command=self._canvas.yview)
        self._inner = tk.Frame(self._canvas, bg=BG)

        self._inner.bind(
            "<Configure>",
            lambda _e: self._canvas.configure(
                scrollregion=self._canvas.bbox("all")
            ),
        )
        self._canvas.create_window((0, 0), window=self._inner, anchor="nw")
        self._canvas.configure(yscrollcommand=vsb.set)
        self._canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        self.bind_all(
            "<MouseWheel>",
            lambda e: self._canvas.yview_scroll(-1 * (e.delta // 120), "units"),
        )

        for i, stage in enumerate(STAGES):
            self._build_row(i, stage)

        # Output box
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", pady=(8, 0))

        out_hdr = tk.Frame(self, bg=BG, padx=20, pady=5)
        out_hdr.pack(fill="x")
        tk.Label(out_hdr, text="Output", bg=BG, fg=GRAY,
                 font=("Segoe UI", 9)).pack(side="left")
        tk.Button(
            out_hdr, text="Clear", bg=BG, fg=GRAY,
            font=("Segoe UI", 8), relief="flat", padx=6, cursor="hand2",
            command=self._clear_output,
        ).pack(side="right")

        self.copy_btn = tk.Button(
            out_hdr, text="Copy Output", bg=BG2, fg=TEXT,
            font=("Segoe UI", 8), relief="flat", padx=8, cursor="hand2",
            activebackground=BORDER, activeforeground=TEXT,
            command=self._copy_output,
        )
        self.copy_btn.pack(side="right", padx=(0, 6))

        self.out = scrolledtext.ScrolledText(
            self, height=9, bg=BG2, fg=TEXT,
            font=("Consolas", 9), relief="flat",
            insertbackground=TEXT, padx=10, pady=8,
            state="disabled",
        )
        self.out.pack(fill="x", padx=20, pady=(0, 16))

    def _build_row(self, i, stage):
        sid = stage["id"]

        row = tk.Frame(
            self._inner, bg=BG2, pady=10, padx=14,
            highlightbackground=BORDER, highlightthickness=1,
        )
        row.pack(fill="x", pady=3)

        dot_cv = tk.Canvas(row, width=18, height=18, bg=BG2, highlightthickness=0)
        dot_cv.pack(side="left", padx=(0, 12))
        dot = dot_cv.create_oval(2, 2, 16, 16, fill=GRAY, outline="")

        lf = tk.Frame(row, bg=BG2)
        lf.pack(side="left", fill="x", expand=True)
        tk.Label(lf, text=stage["label"], bg=BG2, fg=TEXT,
                 font=("Segoe UI", 11, "bold"), anchor="w").pack(fill="x")
        tk.Label(lf, text=stage["desc"], bg=BG2, fg=GRAY,
                 font=("Segoe UI", 9), anchor="w").pack(fill="x")

        st_lbl = tk.Label(row, text="Not run", bg=BG2, fg=GRAY,
                          font=("Segoe UI", 9), width=11, anchor="e")
        st_lbl.pack(side="right", padx=(6, 8))

        btn = tk.Button(
            row, text="▶ Run",
            bg=GRAY, fg="white", font=("Segoe UI", 9, "bold"),
            relief="flat", padx=12, pady=5, cursor="hand2",
            activeforeground="white",
            command=lambda s=stage: self._run_stage(s),
        )
        btn.pack(side="right", padx=(8, 0))

        self.row_widgets[sid] = {
            "row": row, "dot_cv": dot_cv, "dot": dot,
            "btn": btn, "st_lbl": st_lbl,
        }

    # ── State refresh ──────────────────────────────────────────────────────────

    def _refresh(self):
        any_running = any(v == "running" for v in self.states.values())
        all_done    = all(v == "done"    for v in self.states.values())

        self.run_all_btn.configure(
            state="disabled" if (any_running or all_done) else "normal",
            bg=GRAY if (any_running or all_done) else BLUE,
        )
        self.reset_btn.configure(state="disabled" if any_running else "normal")

        COLOR = {"idle": GRAY, "running": ORANGE, "done": GREEN, "error": RED}
        LABEL = {
            "idle": "Not run", "running": "Running…",
            "done": "Done ✓",  "error":   "Failed ✗",
        }

        for i, stage in enumerate(STAGES):
            sid      = stage["id"]
            status   = self.states[sid]
            w        = self.row_widgets[sid]
            unlocked = (i == 0) or (self.states[STAGES[i - 1]["id"]] == "done")
            c        = COLOR[status]

            w["dot_cv"].itemconfig(w["dot"], fill=c)
            w["row"].configure(highlightbackground=c)
            w["st_lbl"].configure(text=LABEL[status], fg=c)

            if status == "running":
                w["btn"].configure(text="⏳ Running…", state="disabled",
                                   bg=GRAY, activebackground=GRAY)
            elif not unlocked:
                w["btn"].configure(text="▶ Run", state="disabled",
                                   bg=GRAY, activebackground=GRAY)
            elif status == "done":
                w["btn"].configure(text="⟳ Re-run", state="normal",
                                   bg=BLUE, activebackground="#1f6feb")
            elif status == "error":
                w["btn"].configure(text="↺ Retry", state="normal",
                                   bg=RED, activebackground="#da3633")
            else:
                w["btn"].configure(text="▶ Run", state="normal",
                                   bg=BLUE, activebackground="#1f6feb")

    # ── Stage execution ────────────────────────────────────────────────────────

    def _run_stage(self, stage, auto_next=False):
        sid = stage["id"]
        self.states[sid] = "running"
        self._refresh()
        self._append(f"\n{'─'*60}\n▶  {stage['label']}\n{'─'*60}\n")

        def _worker():
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            result = subprocess.run(
                stage["cmd"],
                capture_output=True, text=True, encoding="utf-8",
                cwd=str(REPO_ROOT), env=env,
            )
            output = (result.stdout + result.stderr).strip()
            self.states[sid] = "done" if result.returncode == 0 else "error"
            self.after(0, lambda: self._stage_done(
                stage, result.returncode, output, auto_next
            ))

        threading.Thread(target=_worker, daemon=True).start()

    def _stage_done(self, stage, rc, output, auto_next):
        symbol = "✓" if rc == 0 else "✗"
        self._append(f"{output}\n\n[Exit {rc}] {symbol}\n")
        self._refresh()

        if auto_next and rc == 0:
            idx = next(i for i, s in enumerate(STAGES) if s["id"] == stage["id"])
            if idx + 1 < len(STAGES):
                self.after(150, lambda: self._run_stage(
                    STAGES[idx + 1], auto_next=True
                ))

    def _run_all(self):
        for stage in STAGES:
            if self.states[stage["id"]] != "done":
                self._run_stage(stage, auto_next=True)
                break

    def _reset_all(self):
        for s in STAGES:
            self.states[s["id"]] = "idle"
        self._clear_output()
        self._refresh()

    # ── Output helpers ─────────────────────────────────────────────────────────

    def _append(self, text):
        self.out.configure(state="normal")
        self.out.insert("end", text)
        self.out.see("end")
        self.out.configure(state="disabled")

    def _copy_output(self):
        text = self.out.get("1.0", "end").strip()
        if text:
            self.clipboard_clear()
            self.clipboard_append(text)
            self.copy_btn.configure(text="Copied ✓", bg=GREEN, fg="white")
            self.after(2000, lambda: self.copy_btn.configure(
                text="Copy Output", bg=BG2, fg=TEXT
            ))

    def _clear_output(self):
        self.out.configure(state="normal")
        self.out.delete("1.0", "end")
        self.out.configure(state="disabled")


if __name__ == "__main__":
    app = PipelineRunner()
    app.mainloop()
