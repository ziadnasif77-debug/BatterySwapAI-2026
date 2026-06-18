"""
runner.py — BatterySwapAI 2026 Pipeline Runner

Interactive Streamlit dashboard to run each pipeline stage in order.

Each stage has:
  ● Gray  → not started yet
  ● Red   → currently running
  ● Green → finished successfully
  ● Red ✗ → failed (click to re-run)

Usage (from repo root):
    streamlit run runner.py
"""

import subprocess
import sys
from pathlib import Path
import streamlit as st

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent
BASE      = REPO_ROOT / "battery_swap_ai_2026"

# ── Pipeline stage definitions ─────────────────────────────────────────────────
STAGES = [
    {
        "id":    "s1_generate",
        "label": "Generate Data",
        "desc":  "500 sensors · 50 buildings · 6 Norwegian cities",
        "icon":  "📂",
        "cmd":   [sys.executable, str(BASE / "data/raw/generate_dummy_data.py")],
    },
    {
        "id":    "s2_features",
        "label": "Feature Engineering",
        "desc":  "30+ rolling voltage / temperature features",
        "icon":  "⚙️",
        "cmd":   [sys.executable, str(BASE / "model/feature_pipeline.py")],
    },
    {
        "id":    "s3_baseline",
        "label": "Baseline Model",
        "desc":  "Linear RUL extrapolation · reference MAE",
        "icon":  "📏",
        "cmd":   [sys.executable, str(BASE / "model/baseline.py")],
    },
    {
        "id":    "s4_train",
        "label": "Train LightGBM",
        "desc":  "RUL regression + isotonic calibration by building type",
        "icon":  "🤖",
        "cmd":   [sys.executable, str(BASE / "model/train.py")],
    },
    {
        "id":    "s5_uncertainty",
        "label": "Uncertainty Quantification",
        "desc":  "Quantile intervals · p_fail_3d / 7d / 14d  ∈ [0, 1]",
        "icon":  "📊",
        "cmd":   [sys.executable, str(BASE / "model/uncertainty.py")],
    },
    {
        "id":    "s6_priority",
        "label": "Risk & Priority Scoring",
        "desc":  "DEAD / CRITICAL / WARNING / SAFE · building-type weights",
        "icon":  "🎯",
        "cmd":   [sys.executable, str(BASE / "optimization/priority.py")],
    },
    {
        "id":    "s7_scheduler",
        "label": "VRP Scheduler",
        "desc":  "OR-Tools · 3 workers · 480-min shift · unreachable detection",
        "icon":  "🚗",
        "cmd":   [sys.executable, str(BASE / "optimization/scheduler.py")],
    },
    {
        "id":    "s8_simulator",
        "label": "Cost Simulation",
        "desc":  "AGGRESSIVE / NORMAL / CONSERVATIVE · labor + downtime (NOK)",
        "icon":  "💡",
        "cmd":   [sys.executable, str(BASE / "optimization/simulator.py")],
    },
    {
        "id":    "s9_map",
        "label": "Build Norway Map",
        "desc":  "Folium interactive map · risk markers · worker routes",
        "icon":  "🗺️",
        "cmd":   [sys.executable, str(BASE / "demo/map_builder.py")],
    },
    {
        "id":    "s10_test",
        "label": "Run Test Suite",
        "desc":  "46-check end-to-end pipeline validation · PASS / FAIL report",
        "icon":  "✅",
        "cmd":   [sys.executable, str(REPO_ROOT / "test_full_pipeline.py")],
    },
]

# ── Session state bootstrap ────────────────────────────────────────────────────
for s in STAGES:
    key = s["id"]
    if key not in st.session_state:
        st.session_state[key] = {
            "status":     "idle",   # idle | running | done | error
            "output":     "",
            "returncode": None,
        }
if "run_all" not in st.session_state:
    st.session_state["run_all"] = False


# ── Execute any stage that is queued as "running" ──────────────────────────────
# (Two-phase: button sets status → rerun renders red → this block executes)
_ran_this_cycle = False
for _s in STAGES:
    _sid = _s["id"]
    _st  = st.session_state[_sid]
    if _st["status"] == "running" and _st["output"] == "":
        with st.spinner(f"Running  {_s['icon']}  {_s['label']} …"):
            result = subprocess.run(
                _s["cmd"],
                capture_output=True,
                text=True,
                cwd=str(REPO_ROOT),
            )
        _st["output"]     = (result.stdout + result.stderr).strip()
        _st["returncode"] = result.returncode
        _st["status"]     = "done" if result.returncode == 0 else "error"
        _ran_this_cycle   = True
        # If "Run All" is active and stage succeeded, queue the next one
        if st.session_state["run_all"] and result.returncode == 0:
            _idx = next((i for i, x in enumerate(STAGES) if x["id"] == _sid), -1)
            if _idx + 1 < len(STAGES):
                _next = STAGES[_idx + 1]["id"]
                st.session_state[_next]["status"] = "running"
                st.session_state[_next]["output"] = ""
            else:
                st.session_state["run_all"] = False
        elif st.session_state["run_all"] and result.returncode != 0:
            st.session_state["run_all"] = False   # abort on failure
        st.rerun()
        break


# ── Page config (must come before any other st.* call on a fresh render) ───────
st.set_page_config(
    page_title="BatterySwapAI — Pipeline Runner",
    page_icon="⚙️",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stage-row {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 14px 18px;
    border-radius: 10px;
    margin-bottom: 10px;
    background: #161b22;
    border: 1px solid #30363d;
  }
  .stage-row.done   { border-left: 4px solid #3fb950; }
  .stage-row.error  { border-left: 4px solid #f85149; }
  .stage-row.running{ border-left: 4px solid #f0883e; }
  .stage-row.idle   { border-left: 4px solid #484f58; }

  .dot {
    width: 18px; height: 18px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .dot-idle    { background: #484f58; }
  .dot-running { background: #f0883e; box-shadow: 0 0 8px #f0883e88; }
  .dot-done    { background: #3fb950; box-shadow: 0 0 6px #3fb95066; }
  .dot-error   { background: #f85149; box-shadow: 0 0 8px #f8514988; }

  .stage-label { font-weight: 700; font-size: 15px; color: #e6edf3; }
  .stage-desc  { font-size: 12px; color: #8b949e; margin-top: 2px; }
  .stage-num   { font-size: 12px; color: #484f58; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("## ⚙️ BatterySwapAI 2026 — Pipeline Runner")
st.caption("Run each stage in order · click a button to execute · outputs expand below each stage")

# ── Progress bar ───────────────────────────────────────────────────────────────
n_done  = sum(1 for s in STAGES if st.session_state[s["id"]]["status"] == "done")
n_error = sum(1 for s in STAGES if st.session_state[s["id"]]["status"] == "error")
n_total = len(STAGES)

prog_col1, prog_col2, prog_col3, prog_col4 = st.columns([4, 1, 1, 1])
with prog_col1:
    st.progress(n_done / n_total, text=f"{n_done} / {n_total} stages complete")
with prog_col2:
    st.metric("Done",   n_done,  delta=None)
with prog_col3:
    st.metric("Failed", n_error, delta=None)
with prog_col4:
    remaining = n_total - n_done - n_error
    st.metric("Pending", remaining, delta=None)

st.divider()

# ── Top action buttons ─────────────────────────────────────────────────────────
act1, act2, act3 = st.columns([2, 2, 6])

any_running = any(
    st.session_state[s["id"]]["status"] == "running" for s in STAGES
)

with act1:
    run_all_disabled = any_running or st.session_state["run_all"] or n_done == n_total
    if st.button("▶▶  Run All Stages", disabled=run_all_disabled,
                 use_container_width=True, type="primary"):
        # Find first non-done stage and queue it
        for s in STAGES:
            if st.session_state[s["id"]]["status"] != "done":
                st.session_state[s["id"]]["status"] = "running"
                st.session_state[s["id"]]["output"] = ""
                break
        st.session_state["run_all"] = True
        st.rerun()

with act2:
    if st.button("🔄  Reset All", disabled=any_running,
                 use_container_width=True):
        for s in STAGES:
            st.session_state[s["id"]] = {
                "status": "idle", "output": "", "returncode": None
            }
        st.session_state["run_all"] = False
        st.rerun()

st.divider()


# ── Stage rows ─────────────────────────────────────────────────────────────────
for i, stage in enumerate(STAGES):
    sid    = stage["id"]
    sstate = st.session_state[sid]
    status = sstate["status"]

    # Unlock rule: stage 1 always unlocked; rest need previous = done
    unlocked = (i == 0) or (st.session_state[STAGES[i - 1]["id"]]["status"] == "done")

    # Dot CSS class
    dot_cls = f"dot-{status}"

    # Row CSS class
    row_cls = status

    # Status label text
    status_labels = {
        "idle":    "Not run",
        "running": "Running…",
        "done":    "Done ✓",
        "error":   "Failed ✗",
    }

    # ── Render: status dot + labels on left, button on right ──────────────────
    left_col, btn_col = st.columns([5, 1])

    with left_col:
        st.markdown(f"""
<div class="stage-row {row_cls}">
  <div class="dot {dot_cls}"></div>
  <div>
    <div class="stage-num">Stage {i + 1} of {n_total}
      &nbsp;·&nbsp; <span style="color:{'#3fb950' if status=='done' else '#f85149' if status=='error' else '#f0883e' if status=='running' else '#484f58'}">{status_labels[status]}</span>
    </div>
    <div class="stage-label">{stage['icon']} &nbsp;{stage['label']}</div>
    <div class="stage-desc">{stage['desc']}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    with btn_col:
        # Button label
        if status == "idle":
            btn_txt = "▶ Run"
        elif status == "running":
            btn_txt = "⏳ Running"
        elif status == "done":
            btn_txt = "⟳ Re-run"
        else:
            btn_txt = "↺ Retry"

        btn_disabled = not unlocked or status == "running"
        btn_type     = "primary" if status in ("idle",) and unlocked else "secondary"

        if st.button(btn_txt, key=f"btn_{sid}",
                     disabled=btn_disabled,
                     use_container_width=True,
                     type=btn_type):
            st.session_state[sid]["status"] = "running"
            st.session_state[sid]["output"] = ""
            st.session_state["run_all"]     = False
            st.rerun()

    # ── Output expander ────────────────────────────────────────────────────────
    if sstate["output"]:
        with st.expander(
            f"{'❌ Error output' if status == 'error' else '📄 Output'} — {stage['label']}",
            expanded=(status == "error"),
        ):
            rc = sstate["returncode"]
            if rc is not None:
                rc_color = "#3fb950" if rc == 0 else "#f85149"
                st.markdown(
                    f'<small style="color:{rc_color}">Exit code: {rc}</small>',
                    unsafe_allow_html=True,
                )
            st.code(sstate["output"], language="bash")
