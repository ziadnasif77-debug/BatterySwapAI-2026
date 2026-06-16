"""
verify_install.py
Run once after setup to confirm every required package imports correctly.
Usage: python verify_install.py
"""

import sys

REQUIRED = [
    ("numpy",       "numpy",        "2.0"),
    ("pandas",      "pandas",       "2.0"),
    ("scipy",       "scipy",        "1.10"),
    ("sklearn",     "scikit-learn", "1.3"),
    ("lightgbm",    "lightgbm",     "4.0"),
    ("ortools",     "ortools",      "9.0"),
    ("folium",      "folium",       "0.15"),
    ("plotly",      "plotly",       "5.0"),
    ("streamlit",   "streamlit",    "1.30"),
    ("matplotlib",  "matplotlib",   "3.7"),
]

STDLIB = ["datetime", "pathlib", "json", "math", "os", "pickle", "subprocess"]

ok = True
print()
print("╔════════════════════════════════════════════════╗")
print("║   BatterySwapAI 2026 — Install Verification   ║")
print("╚════════════════════════════════════════════════╝")
print()

# Python version
pymaj, pymin = sys.version_info.major, sys.version_info.minor
pyok = pymaj == 3 and pymin >= 10
sym = "✓" if pyok else "✗"
print(f"  {sym}  Python {sys.version.split()[0]}  (need 3.10+)")
if not pyok:
    ok = False

print()
print("  Third-party packages:")

for import_name, pip_name, min_ver in REQUIRED:
    try:
        mod = __import__(import_name)
        ver = getattr(mod, "__version__", "?")
        # crude version compare: split on '.' and compare first two parts
        try:
            actual = tuple(int(x) for x in ver.split(".")[:2])
            needed = tuple(int(x) for x in min_ver.split(".")[:2])
            ver_ok = actual >= needed
        except Exception:
            ver_ok = True
        sym = "✓" if ver_ok else "⚠"
        if not ver_ok:
            ok = False
        print(f"  {sym}  {pip_name:<18} {ver:<12} (min {min_ver})")
    except ImportError:
        print(f"  ✗  {pip_name:<18} NOT INSTALLED")
        ok = False

# OR-Tools sub-import (the one actually used)
try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2  # noqa: F401
    print("       └─ ortools.constraint_solver  ✓")
except ImportError as e:
    print(f"       └─ ortools.constraint_solver  ✗  {e}")
    ok = False

print()
print("  Standard library:")
for mod in STDLIB:
    try:
        __import__(mod)
        print(f"  ✓  {mod}")
    except ImportError:
        print(f"  ✗  {mod}  (unexpected — reinstall Python)")
        ok = False

print()
if ok:
    print("  ══════════════════════════════════════")
    print("  ✓  ALL CHECKS PASSED — ready to run!")
    print("  ══════════════════════════════════════")
    print()
    print("  Next steps:")
    print("    python test_full_pipeline.py          # full test suite")
    print("    streamlit run demo/dashboard.py       # launch dashboard")
    print("    open demo/battery_map.html            # interactive map")
else:
    print("  ══════════════════════════════════════")
    print("  ✗  SOME CHECKS FAILED")
    print("     Run: pip install -r requirements.txt")
    print("  ══════════════════════════════════════")
    sys.exit(1)

print()
