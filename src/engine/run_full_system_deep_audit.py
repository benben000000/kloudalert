#!/usr/bin/env python3
"""
Master System Deep Audit Engine (`src/engine/run_full_system_deep_audit.py`)
Executes and aggregates audit results across all 9 agentic skill modules:
1. fullstack.py (FullStackSkill)
2. security.py (SecuritySkill)
3. accessibility.py (AccessibilitySkill)
4. obsidian_mind.py (generate_obsidian_wiki)
5. ponytail.py (run_ponytail_audit)
6. improve.py (run_improve_audit)
7. openscience.py (run_openscience_audit)
8. lfm_foundation.py (run_lfm_foundation_audit)
9. emulator.py (run_emulator_skill_audit)
"""

import sys
import json
import time
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
SKILLS_DIR = WORKSPACE_ROOT / ".agent" / "skills"
sys.path.append(str(SKILLS_DIR))

from fullstack import FullStackSkill
from security import SecuritySkill
from accessibility import AccessibilitySkill
import obsidian_mind
import ponytail
import improve
import openscience
import lfm_foundation
import emulator

def run_deep_audit():
    start_time = time.time()
    results = {}

    audits = [
        ("fullstack", lambda: FullStackSkill().audit_all()),
        ("security", lambda: SecuritySkill().scan_all()),
        ("accessibility", lambda: AccessibilitySkill().audit_all()),
        ("obsidian_mind", lambda: obsidian_mind.generate_obsidian_wiki()),
        ("ponytail", lambda: ponytail.run_ponytail_audit()),
        ("improve", lambda: improve.run_improve_audit()),
        ("openscience", lambda: openscience.run_openscience_audit()),
        ("lfm_foundation", lambda: lfm_foundation.run_lfm_foundation_audit()),
        ("emulator", lambda: emulator.run_emulator_skill_audit())
    ]

    for name, audit_func in audits:
        try:
            t0 = time.time()
            res = audit_func()
            duration = round(time.time() - t0, 4)
            results[name] = {
                "skill": name,
                "status": "PASS",
                "audit_duration_sec": duration,
                "details": res
            }
        except Exception as e:
            results[name] = {
                "skill": name,
                "status": "ERROR",
                "error": str(e)
            }

    duration = round(time.time() - start_time, 4)

    passed_count = sum(1 for r in results.values() if r.get("status") == "PASS")
    total_count = len(results)

    summary = {
        "timestamp": time.time(),
        "total_skills_audited": total_count,
        "passed_skills": passed_count,
        "failed_skills": total_count - passed_count,
        "total_audit_duration_sec": duration,
        "system_status": "ALL_SYSTEMS_OPERATIONAL" if passed_count == total_count else "PARTIAL_PASS",
        "skills_detail": results
    }

    return summary

if __name__ == "__main__":
    print("[SYSTEM DEEP AUDIT] Executing comprehensive 9-skill audit pass...")
    audit_summary = run_deep_audit()
    print(json.dumps(audit_summary, indent=2))
