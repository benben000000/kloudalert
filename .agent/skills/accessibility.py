#!/usr/bin/env python3
"""
Accessibility & Quality Skill Module
Enforces HTML5 semantics, WCAG 2.1 accessibility standards,
and DOM quality rules across frontend repositories.
"""

import os
import re
import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from verifier.runner import EmpiricalVerifier

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
REPOS_DIR = WORKSPACE_ROOT / "repos"

# Accessibility rules
A11Y_RULES = [
    (re.compile(r'<img(?![^>]*\balt=)[^>]*>'), "WCAG 1.1.1: Image missing alt attribute"),
    (re.compile(r'<button(?![^>]*\baria-label=)[^>]*>\s*</button>'), "WCAG 4.1.2: Empty button without aria-label"),
    (re.compile(r'<a(?![^>]*\bhref=)[^>]*>'), "WCAG 2.4.4: Anchor tag missing href attribute"),
    (re.compile(r'<div[^>]*onclick=(?![^>]*\brole=)[^>]*>'), "WCAG 2.1.1: Interactive div missing role attribute"),
    (re.compile(r'<input(?![^>]*\b(aria-label|id|aria-labelledby)=)[^>]*>'), "WCAG 1.3.1: Form input missing accessible label/id binding")
]

class AccessibilitySkill:
    def __init__(self):
        self.verifier = EmpiricalVerifier()

    def audit_repo(self, repo_name):
        repo_path = REPOS_DIR / repo_name
        if not repo_path.exists():
            return {"success": False, "error": f"Repo {repo_name} not found"}

        violations = []
        scanned_files = 0

        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in ('.git', 'node_modules', '__pycache__', 'dist', 'build')]
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix in ('.html', '.jsx', '.tsx', '.vue', '.svelte', '.php'):
                    scanned_files += 1
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()

                        for pattern, rule_desc in A11Y_RULES:
                            matches = pattern.findall(content)
                            if matches:
                                relative_path = file_path.relative_to(repo_path)
                                violations.append({
                                    "severity": "WARNING",
                                    "rule": rule_desc,
                                    "file": str(relative_path),
                                    "count": len(matches)
                                })
                    except Exception as e:
                        pass

        return {
            "repo": repo_name,
            "scanned_files": scanned_files,
            "violations_count": len(violations),
            "violations": violations
        }

    def audit_all(self):
        reports = {}
        for repo_dir in REPOS_DIR.iterdir():
            if repo_dir.is_dir() and not repo_dir.name.startswith("."):
                reports[repo_dir.name] = self.audit_repo(repo_dir.name)
        return reports

if __name__ == "__main__":
    a11y = AccessibilitySkill()
    print("Running Accessibility & Quality Audit across all repos...")
    report = a11y.audit_all()
    print(json.dumps(report, indent=2))
