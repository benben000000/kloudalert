#!/usr/bin/env python3
"""
Security & Privacy Skill Module
Executes static application security testing (SAST), secret scanning,
dependency vulnerability audits, and privacy checks across repositories.
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

# Sensitive secret patterns
SECRET_PATTERNS = [
    (re.compile(r'(?i)(api[_\-]?key|secret|token|password|auth_token)\s*=\s*[\'"][A-Za-z0-9_\-]{16,}[\'"]'), "Hardcoded API Key / Secret"),
    (re.compile(r'-----BEGIN (RSA|EC|OPENSSH|DSA)? PRIVATE KEY-----'), "Hardcoded Private Key"),
    (re.compile(r'(?i)ghp_[A-Za-z0-9_]{36}'), "GitHub Personal Access Token"),
    (re.compile(r'(?i)sk_live_[0-9a-zA-Z]{24}'), "Stripe Live Secret Key")
]

# SAST security patterns
SAST_PATTERNS = [
    (re.compile(r'\beval\s*\('), "Use of eval() function"),
    (re.compile(r'\bexec\s*\('), "Use of exec() function"),
    (re.compile(r'dangerouslySetInnerHTML'), "Dangerous React HTML Insertion"),
    (re.compile(r'SELECT\s+.*\s+FROM\s+.*\s+\+\s*'), "Potential Unescaped SQL Concatenation")
]

class SecuritySkill:
    def __init__(self):
        self.verifier = EmpiricalVerifier()

    def scan_repo(self, repo_name):
        repo_path = REPOS_DIR / repo_name
        if not repo_path.exists():
            return {"success": False, "error": f"Repo {repo_name} not found"}

        findings = []
        scanned_files = 0

        for root, dirs, files in os.walk(repo_path):
            # Skip git / node_modules
            dirs[:] = [d for d in dirs if d not in ('.git', 'node_modules', '__pycache__', 'dist', 'build')]
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix in ('.js', '.ts', '.jsx', '.tsx', '.py', '.json', '.md', '.env', '.yaml', '.yml'):
                    scanned_files += 1
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()

                        # Secret Scan
                        for pattern, rule_name in SECRET_PATTERNS:
                            matches = pattern.findall(content)
                            if matches:
                                relative_path = file_path.relative_to(repo_path)
                                findings.append({
                                    "severity": "HIGH",
                                    "category": "Secret Scan",
                                    "rule": rule_name,
                                    "file": str(relative_path),
                                    "count": len(matches)
                                })

                        # SAST Scan
                        for pattern, rule_name in SAST_PATTERNS:
                            matches = pattern.findall(content)
                            if matches:
                                relative_path = file_path.relative_to(repo_path)
                                findings.append({
                                    "severity": "MEDIUM",
                                    "category": "SAST",
                                    "rule": rule_name,
                                    "file": str(relative_path),
                                    "count": len(matches)
                                })

                    except Exception as e:
                        pass

        # Also run npm audit if package.json exists
        audit_res = None
        if (repo_path / "package.json").exists():
            audit_res = self.verifier.run_command(
                "npm audit --json",
                task_name=f"{repo_name}_npm_audit",
                cwd=repo_path
            )

        report = {
            "repo": repo_name,
            "scanned_files": scanned_files,
            "findings_count": len(findings),
            "findings": findings,
            "dependency_audit": audit_res
        }
        return report

    def scan_all(self):
        reports = {}
        for repo_dir in REPOS_DIR.iterdir():
            if repo_dir.is_dir() and not repo_dir.name.startswith("."):
                reports[repo_dir.name] = self.scan_repo(repo_dir.name)
        return reports

if __name__ == "__main__":
    sec = SecuritySkill()
    print("Running Security & SAST Audit across all repos...")
    report = sec.scan_all()
    print(json.dumps(report, indent=2))
