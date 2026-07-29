#!/usr/bin/env python3
"""
Full-Stack Development Skill Module
Provides automated linters, syntax checkers, build verification, and test execution
for frontend (Node/TypeScript/React) and backend (Node/Python) repositories.
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory for verifier import
sys.path.append(str(Path(__file__).resolve().parent.parent))
from verifier.runner import EmpiricalVerifier

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
REPOS_DIR = WORKSPACE_ROOT / "repos"

class FullStackSkill:
    def __init__(self):
        self.verifier = EmpiricalVerifier()

    def audit_repo(self, repo_name):
        repo_path = REPOS_DIR / repo_name
        if not repo_path.exists():
            return {"success": False, "error": f"Repository {repo_name} not found"}

        results = {
            "repo": repo_name,
            "frontend_checks": [],
            "backend_checks": []
        }

        # Node / JS / TS check
        package_json = repo_path / "package.json"
        if package_json.exists():
            try:
                with open(package_json, "r", encoding="utf-8") as f:
                    pkg_data = json.load(f)
                
                scripts = pkg_data.get("scripts", {})
                results["has_package_json"] = True
                results["available_scripts"] = list(scripts.keys())

                # Run lint if present
                if "lint" in scripts:
                    lint_res = self.verifier.run_command(
                        "npm run lint",
                        task_name=f"{repo_name}_fullstack_lint",
                        cwd=repo_path
                    )
                    results["frontend_checks"].append(lint_res)

                # Run build dry-run / check if script exists
                if "build" in scripts:
                    build_res = self.verifier.run_command(
                        "npm run build -- --no-emit" if "tsc" in scripts.get("build", "") else "npm run build",
                        task_name=f"{repo_name}_fullstack_build",
                        cwd=repo_path
                    )
                    results["frontend_checks"].append(build_res)
            except Exception as e:
                results["frontend_checks"].append({"error": str(e)})

        # Python check
        py_files = list(repo_path.glob("**/*.py"))
        if py_files:
            results["has_python"] = True
            py_check = self.verifier.run_command(
                f"{sys.executable} -m py_compile " + " ".join(f'"{p}"' for p in py_files[:10]),
                task_name=f"{repo_name}_python_compile_check",
                cwd=repo_path
            )
            results["backend_checks"].append(py_check)

        return results

    def audit_all(self):
        reports = {}
        for repo_dir in REPOS_DIR.iterdir():
            if repo_dir.is_dir() and not repo_dir.name.startswith("."):
                reports[repo_dir.name] = self.audit_repo(repo_dir.name)
        return reports

if __name__ == "__main__":
    fs = FullStackSkill()
    print("Running Full-Stack Skill Audit across all repos...")
    report = fs.audit_all()
    print(json.dumps(report, indent=2))
