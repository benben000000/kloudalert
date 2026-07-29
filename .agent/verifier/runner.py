#!/usr/bin/env python3
"""
Empirical Verifier Engine for Agentic Loop
Enforces Zero False Positives by executing tasks in isolated subprocesses,
capturing stdout/stderr/exit codes, logging results, and validating output evidence.
"""

import sys
import os
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = WORKSPACE_ROOT / ".agent" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

class EmpiricalVerifier:
    def __init__(self, log_dir=LOGS_DIR):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def run_command(self, command, task_name="cmd_execution", cwd=WORKSPACE_ROOT):
        """
        Executes a shell command, logs all output, and verifies execution empirically.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{timestamp}_{task_name}.log"
        log_path = self.log_dir / log_filename

        start_time = time.time()
        
        try:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(cwd)
            )
            stdout, stderr = process.communicate()
            exit_code = process.returncode
        except Exception as e:
            stdout = ""
            stderr = str(e)
            exit_code = -1

        duration = round(time.time() - start_time, 3)
        success = (exit_code == 0)

        # Log structure
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "task_name": task_name,
            "command": command,
            "cwd": str(cwd),
            "exit_code": exit_code,
            "duration_seconds": duration,
            "success": success,
            "stdout": stdout,
            "stderr": stderr
        }

        # Write to log file
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"=== EMPIRICAL VERIFICATION LOG ===\n")
            f.write(f"Timestamp: {log_entry['timestamp']}\n")
            f.write(f"Task: {task_name}\n")
            f.write(f"Command: {command}\n")
            f.write(f"CWD: {cwd}\n")
            f.write(f"Exit Code: {exit_code}\n")
            f.write(f"Duration: {duration}s\n")
            f.write(f"Success Status: {'PASSED' if success else 'FAILED'}\n")
            f.write("=" * 35 + " STDOUT " + "=" * 35 + "\n")
            f.write(stdout if stdout else "(No stdout)\n")
            f.write("=" * 35 + " STDERR " + "=" * 35 + "\n")
            f.write(stderr if stderr else "(No stderr)\n")

        # Save summary meta JSON alongside
        meta_filename = f"{timestamp}_{task_name}_meta.json"
        meta_path = self.log_dir / meta_filename
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(log_entry, f, indent=2)

        return {
            "success": success,
            "exit_code": exit_code,
            "duration": duration,
            "log_file": str(log_path),
            "meta_file": str(meta_path),
            "stdout_preview": stdout[:500] if stdout else "",
            "stderr_preview": stderr[:500] if stderr else ""
        }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python runner.py <command> [task_name]")
        sys.exit(1)
    
    cmd = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else "manual_verify"
    
    verifier = EmpiricalVerifier()
    result = verifier.run_command(cmd, name)
    print(json.dumps(result, indent=2))
