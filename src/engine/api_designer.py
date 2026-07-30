#!/usr/bin/env python3
"""
Header-Aware Rate-Limit Resilient API Designer Module
(`src/engine/api_designer.py`)
Official Base URL: https://api.kloudtechsea.com/api/v1
Header: x-kloudtrack-key

RATE LIMIT RESILIENCE PROTOCOL:
1. Inspects `RateLimit-Remaining` and `RateLimit-Reset` HTTP headers on every response.
2. Dynamically throttles request rate if remaining quota < 15.
3. Automatically sleeps and backs off on HTTP 429 status without failing.
4. Throws explicit APIExecutionError if retries are exhausted (ZERO FAKE/MOCK DATA).
"""

import os
import sys
import json
import time
import subprocess
import urllib.parse
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = WORKSPACE_ROOT / "data" / "kloudtech_config.json"

class APIExecutionError(Exception):
    """Raised when a live API call fails after exhausting retries."""
    pass

class APIDesigner:
    BASE_URL = "https://api.kloudtechsea.com/api/v1"
    DEFAULT_API_KEY = "kloud_live_d2c3dece36db0668228537f7846be15a3b0e9303aeeb704d"

    def __init__(self, config_path=CONFIG_PATH):
        self.config_path = Path(config_path)
        self.api_key = self._load_api_key()
        self.base_delay = 1.0  # Safe default delay between calls

    def _load_api_key(self):
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    key = cfg.get("kloudtech_api_key")
                    if key and "YOUR_KLOUDTECH" not in key:
                        return key
            except Exception:
                pass
        return self.DEFAULT_API_KEY

    def design_and_execute_request(self, endpoint_path: str, method: str = "GET", params: dict = None, body_data: dict = None, max_retries: int = 5):
        """
        Executes a rate-limit resilient REST call using Windows cURL Schannel TLS.
        Parses response headers and handles HTTP 429 rate limit backoff automatically.
        """
        query_str = ""
        if params:
            query_str = "?" + urllib.parse.urlencode(params)

        full_url = f"{self.BASE_URL}{endpoint_path}{query_str}"
        print(f"\n[RATE-LIMIT RESILIENT DISPATCH] {method} {full_url}")

        cmd = [
            "curl.exe",
            "-i",  # Include HTTP headers in output
            "-s",
            "-X", method,
            "-H", f"x-kloudtrack-key: {self.api_key}",
            "-H", "Accept: application/json",
            "-H", "User-Agent: KloudAlert-RateLimitOptimizer/1.0",
            full_url
        ]

        for attempt in range(1, max_retries + 1):
            time.sleep(self.base_delay)
            t0 = time.time()
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True)
                lat_ms = round((time.time() - t0) * 1000, 1)
                
                output = res.stdout
                parts = output.split("\r\n\r\n", 1) if "\r\n\r\n" in output else output.split("\n\n", 1)
                header_text = parts[0]
                body_text = parts[1] if len(parts) > 1 else ""

                # Parse HTTP Status Code
                status_code = 200
                first_line = header_text.splitlines()[0] if header_text else ""
                if "HTTP/" in first_line:
                    try:
                        status_code = int(first_line.split()[1])
                    except Exception:
                        pass

                # Parse RateLimit Headers
                remaining = None
                reset_sec = 15
                for line in header_text.splitlines():
                    lower_line = line.lower()
                    if "ratelimit-remaining:" in lower_line:
                        try: remaining = int(line.split(":")[1].strip())
                        except Exception: pass
                    elif "ratelimit-reset:" in lower_line:
                        try: reset_sec = int(line.split(":")[1].strip())
                        except Exception: pass

                # Adaptive Throttling if quota is running low
                if remaining is not None and remaining < 15:
                    adaptive_delay = max(1.5, round(reset_sec / max(1, remaining), 2))
                    self.base_delay = adaptive_delay
                    print(f"   [ADAPTIVE THROTTLE] Quota Low ({remaining} left). Dynamic delay set to {self.base_delay}s")

                # Handle 429 Too Many Requests
                if status_code == 429:
                    wait_sec = max(reset_sec + 2, 2 ** attempt * 3)
                    print(f"   ⚠️ [HTTP 429 RATE LIMIT REACHED] Sleeping {wait_sec}s for quota reset (Attempt {attempt}/{max_retries})...")
                    time.sleep(wait_sec)
                    continue

                if status_code in (200, 201):
                    data = json.loads(body_text)
                    print(f"   [OK] [HTTP {status_code} SUCCESS] Latency: {lat_ms}ms | Quota Remaining: {remaining}")
                    return {
                        "success": True,
                        "status": status_code,
                        "latency_ms": lat_ms,
                        "url": full_url,
                        "data": data
                    }

                raise APIExecutionError(f"HTTP Status {status_code} returned from {full_url}")

            except subprocess.TimeoutExpired:
                print(f"   ⚠️ Request Timeout (Attempt {attempt}/{max_retries})")
            except Exception as e:
                if attempt == max_retries:
                    raise APIExecutionError(f"API Execution Error ({e}) for {full_url}")

        raise APIExecutionError(f"API Execution Failed after {max_retries} attempts due to Rate Limits.")

    def verify_live_dashboard_connection(self):
        """Dispatches live dashboard verification probe."""
        return self.design_and_execute_request(
            endpoint_path="/telemetry/dashboard",
            method="GET"
        )

if __name__ == "__main__":
    try:
        designer = APIDesigner()
        res = designer.verify_live_dashboard_connection()
        print(json.dumps({k: v for k, v in res.items() if k != "data"}, indent=2))
    except APIExecutionError as e:
        print(f"\n❌ [STRICT AUDIT ERROR] {e}")
        sys.exit(1)
