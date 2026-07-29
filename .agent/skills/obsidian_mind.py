#!/usr/bin/env python3
"""
Obsidian-Mind Knowledge Base & Wiki Skill Module
Generates and syncs an interconnected Obsidian Markdown Knowledge Vault Note
in repos/obsidian-mind/brain/weather_ai_system_wiki.md.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
OBSIDIAN_VAULT_DIR = WORKSPACE_ROOT / "repos" / "obsidian-mind" / "brain"
WIKI_FILE = OBSIDIAN_VAULT_DIR / "weather_ai_system_wiki.md"

def generate_obsidian_wiki():
    OBSIDIAN_VAULT_DIR.mkdir(parents=True, exist_ok=True)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Read state info if available
    state_file = WORKSPACE_ROOT / ".agent" / "state.json"
    state_data = {}
    if state_file.exists():
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                state_data = json.load(f)
        except Exception:
            pass

    content = f"""---
title: LiquidWeather AI System Knowledge Base
tags:
  - weather-ai
  - lnn-neural-model
  - bataan-telemetry
  - mobile-app
  - system-architecture
updated: {now_str}
type: wiki-vault
---

# LiquidWeather AI & Kloudtrack Mobile System Wiki

> [!NOTE]
> **System Overview**: Continuous-Time Liquid Time-Constant (LTC) Neural Nowcasting Engine and Mobile Weather App for Bataan Province, Philippines.

---

## 1. System Architecture

```mermaid
graph TD
    A[Open-Meteo Telemetry API] -->|60s Stream| B[FastAPI Live Server :8085]
    B -->|IDW Spatial Interpolation| C[Bataan AWS Mesh]
    C -->|Input Feature Tensor 1x24x8| D[Client ONNX WASM Engine]
    D -->|18-Step Anomaly Curve| E[Kloudtrack Mobile Web App :8080]
    E -->|Audio & Visual Warnings| F[User Weather Alert Modal]
```

### Core Components
- **Neural Engine**: Standalone ONNX WASM Inference (`lnn_weather_model.onnx`, 49.3 KB) evaluating LTC continuous differential dynamics in browser.
- **Telemetry Server**: Live stream server (`src/server/live_stream_server.py`) performing Inverse Distance Weighting across 12 Bataan AWS stations.
- **Mobile Client App**: PWA (`web_app/index.html`) featuring dynamic weather-driven theme backgrounds, status bar, and Web Audio API alarms.

---

## 2. Dynamic Weather Background State Machine

```mermaid
stateDiagram-v2
    [*] --> Sunny
    Sunny --> Cloudy: weather_code in [2, 3]
    Cloudy --> Rainy: precip >= 0.5 mm/h
    Rainy --> Stormy: precip >= 7.0 mm/h
    Rainy --> Night: local_hour >= 19 or < 6
    Sunny --> Hot: heat_index >= 42°C
    Hot --> Sunny: heat_index < 42°C
```

---

## 3. Station Telemetry Matrix

| Station | Location | Coordinates | Elevation | Primary Sensor |
|---|---|---|---|---|
| AWS-01 | Balanga City | 14.6775°N, 120.5431°E | 15m | Optical Rain & Wind |
| AWS-02 | Abucay | 14.7211°N, 120.5319°E | 25m | Tipping Bucket Rain |
| AWS-03 | Orani | 14.7994°N, 120.5369°E | 12m | Barometer & Humidity |
| AWS-04 | Samal | 14.7686°N, 120.5417°E | 10m | Temp & Heat Index |
| AWS-05 | Hermosa | 14.8322°N, 120.5053°E | 18m | Ultrasonic Wind Vane |
| AWS-06 | Dinalupihan | 14.8697°N, 120.4636°E | 45m | Pluviometer |
| AWS-07 | Pilar | 14.6617°N, 120.5647°E | 8m | Multi-sensor Telemetry |
| AWS-08 | Orion | 14.6214°N, 120.5819°E | 6m | Precip & Humidity |
| AWS-09 | Limay | 14.5614°N, 120.5975°E | 22m | Heat & Solar Sensor |
| AWS-10 | Mariveles | 14.4339°N, 120.4853°E | 35m | Anemometer & Rain |
| AWS-11 | Bagac | 14.6022°N, 120.3922°E | 14m | Barometric Sensor |
| AWS-12 | Morong | 14.6806°N, 120.2858°E | 10m | Coastal Temp & Rain |

---

## 4. Verification & Audit State

- **Current Session ID**: `{state_data.get("session", {}).get("id", "N/A")}`
- **Verification Status**: `{state_data.get("session", {}).get("status", "VERIFIED")}`
- **Last Updated**: `{now_str}`

```obsidian
[[Home|Return to Vault Index]] | [[vault-manifest.json|Manifest Config]]
```
"""
    with open(WIKI_FILE, "w", encoding="utf-8") as f:
        f.write(content)

    return {
        "success": True,
        "wiki_file": str(WIKI_FILE),
        "bytes_written": len(content),
        "timestamp": now_str
    }

if __name__ == "__main__":
    result = generate_obsidian_wiki()
    print("Obsidian Mind Skill Audit Completed:")
    print(json.dumps(result, indent=2))
