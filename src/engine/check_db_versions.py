import sqlite3
from pathlib import Path

db_path = Path("data/db/kloudalert_central.db")
if db_path.exists():
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT version_tag, avg_loss, verified_sample_count, deployed_at FROM model_versions ORDER BY version_id DESC LIMIT 5")
    rows = c.fetchall()
    print("=================================================================")
    print("SQLITE CENTRAL DATABASE MODEL VERSION REGISTRY")
    print("=================================================================")
    for r in rows:
        loss_val = f"{r[1]:.5f}" if r[1] is not None else "N/A"
        samples = f"{r[2]:,}" if r[2] is not None else "N/A"
        print(f"  • Version Tag: {r[0]}")
        print(f"    - Avg Loss:     {loss_val}")
        print(f"    - Sample Count: {samples}")
        print(f"    - Timestamp:    {r[3]}")
        print("  ---------------------------------------------------------------")
    conn.close()
