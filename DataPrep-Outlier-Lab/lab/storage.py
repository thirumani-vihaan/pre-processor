import json
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4
from datetime import datetime, timezone

import joblib
import pandas as pd


def save_run(root: Path, report, pipeline, predictions, audit, cleaned, render_report):
    root.mkdir(parents=True, exist_ok=True)
    run_id = uuid4().hex
    report["id"] = run_id
    report["created_at"] = datetime.now(timezone.utc).isoformat()
    with TemporaryDirectory(prefix=".pending-", dir=root) as temporary:
        folder = Path(temporary)
        (folder / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        predictions.to_csv(folder / "predictions.csv", index=False)
        audit.to_csv(folder / "training_audit.csv", index=False)
        cleaned.to_csv(folder / "processed_training.csv", index=False)
        rows = [{k: v for k, v in row.items() if k not in ("confusion", "fold_scores")} for row in report["results"]]
        pd.DataFrame(rows).to_csv(folder / "comparison.csv", index=False)
        joblib.dump(pipeline, folder / "model.joblib")
        (folder / "report.html").write_text(render_report(report), encoding="utf-8")
        folder.rename(root / run_id)
    return run_id
