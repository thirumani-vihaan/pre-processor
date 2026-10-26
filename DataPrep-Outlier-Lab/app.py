import argparse
import io
import json
import logging
import re
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from lab.data import CATALOG, ValidationError, builtin_dataset, csv_dataset, prepare, read_csv
from lab.engine import Config, METHOD_NAMES, MODEL_NAMES, experiment
from lab.storage import save_run


ROOT = Path(__file__).resolve().parent


def create_app(artifact_root=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 9 * 1024 * 1024
    artifacts = Path(artifact_root) if artifact_root else ROOT / "artifacts"
    uploads = artifacts / "uploads"
    run_lock = Lock()

    def run_folder(run_id):
        if not re.fullmatch(r"[a-f0-9]{32}", run_id):
            raise ValidationError("Invalid experiment identifier.")
        folder = artifacts / run_id
        if not (folder / "results.json").is_file():
            raise ValidationError("Experiment not found. Run an experiment first.")
        return folder

    def load_report(run_id):
        return json.loads((run_folder(run_id) / "results.json").read_text(encoding="utf-8"))

    def payload():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise ValidationError("Send a JSON object.")
        return data

    def choose_dataset(data):
        source = data.get("dataset", "demo")
        seed = data.get("config", {}).get("seed", 42)
        if source == "upload":
            upload_id = data.get("upload_id", "")
            if not isinstance(upload_id, str) or not re.fullmatch(r"[a-f0-9]{32}", upload_id):
                raise ValidationError("Upload a CSV before selecting an uploaded dataset.")
            path = uploads / f"{upload_id}.csv"
            if not path.is_file():
                raise ValidationError("Uploaded CSV not found. Please upload it again.")
            exclude = data.get("exclude", [])
            if not isinstance(exclude, list) or any(not isinstance(v, str) for v in exclude):
                raise ValidationError("Excluded columns must be a list of names.")
            return csv_dataset(read_csv(path.read_bytes()), data.get("target"), exclude)
        stress = data.get("stress", False)
        if type(stress) is not bool:
            raise ValidationError("Stress-test setting must be true or false.")
        return builtin_dataset(source, seed=seed, missing=0.04 if stress else 0, anomalies=0.06 if stress else 0)

    @app.before_request
    def local_only():
        if request.host.split(":")[0] not in ("127.0.0.1", "localhost"):
            return jsonify(error="This app is available on localhost only."), 403
        origin = request.headers.get("Origin")
        if origin and urlparse(origin).netloc != request.host:
            return jsonify(error="Cross-origin requests are not allowed."), 403

    @app.after_request
    def response_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; form-action 'self'"
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(ValidationError)
    def validation_error(error):
        return jsonify(error=str(error)), 400

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(error):
        return jsonify(error="Upload is too large. Use a CSV smaller than 8 MB."), 413

    @app.errorhandler(Exception)
    def unexpected_error(error):
        if isinstance(error, HTTPException):
            return jsonify(error=error.description), error.code
        app.logger.exception("Experiment request failed")
        return jsonify(error="The operation failed. Details are in the terminal log; no successful result was saved."), 500

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/guide")
    def guide():
        return render_template("guide.html")

    @app.get("/api/catalog")
    def catalog():
        return jsonify(datasets=CATALOG, models=MODEL_NAMES, methods=METHOD_NAMES)

    @app.get("/api/health")
    def health():
        return jsonify(status="ok")

    @app.get("/api/sample.csv")
    def sample_csv():
        dataset = builtin_dataset("demo")
        frame = dataset.X.copy()
        frame["class"] = dataset.y
        stream = io.BytesIO(frame.to_csv(index=False).encode("utf-8"))
        return send_file(stream, mimetype="text/csv", as_attachment=True, download_name="messy_customer_demo.csv")

    @app.post("/api/upload")
    def upload():
        file = request.files.get("file")
        if not file or not file.filename or not file.filename.lower().endswith(".csv"):
            raise ValidationError("Select a .csv file.")
        content = file.read()
        frame = read_csv(content)
        upload_id = uuid4().hex
        uploads.mkdir(parents=True, exist_ok=True)
        (uploads / f"{upload_id}.csv").write_bytes(content)
        return jsonify(
            upload_id=upload_id, rows=len(frame), columns=[
                {"name": str(col), "unique": int(frame[col].nunique()), "missing": int(frame[col].isna().sum()),
                 "type": "numeric" if pd.api.types.is_numeric_dtype(frame[col]) else "categorical"}
                for col in frame.columns
            ],
        )

    @app.post("/api/profile")
    def profile():
        data = payload()
        Config.parse(data.get("config", {}))
        dataset, description = prepare(choose_dataset(data))
        preview = dataset.X.head(6).astype(object).where(dataset.X.head(6).notna(), None)
        return jsonify(name=dataset.name, notes=dataset.notes, profile=description, preview=preview.to_dict("records"))

    @app.post("/api/run")
    def run():
        data = payload()
        config = Config.parse(data.get("config", {}))
        dataset = choose_dataset(data)
        if not run_lock.acquire(blocking=False):
            return jsonify(error="Another experiment is running. Wait for it to finish."), 409
        try:
            outputs = experiment(dataset, config)
            save_run(artifacts, *outputs, render_report=lambda report: render_template("report.html", report=report))
        finally:
            run_lock.release()
        return jsonify(outputs[0])

    @app.get("/api/runs")
    def history():
        committed = [
            path for path in artifacts.glob("*/results.json")
            if re.fullmatch(r"[a-f0-9]{32}", path.parent.name)
        ]
        files = sorted(committed, key=lambda p: p.stat().st_mtime, reverse=True)[:20]
        runs = []
        for file in files:
            report = json.loads(file.read_text(encoding="utf-8"))
            selected = next(r for r in report["results"] if r["id"] == report["selected_id"])
            runs.append({"id": report["id"], "dataset": report["dataset"], "created_at": report["created_at"],
                         "model": selected["model"], "accuracy": selected["accuracy"]})
        return jsonify(runs=runs)

    @app.get("/api/runs/<run_id>")
    def get_run(run_id):
        return jsonify(load_report(run_id))

    @app.get("/report/<run_id>")
    def report_page(run_id):
        return render_template("report.html", report=load_report(run_id))

    @app.get("/api/runs/<run_id>/download/<kind>")
    def download(run_id, kind):
        folder = run_folder(run_id)
        files = {
            "results": "results.json", "comparison": "comparison.csv", "predictions": "predictions.csv",
            "audit": "training_audit.csv", "processed": "processed_training.csv", "model": "model.joblib",
            "report": "report.html",
        }
        if kind == "bundle":
            stream = io.BytesIO()
            with ZipFile(stream, "w", ZIP_DEFLATED) as archive:
                for filename in files.values():
                    path = folder / filename
                    if path.is_file():
                        archive.write(path, filename)
            stream.seek(0)
            return send_file(stream, mimetype="application/zip", as_attachment=True, download_name=f"dataprep-{run_id[:8]}.zip")
        if kind not in files:
            raise ValidationError("Unknown export type.")
        return send_file(folder / files[kind], as_attachment=True)

    @app.post("/api/runs/<run_id>/predict")
    def predict(run_id):
        report = load_report(run_id)
        data = payload()
        rows = data.get("rows")
        if not isinstance(rows, list) or not 1 <= len(rows) <= 500 or any(not isinstance(row, dict) for row in rows):
            raise ValidationError("Provide 1–500 records as a JSON list in 'rows'.")
        columns = report["profile"]["columns"]
        names = [col["name"] for col in columns]
        if any(set(row) != set(names) for row in rows):
            raise ValidationError("Every prediction record must contain exactly the original input feature names.")
        frame = pd.DataFrame(rows, columns=names)
        for col in columns:
            name = col["name"]
            if any(value is not None and not isinstance(value, (str, int, float, bool)) for value in frame[name]):
                raise ValidationError(f"Feature '{name}' must contain scalar values or null, not arrays or objects.")
            if col["type"] == "numeric":
                if frame[name].map(lambda value: isinstance(value, bool)).any():
                    raise ValidationError(f"Feature '{name}' must contain numbers or null, not booleans.")
                try:
                    frame[name] = pd.to_numeric(frame[name], errors="raise")
                except (TypeError, ValueError) as exc:
                    raise ValidationError(f"Feature '{name}' must contain numbers or null.") from exc
                if np.isinf(frame[name].to_numpy(dtype=float)).any():
                    raise ValidationError(f"Feature '{name}' must contain finite numbers or null.")
            else:
                frame[name] = frame[name].map(lambda v: str(v).strip() if v is not None else np.nan).replace("", np.nan)
        frame = frame.replace([np.inf, -np.inf], np.nan)
        # Only app-generated local models can be loaded; arbitrary model uploads are never accepted.
        pipeline = joblib.load(run_folder(run_id) / "model.joblib")
        labels = pipeline.predict(frame)
        probabilities = pipeline.predict_proba(frame)
        return jsonify(predictions=[
            {"class": str(label), "probabilities": dict(zip(report["classes"], map(float, probability)))}
            for label, probability in zip(labels, probabilities)
        ])

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DataPrep Lab — local course-based project")
    parser.add_argument("--port", type=int, default=5080)
    parser.add_argument("--demo", action="store_true", help="Save a reproducible demo experiment and exit.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    app = create_app()
    if args.demo:
        with app.test_client() as client:
            response = client.post("/api/run", json={"dataset": "demo"})
            if response.status_code != 200:
                raise SystemExit(response.get_json())
            result = response.get_json()
            print(f"Saved experiment {result['id']} in artifacts.")
    else:
        app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)
