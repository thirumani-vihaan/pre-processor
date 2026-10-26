import io
import json
from zipfile import ZipFile

import joblib
import pandas as pd
import pytest

from app import create_app
from lab.data import builtin_dataset


@pytest.fixture
def client(tmp_path):
    app = create_app(tmp_path)
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture
def saved(client):
    response = client.post("/api/run", json={"dataset": "iris", "config": {"models": ["nb"], "methods": ["iqr"]}})
    assert response.status_code == 200, response.get_json()
    return response.get_json()


def test_pages_and_catalog(client):
    for path in ("/", "/guide", "/api/catalog", "/api/health", "/api/runs", "/static/app.js", "/static/style.css"):
        assert client.get(path).status_code == 200


def test_sample_dataset_download(client):
    response = client.get("/api/sample.csv")
    frame = pd.read_csv(io.BytesIO(response.data))
    assert len(frame) == 732
    assert "class" in frame


def test_profile_endpoint(client):
    response = client.post("/api/profile", json={"dataset": "demo"})
    assert response.status_code == 200
    assert response.get_json()["profile"]["duplicates_removed"] == 12


def test_saved_run_and_all_exports(client, saved):
    run_id = saved["id"]
    assert client.get(f"/api/runs/{run_id}").get_json()["selected_id"] == saved["selected_id"]
    assert client.get(f"/report/{run_id}").status_code == 200
    response = client.get(f"/api/runs/{run_id}/download/bundle")
    with ZipFile(io.BytesIO(response.data)) as archive:
        assert set(archive.namelist()) == {
            "results.json", "comparison.csv", "predictions.csv", "training_audit.csv",
            "processed_training.csv", "model.joblib", "report.html",
        }
        persisted = json.loads(archive.read("results.json"))
        assert persisted["id"] == run_id
        model = joblib.load(io.BytesIO(archive.read("model.joblib")))
        prediction = model.predict(pd.DataFrame(saved["sample_input"]))
        assert len(prediction) == 1
    for kind in ("results", "comparison", "predictions", "audit", "processed", "model", "report"):
        assert client.get(f"/api/runs/{run_id}/download/{kind}").status_code == 200
    assert client.get("/api/runs").get_json()["runs"][0]["id"] == run_id


def test_prediction_roundtrip_and_schema_errors(client, saved):
    url = f"/api/runs/{saved['id']}/predict"
    response = client.post(url, json={"rows": saved["sample_input"]})
    assert response.status_code == 200
    row = response.get_json()["predictions"][0]
    assert row["class"] in saved["classes"]
    assert sum(row["probabilities"].values()) == pytest.approx(1)
    assert client.post(url, json={"rows": [{"unknown": 5}]}).status_code == 400
    assert client.post(url, json={"rows": []}).status_code == 400
    bad = dict(saved["sample_input"][0])
    bad[next(iter(bad))] = "not a number"
    assert client.post(url, json={"rows": [bad]}).status_code == 400


def test_csv_workflow(client):
    dataset = builtin_dataset("wine")
    frame = dataset.X.copy()
    frame["label"] = dataset.y
    upload = client.post("/api/upload", data={"file": (io.BytesIO(frame.to_csv(index=False).encode()), "wine.csv")})
    assert upload.status_code == 200
    response = client.post("/api/run", json={
        "dataset": "upload", "upload_id": upload.get_json()["upload_id"], "target": "label",
        "config": {"models": ["knn"], "methods": ["lof"]},
    })
    assert response.status_code == 200, response.get_json()
    assert response.get_json()["dataset"] == "Uploaded CSV"


@pytest.mark.parametrize("data", [
    {"dataset": "unknown"}, {"dataset": ["wine"]}, {"config": {"models": []}},
    {"dataset": "upload", "upload_id": "../../secret"}, {"config": None},
])
def test_invalid_requests_return_actionable_400(client, data):
    response = client.post("/api/run", json=data)
    assert response.status_code == 400
    assert response.get_json()["error"]


def test_non_json_input_rejected(client):
    assert client.post("/api/run", data="not json").status_code == 400


def test_local_host_and_origin_boundaries(client):
    assert client.get("/", headers={"Host": "remote.example"}).status_code == 403
    assert client.post("/api/run", json={}, headers={"Origin": "https://remote.example"}).status_code == 403
    assert client.get("/", headers={"Host": "127.0.0.1:5080"}).status_code == 200


def test_bad_upload_and_invalid_run(client):
    assert client.post("/api/upload", data={"file": (io.BytesIO(b"a"), "bad.txt")}).status_code == 400
    assert client.get("/api/runs/not-a-run").status_code == 400


def test_result_persists_after_application_restart(tmp_path):
    first = create_app(tmp_path).test_client()
    saved = first.post("/api/run", json={"dataset": "iris", "config": {"models": ["nb"], "methods": ["iqr"]}}).get_json()
    second = create_app(tmp_path).test_client()
    assert second.get(f"/api/runs/{saved['id']}").status_code == 200
    assert second.get("/api/runs").get_json()["runs"][0]["id"] == saved["id"]


@pytest.mark.parametrize("value", [[], {}, True, "Infinity", "-Infinity"])
def test_invalid_numeric_prediction_values_are_actionable(client, saved, value):
    row = dict(saved["sample_input"][0])
    row[next(iter(row))] = value
    response = client.post(f"/api/runs/{saved['id']}/predict", json={"rows": [row]})
    assert response.status_code == 400, response.get_json()
    assert response.get_json()["error"]


def test_history_never_exposes_pending_artifacts(tmp_path):
    pending = tmp_path / ".pending-in-progress"
    pending.mkdir()
    (pending / "results.json").write_text("incomplete JSON", encoding="utf-8")
    client = create_app(tmp_path).test_client()
    response = client.get("/api/runs")
    assert response.status_code == 200
    assert response.get_json()["runs"] == []


def test_failed_save_never_publishes_partial_run(tmp_path, monkeypatch):
    from lab import storage

    def fail_model_save(*args, **kwargs):
        raise OSError("Simulated disk write failure")

    monkeypatch.setattr(storage.joblib, "dump", fail_model_save)
    client = create_app(tmp_path).test_client()
    response = client.post("/api/run", json={"dataset": "iris", "config": {"models": ["nb"], "methods": ["iqr"]}})
    assert response.status_code == 500
    assert client.get("/api/runs").get_json()["runs"] == []
    assert not list(tmp_path.glob(".pending-*"))
    assert not list(tmp_path.glob("*/results.json"))


def test_simultaneous_experiments_are_rejected_explicitly(tmp_path, monkeypatch):
    import app as app_module
    from threading import Event, Thread

    entered, release = Event(), Event()

    def hold_experiment(*args, **kwargs):
        entered.set()
        assert release.wait(10), "Concurrent-request test timed out"
        raise app_module.ValidationError("Deliberately stopped test experiment.")

    monkeypatch.setattr(app_module, "experiment", hold_experiment)
    app = create_app(tmp_path)
    statuses = []

    def first_request():
        with app.test_client() as client:
            statuses.append(client.post("/api/run", json={"dataset": "iris"}).status_code)

    thread = Thread(target=first_request)
    thread.start()
    try:
        assert entered.wait(10)
        with app.test_client() as client:
            response = client.post("/api/run", json={"dataset": "iris"})
            assert response.status_code == 409
            assert "Another experiment" in response.get_json()["error"]
    finally:
        release.set()
        thread.join(timeout=10)
    assert statuses == [400]


def test_request_size_limit_is_actionable(client):
    response = client.post("/api/upload", data=b"x" * (9 * 1024 * 1024 + 1), content_type="application/octet-stream")
    assert response.status_code == 413
    assert response.get_json()["error"]
