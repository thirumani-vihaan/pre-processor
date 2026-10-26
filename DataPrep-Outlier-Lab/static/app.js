"use strict";
const $ = (id) => document.getElementById(id);
const colors = ["#527bea", "#269b91", "#8b7ac3", "#c79a53", "#b87999", "#68a4ba", "#929bb5"];
const esc = (value) => String(value).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (number) => `${(number * 100).toFixed(1)}%`;
const pp = (number) => `${number > 0 ? "+" : ""}${(number * 100).toFixed(2)} pp`;
let catalog = null;
let current = null;
let uploadId = null;
let busy = false;

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status}).`);
  return data;
}
function post(path, data) {
  return api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
}
function showError(error) {
  $("error").textContent = error.message || String(error);
  $("error").hidden = false;
  $("error").scrollIntoView({ behavior: "smooth", block: "center" });
}
function clearError() { $("error").hidden = true; }
function settings() {
  return {
    dataset: $("dataset").value, upload_id: uploadId, target: $("target").value,
    exclude: Array.from($("exclude").selectedOptions, (option) => option.value),
    stress: $("stress").checked,
    config: {
      seed: Number($("seed").value), test_size: Number($("test-size").value) / 100,
      folds: Number($("folds").value), scaler: $("scaler").value, pca: $("pca").checked,
      contamination: Number($("contamination").value), iqr_factor: Number($("iqr-factor").value),
      z_threshold: Number($("z-threshold").value), dbscan_eps: Number($("dbscan-eps").value),
      models: Array.from(document.querySelectorAll("#models input:checked"), (el) => el.value),
      methods: Array.from(document.querySelectorAll("#methods input:checked"), (el) => el.value),
    },
  };
}
function setBusy(value) {
  busy = value;
  $("experiment-form").inert = value;
  $("quick-demo").disabled = value;
  $("run-button").disabled = value;
  $("profile-button").disabled = value;
  $("run-button").textContent = value ? "Experiment running…" : "Run experiment →";
  $("loading").hidden = !value;
  if (value) { $("empty-state").hidden = true; $("results").hidden = true; }
}
function svg(content, width = 660, height = 360, label = "Data visualization") {
  return `<svg class="viz-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(label)}">${content}</svg>`;
}
function profileHTML(profile) {
  const tiles = [
    [profile.rows, "usable records"], [profile.features, "input features"],
    [profile.missing_cells, "missing feature cells"], [profile.duplicates_removed, "duplicates removed"],
    [profile.missing_targets_removed, "blank labels removed"], [Object.keys(profile.classes).length, "target classes"],
  ];
  return `<div class="quality-grid">${tiles.map(([value, label]) => `<div class="quality-item"><strong>${value}</strong>${label}</div>`).join("")}</div>
    <div class="class-pills">${Object.entries(profile.classes).map(([name, count]) => `<span>${esc(name)} <b>${count}</b></span>`).join("")}</div>
    <p class="helper">${profile.numeric_features} numeric + ${profile.categorical_features} categorical features. Missing values are imputed inside training partitions, never globally.</p>`;
}
async function inspect() {
  clearError();
  try {
    const data = await post("/api/profile", settings());
    $("profile-card").hidden = false;
    $("profile-content").innerHTML = `<h3>${esc(data.name)}</h3>${profileHTML(data.profile)}
      <p class="helper">${esc(data.notes)}</p><div class="table-scroll"><table class="profile-table"><thead><tr><th>Feature</th><th>Type</th><th>Missing</th><th>Unique</th></tr></thead><tbody>
      ${data.profile.columns.map((col) => `<tr><td>${esc(col.name)}</td><td>${esc(col.type)}</td><td>${col.missing}</td><td>${col.unique}</td></tr>`).join("")}</tbody></table></div>`;
  } catch (error) { showError(error); }
}
async function runExperiment(event) {
  if (event) event.preventDefault();
  if (busy) return;
  clearError();
  if (!$("experiment-form").reportValidity()) return;
  if (!settings().config.models.length || !settings().config.methods.length) {
    showError(new Error("Choose at least one classifier and one outlier detector."));
    return;
  }
  setBusy(true);
  try {
    current = await post("/api/run", settings());
    renderResults();
    try { await loadHistory(); }
    catch (error) { showError(new Error(`Experiment saved successfully, but recent history could not refresh: ${error.message}`)); }
  } catch (error) {
    showError(error);
    if (current) $("results").hidden = false;
    else $("empty-state").hidden = false;
  } finally { setBusy(false); }
}
function renderResults() {
  const report = current;
  const selected = report.results.find((row) => row.id === report.selected_id);
  const baseline = report.results.find((row) => row.id === report.baseline_id);
  const bestOutlier = report.results.find((row) => row.id === report.best_outlier_id);
  const outlierPreprocessed = bestOutlier && report.results.find((row) => row.model_key === bestOutlier.model_key && row.arm === "preprocessed");
  $("empty-state").hidden = true;
  $("profile-card").hidden = true;
  $("results").hidden = false;
  $("result-title").textContent = report.dataset;
  $("run-meta").textContent = `${report.split.train} train / ${report.split.test} test · ${report.config.folds}-fold CV · seed ${report.config.seed} · ${report.seconds}s · ${report.id.slice(0, 8)}`;
  $("bundle-link").href = `/api/runs/${report.id}/download/bundle`;
  const metricTiles = [
    ["SELECTED TEST ACCURACY", pct(selected.accuracy), "Winner chosen by training-CV F1"],
    ["VS SAME-MODEL BASELINE", pp(report.accuracy_delta), `Baseline: ${pct(baseline.accuracy)}`],
    ["CROSS-VALIDATION F1", pct(selected.cv_f1), `Fold SD: ${(selected.cv_std * 100).toFixed(2)} pp`],
    ["TRAINING ROWS REMOVED", selected.removed, `${selected.training_rows} retained · ${selected.treatment}`],
  ];
  $("metrics").innerHTML = metricTiles.map(([label, value, note]) => `<div class="metric-card"><div class="metric-label">${esc(label)}</div><strong>${esc(value)}</strong><small>${esc(note)}</small></div>`).join("");
  const conclusion = report.accuracy_delta > 0 ? "Measured improvement, not an assumption." : report.accuracy_delta < 0 ? "A useful negative result: more treatment was not better." : "No accuracy gain over the matched baseline.";
  $("result-insight").innerHTML = `<strong>${conclusion}</strong>${esc(selected.model)} + ${esc(selected.treatment)} was selected using training-only CV.
    ${bestOutlier ? `The best CV-ranked outlier arm was ${esc(bestOutlier.model)} + ${esc(bestOutlier.treatment)}: ${pp(report.outlier_accuracy_delta)} test accuracy versus its minimal baseline, and ${pp(bestOutlier.accuracy - outlierPreprocessed.accuracy)} versus the same preprocessed model without removal.` : "Every selected detector was rejected by the training-data safety checks; see Reproduce."}
    <div class="helper">This is one held-out estimate, not proof of statistical significance. Do not tune settings to this test set.</div>`;
  const comparable = report.results.filter((row) => row.model_key === selected.model_key);
  $("comparison-chart").innerHTML = comparable.map((row) => `<div class="bar-row"><span>${esc(row.treatment)}</span><div class="bar-track"><div class="bar-fill" style="width:${row.accuracy * 100}%;background:${row.arm === "baseline" ? "#b6c4db" : row.arm === "preprocessed" ? "#7293e5" : "#65afa5"}"></div></div><span class="bar-score">${pct(row.accuracy)}</span></div>`).join("");
  renderLeaderboard();
  $("quality-summary").innerHTML = profileHTML(report.profile) + `<p class="helper">${esc(report.notes)}</p>`;
  $("outlier-view").innerHTML = `<option value="none">No removal</option>` + Object.entries(report.outlier_views).map(([key, view]) => `<option value="${esc(key)}">${esc(view.name)} · ${view.removed} removed</option>`).join("");
  $("outlier-view").value = selected.method || bestOutlier?.method || "none";
  renderScatter();
  renderBoxes();
  renderCorrelation();
  renderEvaluation(selected);
  $("prediction-json").value = JSON.stringify(report.sample_input, null, 2);
  $("prediction-output").textContent = "";
  $("protocol-list").innerHTML = report.protocol.map((line) => `<li>${esc(line)}</li>`).join("");
  $("config-summary").textContent = JSON.stringify(report.config, null, 2);
  const exports = [["comparison", "Comparison CSV"], ["predictions", "Test predictions CSV"], ["audit", "Training outlier audit CSV"], ["processed", "Processed training CSV"], ["results", "Full results JSON"], ["model", "Trained pipeline (.joblib)"]];
  $("export-links").innerHTML = exports.map(([key, name]) => `<a class="button secondary" href="/api/runs/${report.id}/download/${key}">${name}<span>↓</span></a>`).join("");
  const warnings = report.warnings.map((text) => `<div class="warning-item">${esc(text)}</div>`);
  warnings.push(...report.rejected.map((item) => `<div class="warning-item"><b>${esc(item.model)} / ${esc(item.method)}</b><br>${esc(item.reason)}</div>`));
  $("warnings-card").hidden = !warnings.length;
  $("warnings-content").innerHTML = warnings.join("");
}
function renderLeaderboard() {
  if (!current) return;
  const sort = $("sort-results").value;
  const rows = [...current.results].sort((a, b) => b[sort] - a[sort]);
  $("leaderboard").innerHTML = rows.map((row) => `<tr class="${row.id === current.selected_id ? "winner-row" : ""}"><td><span class="model-title">${esc(row.model)}</span><span class="treatment-label">${esc(row.treatment)}</span>${row.id === current.selected_id ? '<span class="selected-badge">CV WINNER</span>' : ""}</td><td>${pct(row.cv_f1)}<span class="treatment-label"> ± ${(row.cv_std * 100).toFixed(1)}</span></td><td>${pct(row.accuracy)}</td><td>${pct(row.f1)}</td><td>${row.roc_auc.toFixed(3)}</td><td>${row.removed}</td></tr>`).join("");
}
function renderScatter() {
  const points = current.scatter;
  const view = current.outlier_views[$("outlier-view").value];
  const xmin = Math.min(...points.map((point) => point.x)), xmax = Math.max(...points.map((point) => point.x));
  const ymin = Math.min(...points.map((point) => point.y)), ymax = Math.max(...points.map((point) => point.y));
  const sx = (value) => 55 + (value - xmin) / (xmax - xmin || 1) * 570;
  const sy = (value) => 310 - (value - ymin) / (ymax - ymin || 1) * 275;
  let content = '<rect x="55" y="25" width="570" height="285" rx="8" fill="#fafcfe" stroke="#e5ecf5"/>';
  for (let i = 1; i < 5; i++) {
    content += `<path d="M55 ${25 + i * 57}H625M${55 + i * 114} 25V310" stroke="#e5ebf5" stroke-dasharray="3 5"/>`;
  }
  points.forEach((point, index) => {
    const x = sx(point.x), y = sy(point.y);
    const outlier = view ? view.mask[index] : false;
    const label = `Row ${point.row}: ${point.class}${outlier ? " / outlier" : ""}`;
    content += outlier
      ? `<path data-anomaly="true" d="M${x} ${y - 5}l5 5-5 5-5-5Z" fill="#dd8261" stroke="white" stroke-width=".8"><title>${esc(label)}</title></path>`
      : `<circle cx="${x}" cy="${y}" r="3.2" fill="${colors[current.classes.indexOf(point.class) % colors.length]}" fill-opacity=".7" stroke="white" stroke-width=".5"><title>${esc(label)}</title></circle>`;
  });
  content += '<text x="340" y="344" text-anchor="middle" font-size="11">Principal component 1</text><text x="16" y="170" text-anchor="middle" transform="rotate(-90 16 170)" font-size="11">Principal component 2</text>';
  $("scatter-chart").innerHTML = svg(content, 660, 355, "Training data projected to two principal components, with possible outliers highlighted") +
    `<div class="scatter-legend">${current.classes.map((label, index) => `<span><i style="background:${colors[index % colors.length]}"></i>${esc(label)}</span>`).join("")}<span><i style="background:#dd8261;border-radius:1px;transform:rotate(45deg)"></i>Possible outlier</span></div>`;
  $("scatter-treatment").textContent = view ? `${view.name}: ${view.removed} of ${points.length} training rows flagged. These are candidate anomalies, not confirmed errors.` : "No detector applied in this view.";
  const classCounts = view ? view.classes : current.classes.map((label) => ({
    class: label, total: points.filter((point) => point.class === label).length, removed: 0,
  }));
  $("class-impact").innerHTML = `<div class="table-scroll"><table><thead><tr><th>Class</th><th>Before</th><th>Removed</th><th>Retained</th><th>Removal rate</th></tr></thead><tbody>${classCounts.map((item) => `<tr><td>${esc(item.class)}</td><td>${item.total}</td><td>${item.removed}</td><td>${item.total - item.removed}</td><td>${pct(item.removed / item.total)}</td></tr>`).join("")}</tbody></table></div>`;
}
function renderBoxes() {
  const rows = current.quantiles_by_method[$("outlier-view").value].filter((row) => row.before && row.after);
  if (!rows.length) { $("box-chart").textContent = "No numeric distribution summaries are available."; return; }
  let content = '<text x="190" y="17" font-size="10">Gray: original training data / Teal: retained by chosen detector</text>';
  rows.forEach((row, index) => {
    const y = 47 + index * 62;
    const lower = Math.min(row.before[0], row.after[0]), upper = Math.max(row.before[4], row.after[4]);
    const scale = (v) => 190 + (v - lower) / (upper - lower || 1) * 420;
    content += `<text x="3" y="${y + 10}" font-size="10">${esc(row.feature.length > 27 ? row.feature.slice(0, 26) + "…" : row.feature)}</text>`;
    [row.before, row.after].forEach((values, arm) => {
      const yy = y + arm * 19, color = arm ? "#8fc6bd" : "#d5dfea";
      content += `<path d="M${scale(values[0])} ${yy + 6}H${scale(values[4])}M${scale(values[0])} ${yy}v12M${scale(values[4])} ${yy}v12" stroke="#9aaac0" fill="none"/><rect x="${scale(values[1])}" y="${yy}" width="${Math.max(1, scale(values[3]) - scale(values[1]))}" height="12" rx="2" fill="${color}" stroke="#8fa4be" stroke-width=".6"/><path d="M${scale(values[2])} ${yy}v12" stroke="#6580a5" stroke-width="1.5"/>`;
    });
  });
  $("box-chart").innerHTML = svg(content, 660, rows.length * 62 + 43, "Before and after box summaries of numeric training features");
}
function renderCorrelation() {
  const matrix = current.correlation;
  $("correlation-chart").innerHTML = `<table class="heatmap"><thead><tr><th></th>${matrix.columns.map((col) => `<th>${esc(col)}</th>`).join("")}</tr></thead><tbody>${matrix.values.map((values, index) => `<tr><th>${esc(matrix.columns[index])}</th>${values.map((v) => `<td style="background:${v >= 0 ? `rgba(82,123,234,${Math.abs(v) * .48})` : `rgba(221,130,97,${Math.abs(v) * .48})`}">${v.toFixed(2)}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}
function renderEvaluation(selected) {
  const peak = Math.max(...selected.confusion.flat());
  $("confusion-chart").innerHTML = `<table class="confusion-table"><thead><tr><th>Actual ↓<br>Predicted →</th>${current.classes.map((name) => `<th>${esc(name)}</th>`).join("")}</tr></thead><tbody>${selected.confusion.map((row, index) => `<tr><td>${esc(current.classes[index])}</td>${row.map((value) => `<td style="background:rgba(82,123,234,${.04 + .36 * value / (peak || 1)})">${value}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
  $("metric-details").innerHTML = [["Macro precision", selected.precision], ["Macro recall", selected.recall], ["Macro F1", selected.f1], ["Balanced accuracy", selected.balanced_accuracy], ["ROC-AUC (OVR for multiclass)", selected.roc_auc]].map(([name, value]) => `<div class="detail-metric"><span>${name}</span><b>${pct(value)}</b></div>`).join("");
  if (!current.roc) {
    $("roc-chart").innerHTML = '<p class="helper">Multiclass ROC-AUC uses macro one-vs-rest. The curve is shown only for binary datasets.</p>';
    return;
  }
  const path = current.roc.fpr.map((fpr, index) => `${index ? "L" : "M"}${40 + 250 * fpr},${200 - 170 * current.roc.tpr[index]}`).join(" ");
  $("roc-chart").innerHTML = svg(`<path d="M40 30V200H290" stroke="#ccd8e9" fill="none"/><path d="M40 200L290 30" stroke="#d1dceb" stroke-dasharray="4 4"/><path d="${path}" stroke="#527bea" stroke-width="2.5" fill="none" stroke-linejoin="round"/><text x="160" y="225" font-size="10" text-anchor="middle">False positive rate</text><text x="8" y="115" font-size="10" transform="rotate(-90 8 115)" text-anchor="middle">True positive rate</text>`, 320, 240, "ROC curve of selected model") + `<p class="helper">Positive class: ${esc(current.roc.positive_class)}</p>`;
}
async function loadHistory() {
  const data = await api("/api/runs");
  $("history").innerHTML = data.runs.length ? data.runs.map((run) => `<button class="history-row" type="button" data-run="${esc(run.id)}"><span><b>${esc(run.dataset)}</b><small>${esc(run.model)} · ${new Date(run.created_at).toLocaleString()}</small></span><span class="history-score">${pct(run.accuracy)} ↗</span></button>`).join("") : '<p class="helper">No saved experiments yet. Run the demo to create one.</p>';
}
async function uploadCSV() {
  clearError();
  const file = $("csv-file").files[0];
  if (!file) return;
  const data = new FormData();
  data.append("file", file);
  uploadId = null;
  $("upload-status").textContent = "Inspecting CSV…";
  $("run-button").disabled = true;
  $("profile-button").disabled = true;
  try {
    const result = await api("/api/upload", { method: "POST", body: data });
    uploadId = result.upload_id;
    const options = result.columns.map((col) => `<option value="${esc(col.name)}">${esc(col.name)} · ${col.unique} unique</option>`).join("");
    $("target").innerHTML = options;
    $("target").value = result.columns[result.columns.length - 1].name;
    $("exclude").innerHTML = result.columns.map((col) => `<option value="${esc(col.name)}">${esc(col.name)}</option>`).join("");
    $("upload-status").textContent = `${result.rows} records · ${result.columns.length} columns. Confirm the target and exclude identifiers.`;
  } catch (error) { $("upload-status").textContent = "Upload failed."; showError(error); }
  finally { $("run-button").disabled = false; $("profile-button").disabled = false; }
}
async function predict() {
  clearError();
  $("prediction-output").textContent = "";
  const runId = current.id;
  $("predict-button").disabled = true;
  try {
    let rows;
    try { rows = JSON.parse($("prediction-json").value); }
    catch { throw new Error("Prediction input must be a valid JSON array."); }
    const result = await post(`/api/runs/${runId}/predict`, { rows });
    if (current.id !== runId) return;
    $("prediction-output").innerHTML = result.predictions.map((record, index) => `<div class="prediction-card">Record ${index + 1} → <strong>${esc(record.class)}</strong><div class="helper">${Object.entries(record.probabilities).map(([label, probability]) => `${esc(label)}: ${pct(probability)}`).join(" · ")}</div></div>`).join("");
  } catch (error) { showError(error); }
  finally { $("predict-button").disabled = false; }
}
async function init() {
  try {
    catalog = await api("/api/catalog");
    for (const [container, values] of [["models", catalog.models], ["methods", catalog.methods]]) {
      $(container).innerHTML = Object.entries(values).map(([key, label]) => `<label><input type="checkbox" value="${key}" ${["entropy", "mlp"].includes(key) ? "" : "checked"}><span>${esc(label)}</span></label>`).join("");
    }
    $("dataset-description").textContent = catalog.datasets.demo.description;
    $("dataset").dispatchEvent(new Event("change"));
    await loadHistory();
  } catch (error) { showError(error); }
}
$("experiment-form").addEventListener("submit", runExperiment);
$("quick-demo").addEventListener("click", () => { $("dataset").value = "demo"; $("dataset").dispatchEvent(new Event("change")); runExperiment(); });
$("profile-button").addEventListener("click", inspect);
$("test-size").addEventListener("input", () => { $("test-size-value").textContent = `${$("test-size").value}%`; });
$("dataset").addEventListener("change", () => {
  const upload = $("dataset").value === "upload";
  $("upload-controls").hidden = !upload;
  $("stress-row").hidden = upload || $("dataset").value === "demo";
  $("dataset-description").textContent = upload ? "Your data stays in the local project folder. Choose a categorical target." : catalog?.datasets[$("dataset").value]?.description || "";
});
$("csv-file").addEventListener("change", uploadCSV);
$("sort-results").addEventListener("change", renderLeaderboard);
$("outlier-view").addEventListener("change", () => { renderScatter(); renderBoxes(); });
$("predict-button").addEventListener("click", predict);
$("refresh-history").addEventListener("click", () => loadHistory().catch(showError));
$("history").addEventListener("click", async (event) => {
  const button = event.target.closest("[data-run]");
  if (!button || busy) return;
  clearError();
  try { current = await api(`/api/runs/${button.dataset.run}`); renderResults(); }
  catch (error) { showError(error); }
});
document.querySelectorAll("[data-tab]").forEach((button) => button.addEventListener("click", () => {
  document.querySelectorAll("[data-tab]").forEach((tab) => tab.classList.toggle("active", tab === button));
  document.querySelectorAll(".tab-content").forEach((section) => { section.hidden = section.id !== `tab-${button.dataset.tab}`; });
}));
init();
