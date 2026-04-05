import { useCallback, useEffect, useState } from "react";
import { apiText, fetchJsonOk, modelDisplayName, prettyClassName } from "./api.js";
import { TEXT_MODEL_STORAGE_KEY } from "./constants.js";
import { TopKChart } from "./TopKChart.jsx";

function readStoredTextModelId() {
  try {
    return localStorage.getItem(TEXT_MODEL_STORAGE_KEY) || "";
  } catch {
    return "";
  }
}

export function TextTab() {
  const [datasetSamples, setDatasetSamples] = useState([]);
  const [samplesLoading, setSamplesLoading] = useState(true);
  const [samplesErr, setSamplesErr] = useState(null);
  const [samplesWarn, setSamplesWarn] = useState(null);
  const [modelOptions, setModelOptions] = useState([]);
  const [modelsErr, setModelsErr] = useState(null);
  const [selectedModelId, setSelectedModelId] = useState(readStoredTextModelId);
  const [bodyText, setBodyText] = useState("");
  const [sampleKey, setSampleKey] = useState(null);
  const [truth, setTruth] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  useEffect(() => {
    fetchJsonOk(apiText("/api/models"))
      .then((d) => {
        setModelOptions(Array.isArray(d.models) ? d.models : []);
        setModelsErr(null);
      })
      .catch((e) => {
        setModelOptions([]);
        setModelsErr(e?.message || String(e));
      });
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(TEXT_MODEL_STORAGE_KEY, selectedModelId);
    } catch {
      /* ignore */
    }
  }, [selectedModelId]);

  useEffect(() => {
    setSamplesLoading(true);
    setSamplesErr(null);
    setSamplesWarn(null);
    fetchJsonOk(apiText("/api/dataset-samples?count=9"))
      .then((data) => {
        if (data.warning) setSamplesWarn(data.warning);
        setDatasetSamples(Array.isArray(data.samples) ? data.samples : []);
      })
      .catch((e) => {
        setDatasetSamples([]);
        setSamplesErr(e?.message || String(e));
      })
      .finally(() => setSamplesLoading(false));
  }, []);

  const pickSample = useCallback((item) => {
    setBodyText(item.text || "");
    setSampleKey(String(item.id ?? item.index ?? ""));
    setTruth(
      item.label != null
        ? { label: item.label, labelId: item.label_id ?? item.labelId }
        : null
    );
    setResult(null);
    setError(null);
  }, []);

  const predict = useCallback(async () => {
    const text = bodyText.trim();
    if (!text) return;
    setLoading(true);
    setError(null);
    setResult(null);
    const q = new URLSearchParams({ topk: "8" });
    if (selectedModelId) q.set("model", selectedModelId);
    try {
      const data = await fetchJsonOk(apiText(`/api/predict?${q.toString()}`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, model: selectedModelId || undefined }),
      });
      setResult(data);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  }, [bodyText, selectedModelId]);

  return (
    <>
      <h1 className="demo-hero-title">
        News topic<br />
        <em>Classifier</em>
      </h1>

      <div className="demo-divider" />

      <div className="demo-section-label">Input</div>
      <section className="demo-card">
        <div className="demo-model-row">
          <label className="demo-field-label" htmlFor="model-pick-text">
            Model
          </label>
          <select
            id="model-pick-text"
            className="demo-select demo-select--wide"
            value={selectedModelId}
            onChange={(e) => setSelectedModelId(e.target.value)}
            aria-label="Chọn model văn bản"
          >
            <option value="">Mặc định server</option>
            {modelOptions.map((m) => (
              <option key={m.id} value={m.id}>
                {modelDisplayName(m)}
              </option>
            ))}
          </select>
          {modelsErr && (
            <p className="demo-hint demo-hint--warn">{modelsErr}</p>
          )}
          {!modelsErr && modelOptions.length === 0 && (
            <p className="demo-hint">Chưa có checkpoint .pth hợp lệ.</p>
          )}
        </div>

        <label className="demo-field-label" htmlFor="body-text">
          Nội dung văn bản
        </label>
        <textarea
          id="body-text"
          className="demo-textarea"
          placeholder="Dán đoạn tin hoặc chọn mẫu bên dưới…"
          value={bodyText}
          onChange={(e) => {
            setBodyText(e.target.value);
            setSampleKey(null);
            setTruth(null);
            setResult(null);
            setError(null);
          }}
          spellCheck="false"
        />

        <p className="demo-field-label" style={{ marginTop: 28 }}>
          Hoặc chọn mẫu có nhãn gốc (demo)
        </p>
        {samplesLoading && <p className="demo-hint">Đang tải mẫu…</p>}
        {samplesWarn && (
          <p className="demo-hint demo-hint--warn">{samplesWarn}</p>
        )}
        {samplesErr && (
          <div className="demo-alert demo-alert--error">{samplesErr}</div>
        )}
        <div className="demo-sample-grid demo-sample-grid--text">
          {datasetSamples.map((item) => {
            const key = String(item.id ?? item.index);
            return (
              <button
                key={key}
                type="button"
                className={
                  sampleKey === key
                    ? "demo-sample-tile demo-sample-tile--text demo-sample-tile--selected"
                    : "demo-sample-tile demo-sample-tile--text"
                }
                onClick={() => pickSample(item)}
              >
                <span className="demo-sample-text-prev">
                  {item.preview || item.text}
                </span>
                <span className="demo-sample-cap" title={item.label}>
                  {item.label ? prettyClassName(item.label) : "—"}
                </span>
                <span className="demo-sample-idx">#{key}</span>
              </button>
            );
          })}
        </div>

        <div className="demo-actions" style={{ marginTop: 20 }}>
          <button
            type="button"
            className="demo-btn-primary"
            onClick={predict}
            disabled={!bodyText.trim() || loading}
          >
            {loading ? "Đang chạy…" : "Classify"}
          </button>
          {!bodyText.trim() && (
            <span className="demo-hint">Nhập hoặc chọn một đoạn văn trước khi classify.</span>
          )}
        </div>
        {error && <div className="demo-alert demo-alert--error">{error}</div>}
      </section>

      {result && (
        <>
          <div className="demo-section-label">Prediction</div>
          <section className="demo-card">
            <dl className="demo-pred-summary">
              <div className="demo-pred-row">
                <dt className="demo-pred-k">Predicted:</dt>
                <dd className="demo-pred-v">
                  {prettyClassName(result.top1_name)}
                </dd>
              </div>
              <div className="demo-pred-row">
                <dt className="demo-pred-k">Actual:</dt>
                <dd className="demo-pred-v">
                  {truth?.label != null ? prettyClassName(truth.label) : "—"}
                </dd>
              </div>
            </dl>
            <TopKChart rows={result.topk} />
          </section>
        </>
      )}
    </>
  );
}
