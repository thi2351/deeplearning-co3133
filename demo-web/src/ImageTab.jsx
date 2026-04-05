import { useCallback, useEffect, useState } from "react";
import {
  apiImage,
  fetchJson,
  modelDisplayName,
  prettyClassName,
} from "./api.js";
import { MODEL_STORAGE_KEY } from "./constants.js";
import { TopKChart } from "./TopKChart.jsx";

function readStoredModelId() {
  try {
    return localStorage.getItem(MODEL_STORAGE_KEY) || "";
  } catch {
    return "";
  }
}

export function ImageTab() {
  const datasetThumbUrl = useCallback(
    (index) => apiImage(`/api/dataset-image?index=${index}`),
    []
  );

  const [datasetSamples, setDatasetSamples] = useState([]);
  const [cifarTestTotal, setCifarTestTotal] = useState(null);
  const [samplesLoading, setSamplesLoading] = useState(true);
  const [samplesErr, setSamplesErr] = useState(null);
  const [modelOptions, setModelOptions] = useState([]);
  const [modelsErr, setModelsErr] = useState(null);
  const [selectedModelId, setSelectedModelId] = useState(readStoredModelId);
  const [indexInput, setIndexInput] = useState("");
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [sampleId, setSampleId] = useState(null);
  const [truth, setTruth] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  useEffect(() => {
    fetchJson(apiImage("/api/models"))
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
      localStorage.setItem(MODEL_STORAGE_KEY, selectedModelId);
    } catch {
      /* ignore */
    }
  }, [selectedModelId]);

  useEffect(() => {
    setSamplesLoading(true);
    setSamplesErr(null);
    fetchJson(apiImage("/api/dataset-samples?count=9&seed=42"))
      .then((data) => {
        if (data.error) throw new Error(data.error);
        if (!data.samples) throw new Error("Thiếu trường samples");
        setDatasetSamples(data.samples);
        if (typeof data.total === "number") setCifarTestTotal(data.total);
      })
      .catch((e) => {
        setDatasetSamples([]);
        setSamplesErr(
          `${e?.message || e} Lần đầu tải CIFAR-100 có thể mất vài phút (~170MB).`
        );
      })
      .finally(() => setSamplesLoading(false));
  }, []);

  const onFile = useCallback((e) => {
    const f = e.target.files?.[0];
    setSampleId(null);
    setIndexInput("");
    setTruth(null);
    setFile(f || null);
    setResult(null);
    setError(null);
    setPreview((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return f ? URL.createObjectURL(f) : null;
    });
  }, []);

  const loadCifarByIndex = useCallback(
    async (rawIndex, truthFromList = null) => {
      const idx = Number(rawIndex);
      if (!Number.isFinite(idx) || !Number.isInteger(idx)) {
        setError("Index phải là số nguyên.");
        return;
      }
      if (idx < 0) {
        setError("Index không được âm.");
        return;
      }
      if (cifarTestTotal != null && idx >= cifarTestTotal) {
        setError(`Index phải nhỏ hơn ${cifarTestTotal} (test set CIFAR-100).`);
        return;
      }

      setResult(null);
      setError(null);
      try {
        const res = await fetch(datasetThumbUrl(idx));
        if (!res.ok) {
          const j = await res.json().catch(() => ({}));
          throw new Error(j.error || "Không tải được ảnh từ tập test");
        }
        const blob = await res.blob();
        const name = `cifar100_test_${idx}.png`;
        const f = new File([blob], name, { type: blob.type || "image/png" });
        setSampleId(String(idx));
        setIndexInput(String(idx));
        setFile(f);
        setPreview((prev) => {
          if (prev) URL.revokeObjectURL(prev);
          return URL.createObjectURL(blob);
        });

        if (truthFromList?.label != null) {
          setTruth({
            label: truthFromList.label,
            labelId: truthFromList.label_id ?? truthFromList.labelId,
          });
        } else {
          const meta = await fetchJson(apiImage(`/api/dataset-label?index=${idx}`));
          if (meta.error) throw new Error(meta.error);
          setTruth({ label: meta.label, labelId: meta.label_id });
        }
      } catch (err) {
        setFile(null);
        setPreview((prev) => {
          if (prev) URL.revokeObjectURL(prev);
          return null;
        });
        setSampleId(null);
        setTruth(null);
        setError(err.message || String(err));
      }
    },
    [cifarTestTotal, datasetThumbUrl]
  );

  const pickSample = useCallback(
    async (item) => {
      setIndexInput(String(item.index));
      await loadCifarByIndex(item.index, {
        label: item.label,
        label_id: item.label_id,
      });
    },
    [loadCifarByIndex]
  );

  const predict = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    const form = new FormData();
    form.append("image", file);
    if (selectedModelId) form.append("model", selectedModelId);
    const q = new URLSearchParams({ topk: "8" });
    if (selectedModelId) q.set("model", selectedModelId);
    try {
      const res = await fetch(apiImage(`/api/predict?${q.toString()}`), {
        method: "POST",
        body: form,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || res.statusText);
      setResult(data);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  }, [file, selectedModelId]);

  const indexMaxHint =
    cifarTestTotal != null ? `0 … ${cifarTestTotal - 1}` : "0 … 9999";

  return (
    <>
      <h1 className="demo-hero-title">
        CIFAR-100<br />
        <em>Classifier</em>
      </h1>

      <div className="demo-divider" />

      <div className="demo-section-label">Input</div>
      <section className="demo-card">
        <div className="demo-model-row">
          <label className="demo-field-label" htmlFor="model-pick">
            Model
          </label>
          <select
            id="model-pick"
            className="demo-select demo-select--wide"
            value={selectedModelId}
            onChange={(e) => setSelectedModelId(e.target.value)}
            aria-label="Chọn model"
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
            <p className="demo-hint">Chưa có checkpoint trong danh sách.</p>
          )}
        </div>

        <label className="demo-field-label" htmlFor="file-up">
          Tải ảnh từ máy
        </label>
        <input
          id="file-up"
          className="demo-file-input"
          type="file"
          accept="image/*"
          onChange={onFile}
        />

        <p className="demo-field-label" style={{ marginTop: 28 }}>
          Hoặc nhập index · tập test CIFAR-100{" "}
          <span className="demo-hint-inline">({indexMaxHint})</span>
        </p>
        <div className="demo-index-row">
          <input
            id="cifar-index"
            className="demo-input-num"
            type="number"
            inputMode="numeric"
            min={0}
            max={cifarTestTotal != null ? cifarTestTotal - 1 : undefined}
            placeholder="vd. 42"
            value={indexInput}
            onChange={(e) => setIndexInput(e.target.value)}
            aria-label="Index ảnh test CIFAR-100"
          />
          <button
            type="button"
            className="demo-btn-secondary"
            onClick={() => loadCifarByIndex(indexInput.trim())}
          >
            Tải ảnh theo index
          </button>
        </div>

        <p className="demo-field-label" style={{ marginTop: 28 }}>
          Hoặc chọn từ gallery ngẫu nhiên
        </p>
        {samplesLoading && <p className="demo-hint">Đang tải danh sách ảnh…</p>}
        {samplesErr && (
          <div className="demo-alert demo-alert--error">{samplesErr}</div>
        )}
        <div className="demo-sample-grid">
          {datasetSamples.map((item) => (
            <button
              key={item.index}
              type="button"
              className={
                sampleId === String(item.index)
                  ? "demo-sample-tile demo-sample-tile--selected"
                  : "demo-sample-tile"
              }
              onClick={() => pickSample(item)}
            >
              <img
                src={datasetThumbUrl(item.index)}
                alt={item.label}
                loading="lazy"
              />
              <span className="demo-sample-cap" title={item.label}>
                {prettyClassName(item.label)}
              </span>
              <span className="demo-sample-idx">#{item.index}</span>
            </button>
          ))}
        </div>

        {preview && (
          <img className="demo-preview" src={preview} alt="Selected preview" />
        )}
        <div className="demo-actions">
          <button
            type="button"
            className="demo-btn-primary"
            onClick={predict}
            disabled={!file || loading}
          >
            {loading ? "Đang chạy…" : "Classify"}
          </button>
          {!file && (
            <span className="demo-hint">Chọn ảnh hoặc ô mẫu trước khi classify.</span>
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
