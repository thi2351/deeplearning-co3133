import { useCallback, useEffect, useState } from "react";
import { apiMultimodal, fetchJsonOk, prettyClassName } from "./api.js";
import { TopKChart } from "./TopKChart.jsx";

export function MultimodalTab() {
  const [datasetSamples, setDatasetSamples] = useState([]);
  const [samplesLoading, setSamplesLoading] = useState(true);
  const [samplesErr, setSamplesErr] = useState(null);
  const [sampleId, setSampleId] = useState(null);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [text, setText] = useState("");
  const [topk, setTopk] = useState("5");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [reportLoading, setReportLoading] = useState(false);
  const [reportError, setReportError] = useState(null);
  const [report, setReport] = useState(null);

  useEffect(() => {
    setSamplesLoading(true);
    setSamplesErr(null);
    fetchJsonOk(apiMultimodal("/dataset-samples?count=9"))
      .then((data) => {
        setDatasetSamples(Array.isArray(data.samples) ? data.samples : []);
      })
      .catch((e) => {
        setDatasetSamples([]);
        setSamplesErr(e?.message || String(e));
      })
      .finally(() => setSamplesLoading(false));
  }, []);

  const onFile = useCallback((e) => {
    const f = e.target.files?.[0];
    setSampleId(null);
    setFile(f || null);
    setResult(null);
    setReport(null);
    setReportError(null);
    setError(null);
    setPreview((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return f ? URL.createObjectURL(f) : null;
    });
  }, []);

  const pickSample = useCallback(async (item) => {
    try {
      const r = await fetch(item.image_url);
      if (!r.ok) throw new Error(`Không tải được ảnh mẫu (${r.status})`);
      const blob = await r.blob();
      const ext = blob.type.includes("png") ? "png" : "jpg";
      const f = new File([blob], `${item.id || "sample"}.${ext}`, { type: blob.type || "image/png" });

      setSampleId(String(item.id ?? ""));
      setFile(f);
      setText(item.text || "");
      setResult(null);
      setReport(null);
      setReportError(null);
      setError(null);
      setPreview((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return URL.createObjectURL(blob);
      });
    } catch (e) {
      setError(e?.message || String(e));
    }
  }, []);

  const predict = useCallback(async () => {
    if (!file) {
      setError("Vui lòng chọn hình ảnh");
      return;
    }
    if (!text.trim()) {
      setError("Vui lòng nhập mô tả văn bản");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);
    setReport(null);
    setReportError(null);

    try {
      const formData = new FormData();
      formData.append("image", file);
      formData.append("text", text);

      const q = new URLSearchParams({ topk: String(topk) });
      const data = await fetchJsonOk(
        apiMultimodal(`/predict?${q.toString()}`),
        {
          method: "POST",
          body: formData,
        }
      );

      if (data.predictions) {
        setResult(data);
      } else {
        throw new Error(data.error || "Invalid response");
      }
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  }, [file, text, topk]);

  const runReport = useCallback(async () => {
    if (!file) {
      setReportError("Vui lòng chọn hình ảnh");
      return;
    }
    if (!text.trim()) {
      setReportError("Vui lòng nhập mô tả văn bản");
      return;
    }

    setReportLoading(true);
    setReportError(null);
    setReport(null);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("image", file);
      formData.append("text", text);
      formData.append("topk", String(topk));
      formData.append("shots", "1");

      const data = await fetchJsonOk(apiMultimodal("/report"), {
        method: "POST",
        body: formData,
      });

      if (!data?.methods) {
        throw new Error("Report response is missing methods");
      }
      setReport(data);
      setResult(null);
    } catch (err) {
      setReportError(err.message || String(err));
    } finally {
      setReportLoading(false);
    }
  }, [file, text, topk]);

  const renderMethodCard = (title, block) => {
    const rows = Array.isArray(block?.topk) ? block.topk : [];
    const top1 = block?.top1;
    return (
      <section className="demo-card" style={{ marginBottom: 0 }}>
        <div className="demo-section-label">{title}</div>
        <p className="demo-hint" style={{ marginTop: 0 }}>
          Top-1: {top1 ? prettyClassName(top1.class_name) : "-"}
        </p>
        <p className="demo-hint" style={{ marginTop: 0 }}>
          Latency: {Number(block?.latency_ms || 0).toFixed(1)} ms
        </p>
        <TopKChart
          rows={rows.map((p, idx) => ({
            id: idx,
            name: prettyClassName(p.class_name),
            prob: p.score,
          }))}
        />
      </section>
    );
  };

  return (
    <>
      <h1 className="demo-hero-title">
        Food Classification<br />
        <em>Multimodal (CLIP + Text)</em>
      </h1>

      <div className="demo-divider" />

      <div className="demo-section-label">Input</div>
      <section className="demo-card">
        {/* Image Upload */}
        <div className="demo-model-row">
          <label className="demo-field-label" htmlFor="image-upload">
            Ảnh món ăn
          </label>
          <input
            id="image-upload"
            type="file"
            accept="image/*"
            onChange={onFile}
            className="demo-input"
            aria-label="Chọn ảnh"
          />
          {file && <p className="demo-hint">{file.name}</p>}
        </div>

        {/* Text Description */}
        <label className="demo-field-label" htmlFor="food-text">
          Mô tả món ăn
        </label>
        <textarea
          id="food-text"
          className="demo-textarea"
          placeholder="Mô tả loại thực phẩm, chiều thơm, vị...vv"
          value={text}
          onChange={(e) => {
            setText(e.target.value);
            setResult(null);
            setError(null);
          }}
          rows="3"
        />

        {/* Top-K */}
        <div className="demo-model-row">
          <label className="demo-field-label" htmlFor="topk-mm">
            Top K kết quả
          </label>
          <input
            id="topk-mm"
            type="number"
            min="1"
            max="20"
            value={topk}
            onChange={(e) => setTopk(e.target.value)}
            className="demo-input"
            style={{ width: "80px" }}
          />
        </div>

        {error && <p className="demo-hint demo-hint--error">{error}</p>}

        <p className="demo-field-label" style={{ marginTop: 24 }}>
          Hoặc chọn sample demo
        </p>
        {samplesLoading && <p className="demo-hint">Đang tải mẫu…</p>}
        {samplesErr && <div className="demo-alert demo-alert--error">{samplesErr}</div>}
        <div className="demo-sample-grid">
          {datasetSamples.map((item) => {
            const key = String(item.id ?? "");
            return (
              <button
                key={key}
                type="button"
                className={
                  sampleId === key
                    ? "demo-sample-tile demo-sample-tile--selected"
                    : "demo-sample-tile"
                }
                onClick={() => pickSample(item)}
                title={item.text || "sample"}
              >
                <img src={item.image_url} alt={item.label || key} />
                <span className="demo-sample-cap">{item.label || key}</span>
                <span className="demo-sample-idx">#{key}</span>
              </button>
            );
          })}
        </div>

        <button
          type="button"
          disabled={loading}
          onClick={predict}
          className="demo-button demo-button--primary"
        >
          {loading ? "Processing..." : "Predict"}
        </button>
        <button
          type="button"
          disabled={reportLoading}
          onClick={runReport}
          className="demo-btn-secondary"
          style={{ marginLeft: 8 }}
        >
          {reportLoading ? "Running report..." : "Run 3-way report"}
        </button>
        {reportError && <p className="demo-hint demo-hint--error">{reportError}</p>}
      </section>

      {/* Preview */}
      {preview && (
        <section className="demo-card">
          <div className="demo-section-label">Preview</div>
          <img
            src={preview}
            alt="Preview"
            style={{
              maxWidth: "100%",
              maxHeight: "300px",
              borderRadius: "4px",
            }}
          />
        </section>
      )}

      {/* Results */}
      {result && result.predictions && (
        <section className="demo-card">
          <div className="demo-section-label">Results</div>
          <TopKChart
            rows={result.predictions.map((p, idx) => ({
              id: idx,
              name: prettyClassName(p.class_name),
              prob: p.score,
            }))}
          />
          <table className="demo-table">
            <thead>
              <tr>
                <th>Class</th>
                <th style={{ textAlign: "right" }}>Score</th>
              </tr>
            </thead>
            <tbody>
              {result.predictions.map((pred, i) => (
                <tr key={i}>
                  <td>{prettyClassName(pred.class_name)}</td>
                  <td style={{ textAlign: "right" }}>
                    {(pred.score * 100).toFixed(2)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {report?.methods && (
        <>
          <div className="demo-section-label">Comparison Report</div>
          <div
            style={{
              display: "grid",
              gap: 16,
              gridTemplateColumns: "1fr",
            }}
          >
            {renderMethodCard("Zero-shot (CLIP)", report.methods.zero_shot)}
            {renderMethodCard("Few-shot (Prototype)", report.methods.few_shot)}
            {renderMethodCard("Full model (Head)", report.methods.full_model)}
          </div>
        </>
      )}
    </>
  );
}
