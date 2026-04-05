import { useEffect, useState } from "react";
import { TAB_STORAGE_KEY } from "./constants.js";
import { ImageTab } from "./ImageTab.jsx";
import { TextTab } from "./TextTab.jsx";

function readStoredTab() {
  try {
    const t = localStorage.getItem(TAB_STORAGE_KEY);
    return t === "text" ? "text" : "image";
  } catch {
    return "image";
  }
}

export default function App() {
  const [tab, setTab] = useState(readStoredTab);

  useEffect(() => {
    try {
      localStorage.setItem(TAB_STORAGE_KEY, tab);
    } catch {
      /* ignore */
    }
  }, [tab]);

  return (
    <div className="demo-shell">
      <header className="demo-header">
        <a className="demo-logo" href="#" onClick={(e) => e.preventDefault()}>
          <span>CO3133</span> — Deep Learning
        </a>
        <nav className="demo-nav" aria-label="Course links">
          <a href="https://github.com" target="_blank" rel="noreferrer">
            GitHub ↗
          </a>
        </nav>
      </header>

      <main className="demo-page">
        <div className="demo-hero-tag">Assignment 01 · Live demo</div>

        <div className="demo-tabs" role="tablist" aria-label="Chọn bài toán">
          <button
            type="button"
            role="tab"
            aria-selected={tab === "image"}
            className={
              tab === "image" ? "demo-tab demo-tab--active" : "demo-tab"
            }
            onClick={() => setTab("image")}
          >
            Ảnh · CIFAR-100
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={tab === "text"}
            className={tab === "text" ? "demo-tab demo-tab--active" : "demo-tab"}
            onClick={() => setTab("text")}
          >
            Văn bản · News
          </button>
        </div>

        {tab === "image" ? <ImageTab /> : <TextTab />}
      </main>

      <footer className="demo-footer">
        CO3133 · Assignment 01 demo (ảnh + văn bản) · HCMUT — Lecturer: Lê Thành Sách
      </footer>
    </div>
  );
}
