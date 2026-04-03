/** Màu theo xác suất: thấp → đỏ (hue 0), cao → xanh (hue 120). */
export function colorForProbability(p) {
  const t = Math.max(0, Math.min(1, Number(p) || 0));
  const h = Math.round(t * 120);
  return `hsl(${h} 72% 40%)`;
}

function formatLabel(name) {
  return String(name || "").replace(/_/g, " ");
}

function truncate(s, max) {
  const t = formatLabel(s);
  if (t.length <= max) return t;
  return `${t.slice(0, max - 1)}…`;
}

const BAR_X = 130;
const BAR_W = 320;

/** @param {{ name: string, prob: number, id: number }[]} rows */
export function TopKChart({ rows }) {
  const data = (rows || []).slice(0, 8);
  if (!data.length) return null;

  return (
    <div className="demo-chart demo-chart--hbar" role="img" aria-label="Biểu đồ cột ngang top 8">
      <svg className="demo-chart-svg" viewBox="0 0 480 200" preserveAspectRatio="xMidYMid meet">
        {data.map((row, i) => {
          const y = 8 + i * 24;
          const p = Math.max(0, Math.min(1, Number(row.prob) || 0));
          const w = BAR_W * p;
          const fill = colorForProbability(row.prob);
          return (
            <g key={row.id}>
              <text x={0} y={y + 12} className="demo-chart-label" fontSize="11">
                {truncate(row.name, 18)}
              </text>
              <rect
                x={BAR_X}
                y={y + 2}
                width={BAR_W}
                height={16}
                rx={4}
                fill="var(--bg2)"
                stroke="var(--border)"
                strokeWidth="0.75"
              />
              <rect
                x={BAR_X}
                y={y + 2}
                width={w}
                height={16}
                rx={4}
                fill={fill}
                opacity={0.95}
              />
              <text x={456} y={y + 13} className="demo-chart-pct" fontSize="11" textAnchor="end">
                {(row.prob * 100).toFixed(1)}%
              </text>
            </g>
          );
        })}
      </svg>
      <p className="demo-chart-footnote">
        Chiều dài thanh theo xác suất so với 100% (100% = hết ô); màu: cao → xanh, thấp → đỏ.
      </p>
    </div>
  );
}
