import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";

const API_URL = "http://localhost:8000";

// ── Mock data for demo mode (when API isn't running) ─────────────────────────
const MOCK_RESPONSE = {
  query:
    "How does PacBio HiFi compare to Nanopore for structural variant detection?",
  answer:
    "Based on the retrieved literature, PacBio HiFi and Oxford Nanopore offer complementary strengths for structural variant (SV) detection. HiFi reads achieve base-level accuracy above 99.9%, making them well-suited for precise breakpoint resolution [1][3]. Nanopore sequencing, while historically less accurate per base, produces ultra-long reads exceeding 100 kb that are particularly powerful for resolving complex repeat regions and segmental duplications [2]. Recent benchmarking studies suggest that ensemble approaches combining both technologies yield the highest SV recall, especially for inversions and complex rearrangements [4]. For routine clinical SV analysis, HiFi is currently preferred due to its lower error rate, whereas Nanopore is favoured for centromeric and telomeric regions where read length dominates [5].",
  latency_ms: 1342.5,
  sources: [
    {
      pmid: "38291847",
      title:
        "Benchmarking long-read sequencing for structural variant detection in clinical genomics",
      year: "2024",
      authors: "Li H, Feng X, Chu C",
      score: 0.912,
      has_full: true,
    },
    {
      pmid: "37105621",
      title:
        "Ultra-long Oxford Nanopore reads resolve complex structural variants in repeat-rich regions",
      year: "2023",
      authors: "Jain M, Koren S, Miga K",
      score: 0.887,
      has_full: true,
    },
    {
      pmid: "36941834",
      title:
        "HiFi sequencing accuracy enables direct haplotype-resolved assembly of diploid genomes",
      year: "2023",
      authors: "Wenger A, Peluso P, Rowell W",
      score: 0.864,
      has_full: false,
    },
    {
      pmid: "37842109",
      title:
        "Ensemble long-read SV calling improves sensitivity in population-scale genomics",
      year: "2023",
      authors: "Sedlazeck F, Lee H, Darby C",
      score: 0.841,
      has_full: true,
    },
    {
      pmid: "38012745",
      title:
        "Nanopore sequencing of centromeric repeats reveals structural dynamics in cancer",
      year: "2024",
      authors: "Miga K, Koren S, Phillippy A",
      score: 0.798,
      has_full: false,
    },
  ],
};

const EXAMPLE_QUERIES = [
  "What are the main error profiles of Oxford Nanopore reads?",
  "How is long-read sequencing used for cancer genome analysis?",
  "What are the advantages of PacBio HiFi for genome assembly?",
  "How do long reads improve structural variant detection?",
  "What tools are used for long-read RNA sequencing analysis?",
];

// ── Components ────────────────────────────────────────────────────────────────

// Then in the Spinner component show position
function Spinner({ position }) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 20,
        padding: "40px 0",
      }}>
      <div
        style={{
          width: 36,
          height: 36,
          border: "3px solid #1e293b",
          borderTop: "3px solid #3b82f6",
          borderRadius: "50%",
          animation: "spin 0.8s linear infinite",
        }}
      />
      <span style={{ color: "#475569", fontSize: 13, fontFamily: "monospace" }}>
        {position > 0
          ? `${position} quer${position === 1 ? "y" : "ies"} ahead of yours…`
          : "retrieving · embedding · generating"}
      </span>
    </div>
  );
}

function ScoreBar({ score }) {
  const pct = Math.round(score * 100);
  const color = score > 0.85 ? "#4ade80" : score > 0.7 ? "#facc15" : "#94a3b8";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div
        style={{
          flex: 1,
          height: 4,
          background: "#1e293b",
          borderRadius: 2,
          overflow: "hidden",
        }}>
        <div
          style={{
            width: `${pct}%`,
            height: "100%",
            background: color,
            borderRadius: 2,
            transition: "width 0.8s ease",
          }}
        />
      </div>
      <span
        style={{ fontFamily: "monospace", fontSize: 11, color, minWidth: 36 }}>
        {score.toFixed(3)}
      </span>
    </div>
  );
}

function SourceCard({ source, index, visible }) {
  return (
    <div
      style={{
        background: "#0f172a",
        border: "1px solid #1e293b",
        borderRadius: 8,
        padding: "14px 16px",
        opacity: visible ? 1 : 0,
        transform: visible ? "translateY(0)" : "translateY(12px)",
        transition: `opacity 0.4s ease ${index * 0.08}s, transform 0.4s ease ${index * 0.08}s`,
      }}>
      <div
        style={{
          display: "flex",
          alignItems: "flex-start",
          justifyContent: "space-between",
          gap: 12,
          marginBottom: 8,
        }}>
        <span
          style={{
            background: "#1e3a5f",
            color: "#60a5fa",
            fontFamily: "monospace",
            fontSize: 11,
            padding: "2px 8px",
            borderRadius: 4,
            whiteSpace: "nowrap",
            flexShrink: 0,
          }}>
          [{index + 1}] PMID:{source.pmid}
        </span>
        <div style={{ display: "flex", gap: 6, flexShrink: 0 }}>
          <span
            style={{
              fontSize: 11,
              padding: "2px 7px",
              borderRadius: 4,
              background: source.has_full ? "#14532d" : "#1c1917",
              color: source.has_full ? "#4ade80" : "#78716c",
            }}>
            {source.has_full ? "full text" : "abstract"}
          </span>
          <span style={{ fontSize: 11, color: "#64748b", padding: "2px 0" }}>
            {source.year}
          </span>
        </div>
      </div>
      <p
        style={{
          margin: "0 0 8px",
          color: "#e2e8f0",
          fontSize: 13,
          lineHeight: 1.5,
          fontWeight: 500,
        }}>
        {source.title}
      </p>
      <p style={{ margin: "0 0 10px", color: "#64748b", fontSize: 12 }}>
        {source.authors}
      </p>
      <ScoreBar score={source.score} />
    </div>
  );
}

function AnswerBlock({ text, visible }) {
  // Highlight citation markers like [1], [2]
  const parts = text.split(/(\[\d+\])/g);
  return (
    <div
      style={{
        opacity: visible ? 1 : 0,
        transform: visible ? "translateY(0)" : "translateY(8px)",
        transition: "opacity 0.5s ease, transform 0.5s ease",
        lineHeight: 1.75,
        color: "#cbd5e1",
        fontSize: 15,
        textAlign: "left",
      }}>
      <ReactMarkdown
        components={{
          p: ({ children }) => (
            <p
              style={{
                margin: "0 0 12px",
                color: "#cbd5e1",
                lineHeight: 1.75,
              }}>
              {children}
            </p>
          ),
          strong: ({ children }) => (
            <strong style={{ color: "#e2e8f0", fontWeight: 600 }}>
              {children}
            </strong>
          ),
          ul: ({ children }) => (
            <ul style={{ margin: "8px 0 12px", paddingLeft: 20 }}>
              {children}
            </ul>
          ),
          ol: ({ children }) => (
            <ol style={{ margin: "8px 0 12px", paddingLeft: 20 }}>
              {children}
            </ol>
          ),
          li: ({ children }) => (
            <li style={{ color: "#cbd5e1", marginBottom: 4, lineHeight: 1.6 }}>
              {children}
            </li>
          ),
          code: ({ children }) => (
            <code
              style={{
                background: "#1e293b",
                color: "#7dd3fc",
                padding: "1px 6px",
                borderRadius: 4,
                fontFamily: "monospace",
                fontSize: 13,
              }}>
              {children}
            </code>
          ),
        }}>
        {text}
      </ReactMarkdown>
    </div>
  );
}

function Spinner() {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 20,
        padding: "40px 0",
      }}>
      <div
        style={{
          width: 36,
          height: 36,
          border: "3px solid #1e293b",
          borderTop: "3px solid #3b82f6",
          borderRadius: "50%",
          animation: "spin 0.8s linear infinite",
        }}
      />
      <span style={{ color: "#475569", fontSize: 13, fontFamily: "monospace" }}>
        retrieving · embedding · generating
      </span>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────

export default function App() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [demoMode, setDemoMode] = useState(false);
  const [sourcesVisible, setSourcesVisible] = useState(false);
  const [answerVisible, setAnswerVisible] = useState(false);
  const textareaRef = useRef(null);
  const [queuePosition, setQueuePosition] = useState(null);

  // Poll queue status while loading
  useEffect(() => {
    if (!loading) {
      setQueuePosition(null);
      return;
    }
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_URL}/queue`);
        const data = await res.json();
        setQueuePosition(data.queued);
      } catch {}
    }, 3000);
    return () => clearInterval(interval);
  }, [loading]);

  // useEffect(() => {
  //   if (result) {
  //     setTimeout(() => setAnswerVisible(true), 100);
  //     setTimeout(() => setSourcesVisible(true), 400);
  //   } else {
  //     setAnswerVisible(false);
  //     setSourcesVisible(false);
  //   }
  // }, [result]);

  const submit = async (q = query) => {
    if (!q.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);

    if (demoMode) {
      await new Promise((r) => setTimeout(r, 1800));
      setResult({ ...MOCK_RESPONSE, query: q });
      setLoading(false);
      return;
    }

    try {
      const res = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, top_k: 5 }),
      });
      if (!res.ok) throw new Error(`API error ${res.status}`);
      setResult(await res.json());
    } catch (e) {
      setError(
        e.message.includes("fetch")
          ? "Cannot reach API at localhost:8000. Enable Demo Mode to preview, or start the server with: uvicorn app.main:app --reload"
          : e.message,
      );
    } finally {
      setLoading(false);
    }
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#020817",
        fontFamily: "'IBM Plex Sans', 'Helvetica Neue', sans-serif",
        color: "#e2e8f0",
        padding: "0 0 60px",
      }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
        * { box-sizing: border-box; }
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(-6px); } to { opacity: 1; transform: translateY(0); } }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #0f172a; }
        ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
        textarea:focus { outline: none; }
        .example-btn:hover { background: #1e293b !important; color: #93c5fd !important; }
        .submit-btn:hover:not(:disabled) { background: #2563eb !important; }
        .submit-btn:disabled { opacity: 0.5; cursor: not-allowed; }
      `}</style>

      {/* Header */}
      <div
        style={{
          borderBottom: "1px solid #0f172a",
          padding: "24px 40px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          background: "#020817",
          position: "sticky",
          top: 0,
          zIndex: 10,
          backdropFilter: "blur(8px)",
        }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <div
            style={{
              width: 32,
              height: 32,
              background: "linear-gradient(135deg, #1d4ed8, #0891b2)",
              borderRadius: 8,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 16,
            }}>
            🧬
          </div>
          <div>
            <div
              style={{
                fontWeight: 600,
                fontSize: 15,
                letterSpacing: "-0.01em",
              }}>
              LongRead<span style={{ color: "#3b82f6" }}>RAG</span>
            </div>
            <div
              style={{
                fontSize: 11,
                color: "#475569",
                fontFamily: "monospace",
              }}>
              ~3,000 papers · PubMed/PMC
            </div>
          </div>
        </div>

        <label
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            cursor: "pointer",
            userSelect: "none",
          }}>
          <span
            style={{ fontSize: 12, color: "#475569", fontFamily: "monospace" }}>
            demo mode
          </span>
          <div
            onClick={() => setDemoMode((d) => !d)}
            style={{
              width: 36,
              height: 20,
              background: demoMode ? "#3b82f6" : "#1e293b",
              borderRadius: 10,
              position: "relative",
              transition: "background 0.2s",
              cursor: "pointer",
            }}>
            <div
              style={{
                position: "absolute",
                top: 2,
                left: demoMode ? 18 : 2,
                width: 16,
                height: 16,
                background: "white",
                borderRadius: 8,
                transition: "left 0.2s",
              }}
            />
          </div>
        </label>
      </div>

      <div style={{ maxWidth: 760, margin: "0 auto", padding: "40px 24px 0" }}>
        {/* Query box */}
        <div
          style={{
            background: "#0f172a",
            border: "1px solid #1e293b",
            borderRadius: 12,
            padding: 16,
            marginBottom: 16,
            animation: "fadeIn 0.4s ease",
          }}>
          <textarea
            ref={textareaRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKey}
            placeholder="Ask a question about long-read sequencing literature…"
            rows={3}
            style={{
              width: "100%",
              background: "transparent",
              border: "none",
              color: "#e2e8f0",
              fontSize: 15,
              resize: "none",
              fontFamily: "'IBM Plex Sans', sans-serif",
              lineHeight: 1.6,
              marginBottom: 12,
            }}
          />
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}>
            <span
              style={{
                fontSize: 11,
                color: "#334155",
                fontFamily: "monospace",
              }}>
              ↵ enter to submit · shift+↵ newline
            </span>
            <button
              className="submit-btn"
              onClick={() => submit()}
              disabled={loading || !query.trim()}
              style={{
                background: "#1d4ed8",
                color: "white",
                border: "none",
                borderRadius: 7,
                padding: "8px 20px",
                fontSize: 13,
                fontWeight: 500,
                cursor: "pointer",
                transition: "background 0.15s",
                fontFamily: "'IBM Plex Sans', sans-serif",
              }}>
              {loading ? "Searching…" : "Ask"}
            </button>
          </div>
        </div>

        {/* Example queries */}
        {!result && !loading && (
          <div style={{ marginBottom: 32 }}>
            <div
              style={{
                fontSize: 11,
                color: "#334155",
                fontFamily: "monospace",
                marginBottom: 10,
              }}>
              EXAMPLE QUERIES
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {EXAMPLE_QUERIES.map((q, i) => (
                <button
                  key={i}
                  className="example-btn"
                  onClick={() => {
                    setQuery(q);
                    submit(q);
                  }}
                  style={{
                    background: "#0a0f1a",
                    border: "1px solid #1e293b",
                    borderRadius: 7,
                    padding: "10px 14px",
                    color: "#64748b",
                    fontSize: 13,
                    textAlign: "left",
                    cursor: "pointer",
                    transition: "all 0.15s",
                    fontFamily: "'IBM Plex Sans', sans-serif",
                  }}>
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Loading */}
        {loading && <Spinner />}

        {/* Error */}
        {error && (
          <div
            style={{
              background: "#1c0a0a",
              border: "1px solid #7f1d1d",
              borderRadius: 8,
              padding: 16,
              color: "#fca5a5",
              fontSize: 13,
              fontFamily: "monospace",
              lineHeight: 1.6,
            }}>
            {error}
          </div>
        )}

        {/* Results */}
        {result && (
          <div>
            {/* Query echo */}
            <div
              style={{
                fontFamily: "monospace",
                fontSize: 12,
                color: "#475569",
                marginBottom: 20,
                padding: "8px 12px",
                background: "#0a0f1a",
                borderRadius: 6,
                borderLeft: "3px solid #1d4ed8",
              }}>
              query: {result.query}
              <span style={{ float: "right", color: "#334155" }}>
                {result.latency_ms}ms
              </span>
            </div>

            {/* Answer */}
            <div
              style={{
                background: "#0f172a",
                border: "1px solid #1e293b",
                borderRadius: 10,
                padding: "20px 22px",
                marginBottom: 24,
              }}>
              <div
                style={{
                  fontSize: 11,
                  color: "#3b82f6",
                  fontFamily: "monospace",
                  marginBottom: 14,
                  letterSpacing: "0.08em",
                }}>
                ANSWER
              </div>
              <AnswerBlock text={result.answer} visible={answerVisible} />
            </div>

            {/* Sources */}
            <div>
              <div
                style={{
                  fontSize: 11,
                  color: "#475569",
                  fontFamily: "monospace",
                  marginBottom: 12,
                  letterSpacing: "0.08em",
                }}>
                SOURCES — {result.sources.length} retrieved
              </div>
              <div
                style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {result.sources.map((s, i) => (
                  <SourceCard
                    key={s.pmid}
                    source={s}
                    index={i}
                    visible={sourcesVisible}
                  />
                ))}
              </div>
            </div>

            {/* New query */}
            <button
              onClick={() => {
                setResult(null);
                setQuery("");
                setTimeout(() => textareaRef.current?.focus(), 100);
              }}
              style={{
                marginTop: 28,
                background: "transparent",
                border: "1px solid #1e293b",
                borderRadius: 7,
                padding: "9px 18px",
                color: "#475569",
                fontSize: 13,
                cursor: "pointer",
                fontFamily: "'IBM Plex Sans', sans-serif",
                transition: "all 0.15s",
              }}>
              ← New query
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
