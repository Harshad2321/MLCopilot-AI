// MLCopilot AI — React Dashboard  (frontend/src/App.js)
// ======================================================
// Fast, real-time dashboard for monitoring ML training runs.
// Polls the FastAPI backend every 4 seconds for live updates.

import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine,
} from 'recharts';

// ── API base URL ──────────────────────────────────────────────────────────────
// In production, set REACT_APP_API_URL to your EC2 public URL
const API_URL = process.env.REACT_APP_API_URL || '';

// ── Styles (inline — no CSS file needed) ─────────────────────────────────────
const S = {
  app:        { minHeight: '100vh', background: '#0f1117', color: '#e2e8f0', padding: '0 0 40px' },
  header:     { background: 'linear-gradient(135deg,#667eea,#764ba2)', padding: '20px 32px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' },
  title:      { fontSize: 28, fontWeight: 700, color: '#fff' },
  badge:      { background: 'rgba(255,255,255,0.2)', color: '#fff', padding: '4px 12px', borderRadius: 20, fontSize: 13 },
  body:       { maxWidth: 1400, margin: '0 auto', padding: '24px 24px 0' },
  row:        { display: 'flex', gap: 16, marginBottom: 16, flexWrap: 'wrap' },
  card:       { background: '#1a1d2e', borderRadius: 12, padding: '20px 24px', flex: 1, minWidth: 200 },
  kpiVal:     { fontSize: 32, fontWeight: 700, color: '#667eea', margin: '8px 0 4px' },
  kpiLabel:   { fontSize: 13, color: '#718096' },
  chartCard:  { background: '#1a1d2e', borderRadius: 12, padding: '20px 24px', marginBottom: 16 },
  chartTitle: { fontSize: 16, fontWeight: 600, marginBottom: 16, color: '#a0aec0' },
  select:     { background: '#2d3748', color: '#e2e8f0', border: '1px solid #4a5568', borderRadius: 8, padding: '8px 14px', fontSize: 14, cursor: 'pointer' },
  btn:        { background: 'linear-gradient(135deg,#667eea,#764ba2)', color: '#fff', border: 'none', borderRadius: 8, padding: '10px 20px', fontSize: 14, fontWeight: 600, cursor: 'pointer' },
  issueCard:  (sev) => ({
    borderLeft: `4px solid ${SEV_COLOR[sev] || '#718096'}`,
    background: '#2d3748', borderRadius: 8,
    padding: '14px 18px', marginBottom: 12,
  }),
  issueName:  { fontSize: 16, fontWeight: 700, marginBottom: 6 },
  issueReason:{ fontSize: 13, color: '#a0aec0', marginBottom: 10 },
  fix:        { fontSize: 13, color: '#68d391', marginBottom: 4 },
  llmBox:     { background: '#1a1d2e', borderRadius: 6, padding: '10px 14px', marginTop: 10, fontSize: 13, color: '#bee3f8', lineStyle: 'italic' },
  tabs:       { display: 'flex', gap: 8, marginBottom: 20 },
  tab:        (active) => ({
    padding: '8px 20px', borderRadius: 8, cursor: 'pointer', fontSize: 14, fontWeight: 600,
    background: active ? 'linear-gradient(135deg,#667eea,#764ba2)' : '#2d3748',
    color: '#fff', border: 'none',
  }),
  empty:      { textAlign: 'center', color: '#4a5568', padding: '60px 0', fontSize: 16 },
  dot:        (ok) => ({ width: 10, height: 10, borderRadius: '50%', background: ok ? '#68d391' : '#fc8181', display: 'inline-block', marginRight: 6 }),
};

const SEV_COLOR = { critical: '#fc8181', high: '#f6ad55', medium: '#f6e05e', low: '#68d391' };
const SEV_ICON  = { critical: '🔴', high: '🟠', medium: '🟡', low: '🟢' };

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [runs,       setRuns]       = useState([]);
  const [selectedRun,setSelectedRun]= useState('');
  const [metrics,    setMetrics]    = useState([]);
  const [analysis,   setAnalysis]   = useState(null);
  const [tab,        setTab]        = useState('metrics');
  const [healthy,    setHealthy]    = useState(false);
  const [loading,    setLoading]    = useState(false);
  const [autoRefresh,setAutoRefresh]= useState(true);

  // ── Health check ────────────────────────────────────────────────────────────
  useEffect(() => {
    axios.get(`${API_URL}/health`)
      .then(() => setHealthy(true))
      .catch(() => setHealthy(false));
  }, []);

  // ── Fetch all run IDs from the metrics endpoint ───────────────────────────
  const fetchRuns = useCallback(async () => {
    try {
      const res = await axios.get(`${API_URL}/metrics`);
      const rows = res.data.metrics || [];
      const ids  = [...new Set(rows.map(r => r.run_id))].sort();
      setRuns(ids);
      if (!selectedRun && ids.length > 0) setSelectedRun(ids[ids.length - 1]);
    } catch { /* server not ready yet */ }
  }, [selectedRun]);

  // ── Fetch metrics for selected run ───────────────────────────────────────
  const fetchMetrics = useCallback(async () => {
    if (!selectedRun) return;
    try {
      const res = await axios.get(`${API_URL}/metrics`, { params: { run_id: selectedRun } });
      setMetrics(res.data.metrics || []);
    } catch {}
  }, [selectedRun]);

  // ── Auto-refresh every 4 seconds ─────────────────────────────────────────
  useEffect(() => {
    fetchRuns();
    fetchMetrics();
    if (!autoRefresh) return;
    const id = setInterval(() => { fetchRuns(); fetchMetrics(); }, 4000);
    return () => clearInterval(id);
  }, [fetchRuns, fetchMetrics, autoRefresh]);

  // ── Run analysis on demand ────────────────────────────────────────────────
  const runAnalysis = async () => {
    if (!selectedRun) return;
    setLoading(true);
    try {
      const res = await axios.get(`${API_URL}/analysis`, { params: { run_id: selectedRun } });
      setAnalysis(res.data);
      setTab('analysis');
    } catch (e) {
      alert('Analysis failed: ' + (e.response?.data?.detail || e.message));
    } finally {
      setLoading(false);
    }
  };

  // ── KPIs ─────────────────────────────────────────────────────────────────
  const latest = metrics[metrics.length - 1] || {};

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div style={S.app}>

      {/* Header */}
      <div style={S.header}>
        <div>
          <div style={S.title}>🤖 MLCopilot AI</div>
          <div style={{ fontSize: 13, color: 'rgba(255,255,255,0.7)', marginTop: 4 }}>
            Real-time ML Training Monitor &amp; Debugger
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span><span style={S.dot(healthy)} />{healthy ? 'Backend online' : 'Backend offline'}</span>
          <span style={S.badge}>AI for Bharat Hackathon</span>
        </div>
      </div>

      <div style={S.body}>

        {/* Controls row */}
        <div style={{ ...S.row, alignItems: 'center', marginBottom: 20, marginTop: 20 }}>
          <div style={{ flex: 1 }}>
            <label style={{ fontSize: 13, color: '#a0aec0', marginRight: 10 }}>Training Run:</label>
            <select
              style={S.select}
              value={selectedRun}
              onChange={e => { setSelectedRun(e.target.value); setAnalysis(null); }}
            >
              {runs.length === 0 && <option value="">— No runs yet —</option>}
              {runs.map(r => <option key={r} value={r}>{r}</option>)}
            </select>
          </div>
          <label style={{ fontSize: 13, color: '#a0aec0', cursor: 'pointer' }}>
            <input
              type="checkbox" checked={autoRefresh}
              onChange={e => setAutoRefresh(e.target.checked)}
              style={{ marginRight: 6 }}
            />
            Auto-refresh (4s)
          </label>
          <button style={S.btn} onClick={runAnalysis} disabled={!selectedRun || loading}>
            {loading ? '⏳ Analysing…' : '▶ Run Analysis'}
          </button>
        </div>

        {/* KPI cards */}
        <div style={S.row}>
          <div style={S.card}>
            <div style={S.kpiLabel}>Epochs Logged</div>
            <div style={S.kpiVal}>{metrics.length}</div>
          </div>
          <div style={S.card}>
            <div style={S.kpiLabel}>Latest Train Loss</div>
            <div style={S.kpiVal}>{latest.train_loss != null ? latest.train_loss.toFixed(4) : '—'}</div>
          </div>
          <div style={S.card}>
            <div style={S.kpiLabel}>Latest Val Loss</div>
            <div style={S.kpiVal}>{latest.val_loss != null ? latest.val_loss.toFixed(4) : '—'}</div>
          </div>
          <div style={S.card}>
            <div style={S.kpiLabel}>Latest Accuracy</div>
            <div style={S.kpiVal}>{latest.accuracy != null ? (latest.accuracy * 100).toFixed(1) + '%' : '—'}</div>
          </div>
          <div style={S.card}>
            <div style={S.kpiLabel}>Issues Found</div>
            <div style={{ ...S.kpiVal, color: analysis?.total_issues > 0 ? '#fc8181' : '#68d391' }}>
              {analysis ? analysis.total_issues : '—'}
            </div>
          </div>
        </div>

        {/* Tabs */}
        <div style={S.tabs}>
          <button style={S.tab(tab === 'metrics')}  onClick={() => setTab('metrics')}>📈 Live Metrics</button>
          <button style={S.tab(tab === 'analysis')} onClick={() => setTab('analysis')}>🔍 Analysis</button>
        </div>

        {/* Metrics tab */}
        {tab === 'metrics' && (
          metrics.length === 0
            ? <div style={S.empty}>No metrics yet. Run a training script to get started.</div>
            : <>
                <ChartCard title="Loss Curves">
                  <LineChart data={metrics}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
                    <XAxis dataKey="epoch" stroke="#718096" />
                    <YAxis stroke="#718096" />
                    <Tooltip contentStyle={{ background: '#2d3748', border: 'none' }} />
                    <Legend />
                    <Line type="monotone" dataKey="train_loss" stroke="#667eea" dot={false} name="Train Loss" strokeWidth={2} />
                    <Line type="monotone" dataKey="val_loss"   stroke="#ed8936" dot={false} name="Val Loss"   strokeWidth={2} strokeDasharray="5 5" />
                  </LineChart>
                </ChartCard>

                <div style={S.row}>
                  <div style={{ flex: 1, ...S.chartCard }}>
                    <div style={S.chartTitle}>Accuracy</div>
                    <ResponsiveContainer width="100%" height={200}>
                      <LineChart data={metrics}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
                        <XAxis dataKey="epoch" stroke="#718096" />
                        <YAxis domain={[0, 1]} stroke="#718096" />
                        <Tooltip contentStyle={{ background: '#2d3748', border: 'none' }} />
                        <Line type="monotone" dataKey="accuracy" stroke="#48bb78" dot={false} strokeWidth={2} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>

                  <div style={{ flex: 1, ...S.chartCard }}>
                    <div style={S.chartTitle}>Gradient Norm</div>
                    <ResponsiveContainer width="100%" height={200}>
                      <LineChart data={metrics}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
                        <XAxis dataKey="epoch" stroke="#718096" />
                        <YAxis stroke="#718096" />
                        <Tooltip contentStyle={{ background: '#2d3748', border: 'none' }} />
                        <ReferenceLine y={10} stroke="orange" strokeDasharray="4 4" label={{ value: 'Exploding threshold', fill: 'orange', fontSize: 11 }} />
                        <Line type="monotone" dataKey="gradient_norm" stroke="#fc8181" dot={false} strokeWidth={2} name="Grad Norm" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </>
        )}

        {/* Analysis tab */}
        {tab === 'analysis' && (
          !analysis
            ? <div style={S.empty}>Click <strong>▶ Run Analysis</strong> to detect issues in this run.</div>
            : analysis.total_issues === 0
              ? <div style={{ ...S.empty, color: '#68d391' }}>✅ No issues detected — training looks healthy!</div>
              : analysis.results.map((issue, i) => (
                  <div key={i} style={S.issueCard(issue.severity)}>
                    <div style={{ ...S.issueName, color: SEV_COLOR[issue.severity] }}>
                      {SEV_ICON[issue.severity]} {issue.issue}
                      <span style={{ fontSize: 12, fontWeight: 400, marginLeft: 10, opacity: 0.8 }}>
                        {issue.severity?.toUpperCase()} · Epoch {issue.epoch}
                      </span>
                    </div>
                    <div style={S.issueReason}>{issue.reason}</div>

                    {issue.suggestions?.length > 0 && (
                      <div>
                        <div style={{ fontSize: 12, color: '#718096', marginBottom: 6 }}>🔧 Suggestions:</div>
                        {issue.suggestions.map((s, j) => (
                          <div key={j} style={S.fix}>✓ {s}</div>
                        ))}
                      </div>
                    )}

                    {issue.llm_explanation && (
                      <div style={S.llmBox}>
                        💡 <em>{issue.llm_explanation}</em>
                      </div>
                    )}
                  </div>
                ))
        )}

      </div>
    </div>
  );
}

// ── Chart wrapper ─────────────────────────────────────────────────────────────
function ChartCard({ title, children }) {
  return (
    <div style={{ background: '#1a1d2e', borderRadius: 12, padding: '20px 24px', marginBottom: 16 }}>
      <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 16, color: '#a0aec0' }}>{title}</div>
      <ResponsiveContainer width="100%" height={260}>
        {children}
      </ResponsiveContainer>
    </div>
  );
}
