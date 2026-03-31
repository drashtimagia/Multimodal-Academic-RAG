// src/App.tsx — Academic Research RAG Chatbot
import { useCallback, useEffect, useRef, useState } from 'react';
import './index.css';

import { fetchHealth, fetchPapers, fetchQuery, triggerIngest, fetchIngestStatus } from './api';
import type { ChatMessage, Paper, Source } from './types';

// ── Sub-components ────────────────────────────────────────────────────────────

function StatusBadge() {
  const [online, setOnline] = useState<boolean | null>(null);
  const [vectors, setVectors] = useState(0);

  useEffect(() => {
    const check = async () => {
      try {
        const h = await fetchHealth();
        setOnline(true);
        setVectors(h.vectors);
      } catch { setOnline(false); }
    };
    check();
    const id = setInterval(check, 15000);
    return () => clearInterval(id);
  }, []);

  if (online === null) return null;
  return (
    <span className={`status-badge ${online ? 'online' : 'offline'}`}>
      <span className="status-dot" />
      {online ? `${vectors.toLocaleString()} vectors` : 'API Offline'}
    </span>
  );
}

function TypeBadge({ type }: { type: string }) {
  const cls = ['text', 'table', 'image'].includes(type) ? type : 'text';
  const icons: Record<string, string> = { text: '¶', table: '⊞', image: '⊡' };
  return <span className={`type-badge ${cls}`}>{icons[cls] ?? '¶'} {type.toUpperCase()}</span>;
}

function SourceCard({ source }: { source: Source }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div
      className={`source-card ${expanded ? 'expanded' : ''}`}
      onClick={() => setExpanded(e => !e)}
      role="button" tabIndex={0}
      onKeyDown={e => e.key === 'Enter' && setExpanded(v => !v)}
    >
      <div className="source-head">
        <TypeBadge type={source.element_type} />
        <span className="source-name">{source.source} · p.{source.page}</span>
      </div>
      <p className="source-excerpt">
        {expanded ? source.excerpt : `${source.excerpt.slice(0, 150)}${source.excerpt.length > 150 ? '…' : ''}`}
      </p>
    </div>
  );
}

function AssistantMessage({ msg, onSelectSuggestion }: { msg: ChatMessage, onSelectSuggestion: (q: string) => void }) {
  const [showSources, setShowSources] = useState(false);
  // Typewriter effect
  const [displayed, setDisplayed] = useState('');
  const full = msg.content;

  useEffect(() => {
    if (msg.loading) return;
    setDisplayed('');
    let i = 0;
    const id = setInterval(() => {
      i++;
      setDisplayed(full.slice(0, i));
      if (i >= full.length) clearInterval(id);
    }, 6);
    return () => clearInterval(id);
  }, [full, msg.loading]);

  return (
    <div className="message assistant">
      <div className="msg-avatar">🎓</div>
      <div className="msg-body">
        <div className="msg-bubble">
          {msg.loading
            ? <div className="thinking"><span/><span/><span/></div>
            : displayed
          }
        </div>

        {!msg.loading && (
          <div className="msg-meta">
            <span>{msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
            {msg.sources && msg.sources.length > 0 && (
              <div className="sources-wrap">
                <button className="sources-toggle" onClick={() => setShowSources(s => !s)}>
                  {showSources ? '▾' : '▸'} {msg.sources.length} source{msg.sources.length !== 1 ? 's' : ''}
                </button>
                {showSources && (
                  <div className="sources-list">
                    {msg.sources.map((src, i) => (
                      <SourceCard key={src.chunk_id || i} source={src} />
                    ))}
                  </div>
                )}
              </div>
            )}
            {msg.suggested_questions && msg.suggested_questions.length > 0 && (
              <div className="suggested-questions" style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', marginTop: '0.8rem' }}>
                {msg.suggested_questions.map((q, i) => (
                  <button key={i} className="btn btn-ghost" style={{ textAlign: 'left', padding: '0.4rem 0.6rem', fontSize: '0.85rem', color: 'var(--accent)', background: 'rgba(99, 102, 241, 0.1)' }} onClick={() => onSelectSuggestion(q)}>
                    {q}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function UserMessage({ msg }: { msg: ChatMessage }) {
  return (
    <div className="message user">
      <div className="msg-avatar">U</div>
      <div className="msg-body">
        <div className="msg-bubble">{msg.content}</div>
        <div className="msg-meta">
          {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </div>
      </div>
    </div>
  );
}

// ── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput]       = useState('');
  const [loading, setLoading]   = useState(false);

  // Papers
  const [papers, setPapers]           = useState<Paper[]>([]);
  const [selectedPapers, setSelected] = useState<Set<string>>(new Set());

  // Ingest state
  const [dragOver, setDragOver]   = useState(false);
  const [ingestMsg, setIngestMsg] = useState<string | null>(null);
  const [jobId, setJobId]         = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const bottomRef = useRef<HTMLDivElement>(null);

  // Load papers list
  const loadPapers = useCallback(async () => {
    try {
      const res = await fetchPapers();
      setPapers(res.papers);
    } catch { /* API offline */ }
  }, []);

  useEffect(() => { loadPapers(); }, [loadPapers]);

  // Auto-scroll
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Poll ingest job status
  useEffect(() => {
    if (!jobId) return;
    const id = setInterval(async () => {
      try {
        const s = await fetchIngestStatus(jobId);
        setIngestMsg(`[${s.steps_done}/${s.total_steps}] ${s.status}`);
        if (s.status === 'done' || s.status === 'failed') {
          clearInterval(id);
          setJobId(null);
          if (s.status === 'done') { await loadPapers(); setIngestMsg('✅ Indexed'); }
          else setIngestMsg(`❌ ${s.log.split('\n').pop()}`);
        }
      } catch { clearInterval(id); }
    }, 2000);
    return () => clearInterval(id);
  }, [jobId, loadPapers]);

  const togglePaper = (src: string) => {
    setSelected(prev => {
      const next = new Set(prev);
      next.has(src) ? next.delete(src) : next.add(src);
      return next;
    });
  };

  const sendMessage = async (overrideQ?: string | React.MouseEvent) => {
    const overrideText = typeof overrideQ === 'string' ? overrideQ : undefined;
    const q = (overrideText || input).trim();
    if (!q || loading) return;
    if (!overrideText) setInput('');
    setLoading(true);

    const history = messages.filter(m => m.role === 'user').map(m => m.content);

    const userMsg: ChatMessage = {
      id: crypto.randomUUID(), role: 'user', content: q, timestamp: new Date(),
    };
    const loadingMsg: ChatMessage = {
      id: crypto.randomUUID(), role: 'assistant', content: '', timestamp: new Date(), loading: true,
    };
    setMessages(prev => [...prev, userMsg, loadingMsg]);

    try {
      const res = await fetchQuery(q, history, Array.from(selectedPapers));
      setMessages(prev => prev.map(m =>
        m.id === loadingMsg.id
          ? { ...m, content: res.answer, sources: res.sources, suggested_questions: res.suggested_questions, model: res.model, loading: false }
          : m
      ));
    } catch (err: unknown) {
      const errText = err instanceof Error ? err.message : 'Request failed';
      setMessages(prev => prev.map(m =>
        m.id === loadingMsg.id
          ? { ...m, content: `⚠️ ${errText}`, loading: false }
          : m
      ));
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  };

  const handleIngest = async () => {
    setIngestMsg('Queueing…');
    try {
      const res = await triggerIngest();
      setJobId(res.job_id);
      setIngestMsg('▶ Running…');
    } catch (err: unknown) {
      setIngestMsg(`Error: ${err instanceof Error ? err.message : 'failed'}`);
    }
  };

  const EXAMPLES = [
    'What evaluation metrics are used in this paper?',
    'Summarize the methodology and experiments',
    'What tables compare model performance?',
    'What figures or diagrams does this paper include?',
    'What are the main findings and conclusions?',
  ];

  return (
    <div className="layout">
      {/* Header */}
      <header className="header">
        <div className="logo">
          <div className="logo-icon">📚</div>
          Academic Research RAG
        </div>
        <StatusBadge />
      </header>

      {/* Sidebar */}
      <aside className="sidebar">
        {/* Papers */}
        <div className="sidebar-section">
          <div className="sidebar-label">
            Research Papers {papers.length > 0 && `(${papers.length})`}
          </div>
          {papers.length === 0 && (
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              No papers indexed yet.
            </p>
          )}
          {papers.map(p => (
            <div
              key={p.source}
              className={`paper-item ${selectedPapers.has(p.source) ? 'active' : ''}`}
              onClick={() => togglePaper(p.source)}
              role="checkbox" aria-checked={selectedPapers.has(p.source)} tabIndex={0}
              onKeyDown={e => e.key === 'Enter' && togglePaper(p.source)}
            >
              <span className="paper-check">
                {selectedPapers.has(p.source) ? '☑' : '☐'}
              </span>
              <div>
                <div className="paper-name">{p.source}</div>
                <div className="paper-counts">
                  ¶{p.text} ⊞{p.table} ⊡{p.image} total:{p.total}
                </div>
              </div>
            </div>
          ))}
          {papers.length > 0 && selectedPapers.size > 0 && (
            <button className="btn btn-ghost" onClick={() => setSelected(new Set())} style={{ marginTop: '0.5rem' }}>
              Clear filter
            </button>
          )}
        </div>

        {/* Upload */}
        <div className="sidebar-section">
          <div className="sidebar-label">Add Papers</div>
          <div
            className={`dropzone ${dragOver ? 'drag-over' : ''}`}
            onDragOver={e => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={e => { e.preventDefault(); setDragOver(false); }}
            onClick={() => fileRef.current?.click()}
            role="button" tabIndex={0}
          >
            <div className="dropzone-icon">📄</div>
            Drop PDFs in <code style={{ fontSize: '0.72rem' }}>data/raw_pdfs/</code>
            <input ref={fileRef} type="file" accept=".pdf" multiple style={{ display: 'none' }} />
          </div>
          <button className="btn btn-primary" onClick={handleIngest} disabled={!!jobId}>
            {jobId ? '⏳ Indexing…' : '🔄 Re-index All Papers'}
          </button>
          {ingestMsg && (
            <div className={`alert ${ingestMsg.startsWith('✅') ? 'alert-success' : ingestMsg.startsWith('❌') ? 'alert-error' : 'alert-success'}`}>
              {ingestMsg}
            </div>
          )}
        </div>

        {/* Filter hint */}
        {selectedPapers.size > 0 && (
          <div className="sidebar-section">
            <div className="sidebar-label">Active Filter</div>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>
              Searching <strong style={{ color: 'var(--accent)' }}>{selectedPapers.size}</strong> paper{selectedPapers.size !== 1 ? 's' : ''}
            </p>
          </div>
        )}
      </aside>

      {/* Chat panel */}
      <main className="chat-panel">
        <div className="messages-area">
          {messages.length === 0 ? (
            <div className="welcome">
              <div className="welcome-icon">🔬</div>
              <h2>Research Assistant</h2>
              <p>
                Ask questions across your indexed research papers.
                I can reason over <strong>text</strong>, <strong>tables</strong>, and <strong>figures</strong>.
                {papers.length > 0 && ` ${papers.length} paper${papers.length !== 1 ? 's' : ''} available.`}
              </p>
              <div className="example-chips">
                {EXAMPLES.map(ex => (
                  <span key={ex} className="chip" onClick={() => { setInput(ex); }}>
                    {ex}
                  </span>
                ))}
              </div>
            </div>
          ) : (
            messages.map(msg =>
              msg.role === 'user'
                ? <UserMessage key={msg.id} msg={msg} />
                : <AssistantMessage key={msg.id} msg={msg} onSelectSuggestion={sendMessage} />
            )
          )}
          <div ref={bottomRef} />
        </div>

        {/* Input dock */}
        <div className="input-dock">
          <div className="input-row">
            <textarea
              className="chat-textarea"
              placeholder={
                selectedPapers.size > 0
                  ? `Ask about ${[...selectedPapers].join(', ')}…`
                  : 'Ask anything about your research papers…'
              }
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={1}
              disabled={loading}
            />
            <button className="send-btn" onClick={sendMessage} disabled={loading || !input.trim()}>
              {loading ? '⏳' : '↑'}
            </button>
          </div>
          <p className="input-hint">
            Enter to send · Shift+Enter for new line
            {selectedPapers.size > 0 && ` · Filtering: ${selectedPapers.size} paper${selectedPapers.size !== 1 ? 's' : ''}`}
          </p>
        </div>
      </main>
    </div>
  );
}
