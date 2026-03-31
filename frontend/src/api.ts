// src/api.ts
import type { QueryResponse, HealthResponse, PapersResponse } from './types';

const BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

export async function fetchHealth(): Promise<HealthResponse> {
  const r = await fetch(`${BASE}/health`);
  if (!r.ok) throw new Error('API offline');
  return r.json();
}

export async function fetchPapers(): Promise<PapersResponse> {
  const r = await fetch(`${BASE}/papers`);
  if (!r.ok) throw new Error('Could not load papers');
  return r.json();
}

export async function fetchQuery(
  question: string,
  chatHistory: string[] = [],
  paperFilter: string[] = [],
  topK = 6,
): Promise<QueryResponse> {
  const r = await fetch(`${BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, chat_history: chatHistory, paper_filter: paperFilter, top_k: topK }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail ?? 'Query failed');
  }
  return r.json();
}

export async function triggerIngest(): Promise<{ job_id: string; status: string; message: string }> {
  const r = await fetch(`${BASE}/ingest`, { method: 'POST' });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail ?? 'Ingest failed');
  }
  return r.json();
}

export async function fetchIngestStatus(jobId: string) {
  const r = await fetch(`${BASE}/ingest/status/${jobId}`);
  if (!r.ok) throw new Error('Failed to fetch job status');
  return r.json();
}
