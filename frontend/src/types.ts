// src/types.ts
export interface Source {
  chunk_id: string;
  source: string;
  page: number;
  element_type: 'text' | 'table' | 'image' | string;
  excerpt: string;
}

export interface QueryResponse {
  question: string;
  answer: string;
  sources: Source[];
  suggested_questions: string[];
  model: string;
}

export interface HealthResponse {
  status: string;
  vectors: number;
  version: string;
  scheduler: boolean;
}

export interface Paper {
  source: string;
  total: number;
  text: number;
  table: number;
  image: number;
}

export interface PapersResponse {
  count: number;
  papers: Paper[];
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  suggested_questions?: string[];
  model?: string;
  timestamp: Date;
  loading?: boolean;
}
