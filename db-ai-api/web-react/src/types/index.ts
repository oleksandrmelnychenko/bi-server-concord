// Re-export structured response types
export * from './responses';

// Import StructuredResponse for use in Message type
import type { StructuredResponse } from './responses';

// API Status
export interface ApiStatus {
  online: boolean;
  documents: number | null;
}

// Chat Message
export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  structuredContent?: StructuredResponse;
  reaction?: 'like' | 'dislike';
  sourceQuery?: string;
}

// Ollama SQL Query Result
export interface OllamaQueryResult {
  question?: string;
  mode?: 'sql' | 'rag';
  success?: boolean;
  answer?: string;
  sql?: string;
  explanation?: string;
  execution?: {
    success: boolean;
    rows?: Record<string, unknown>[];
    columns?: string[];
    row_count?: number;
    error?: string;
  };
  error?: string;
}

// Supported query routing
export type QueryType = 'search';
