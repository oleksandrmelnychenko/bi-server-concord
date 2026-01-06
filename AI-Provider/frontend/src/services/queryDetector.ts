import type { QueryType } from '../types';

// Extract number from query (e.g., "top 5" -> 5)
export function extractNumber(text: string): number | null {
  const match = text.match(/\d+/);
  return match ? parseInt(match[0], 10) : null;
}

// Extract product search keyword - kept for potential future use
export function extractProductKeyword(query: string): string | null {
  const match = query.match(/товар(?:ів|и)?\s+(?:з|із)\s+(\w+)/i);
  return match?.[1] || null;
}

// All queries go to Ollama SQL agent
export function detectQueryType(_query: string): QueryType {
  return 'search';
}

// Detect if query should show a region map
export function isRegionMapQuery(query: string): boolean {
  const q = query.toLowerCase();

  // Patterns that suggest region-based visualization
  const regionPatterns = [
    /по\s+регіонах/i,
    /за\s+регіонами/i,
    /регіон/i,
    /област/i,
    /карта/i,
    /map/i,
    /by\s+region/i,
    /per\s+region/i,
    /geography/i,
  ];

  return regionPatterns.some((pattern) => pattern.test(q));
}

// Detect if query mentions specific region names
export function extractRegionFromQuery(query: string): string | null {
  const q = query.toLowerCase();

  const regionMap: Record<string, string> = {
    'київ': 'KI',
    'kyiv': 'KI',
    'львів': 'LV',
    'lviv': 'LV',
    'одеса': 'OD',
    'odesa': 'OD',
    'харк': 'XV',
    'khark': 'XV',
    'дніпро': 'DP',
    'dnipro': 'DP',
    'хмельницьк': 'XM',
    'khmelnytsk': 'XM',
  };

  for (const [name, code] of Object.entries(regionMap)) {
    if (q.includes(name)) {
      return code;
    }
  }

  return null;
}

// Check if result data contains region information
export function hasRegionData(rows: Record<string, unknown>[]): boolean {
  if (!rows || rows.length === 0) return false;

  const firstRow = rows[0];
  const keys = Object.keys(firstRow).map((k) => k.toLowerCase());

  return keys.some((k) =>
    k.includes('region') ||
    k.includes('регіон') ||
    k.includes('oblast') ||
    k === 'regioncode' ||
    k === 'region_code' ||
    k === 'regionid' ||
    k === 'region_id'
  );
}
