import type { QueryType } from '../types';

// Extract product search keyword from query
export function extractProductKeyword(query: string): string | null {
  const patterns = [
    /товар[иі]?\s+(?:із|з|with|містять?|де є|які мають)\s+[«"]?(\w+)[»"]?/i,
    /(?:із|з)\s+(\w+)\s+товар/i,
    /(?:виведи|покажи|знайди)\s+.*товар.*\s+(\w+)/i,
    /товар.*\s+(\w+)$/i,
    /продукт[иі]?\s+(?:із|з|with)\s+[«"]?(\w+)[»"]?/i,
  ];

  for (const pattern of patterns) {
    const match = query.match(pattern);
    if (match && match[1] && match[1].length >= 2) {
      return match[1];
    }
  }
  return null;
}

// Extract number from query
export function extractNumber(text: string): number | null {
  const match = text.match(/\d+/);
  return match ? parseInt(match[0], 10) : null;
}

// Detect query type
export function detectQueryType(query: string): QueryType {
  const lowerQuery = query.toLowerCase();

  // Product keyword search (e.g., "товари із SEM", "покажи товари з SEM")
  const productKeyword = extractProductKeyword(query);
  if (productKeyword && (lowerQuery.includes('товар') || lowerQuery.includes('продукт'))) {
    return 'product_keyword_search';
  }

  // Superlative keywords for "best/most" queries
  const hasSuperlative =
    lowerQuery.includes('найбільше') ||
    lowerQuery.includes('найдорожче') ||
    lowerQuery.includes('найкращ') ||
    lowerQuery.includes('найбільш') ||
    lowerQuery.includes('найменше') ||
    lowerQuery.includes('найдешевш') ||
    lowerQuery.includes('більше всього') ||
    lowerQuery.includes('найпопулярн');

  const hasClientKeyword =
    lowerQuery.includes('клієнт') ||
    lowerQuery.includes('покупець') ||
    lowerQuery.includes('замовник') ||
    lowerQuery.includes('хто купив');

  const hasProductKeyword =
    lowerQuery.includes('товар') ||
    lowerQuery.includes('продукт') ||
    lowerQuery.includes('що продали') ||
    lowerQuery.includes('що купили');

  // Best/most clients queries
  if (hasSuperlative && hasClientKeyword) {
    return 'top_clients';
  }

  // Best/most products queries
  if (hasSuperlative && hasProductKeyword) {
    return 'top_products';
  }

  // General superlative without specific context - check for buying context
  if (hasSuperlative && (lowerQuery.includes('купив') || lowerQuery.includes('купували'))) {
    return 'top_clients';
  }

  // Sales/Analytics queries
  if (
    lowerQuery.includes('продаж') ||
    lowerQuery.includes('продали') ||
    lowerQuery.includes('скільки') ||
    lowerQuery.includes('по роках') ||
    lowerQuery.includes('за рік') ||
    lowerQuery.includes('yearly')
  ) {
    return 'sales';
  }

  // Product queries
  if (lowerQuery.includes('топ') && (lowerQuery.includes('товар') || lowerQuery.includes('продукт'))) {
    return 'top_products';
  }

  // Client queries
  if (lowerQuery.includes('топ') && lowerQuery.includes('клієнт')) {
    return 'top_clients';
  }

  // Debt queries
  if (
    lowerQuery.includes('борг') ||
    lowerQuery.includes('заборгованість') ||
    lowerQuery.includes('debt')
  ) {
    return 'debts';
  }

  // Region queries
  if (
    lowerQuery.includes('регіон') ||
    lowerQuery.includes('область') ||
    lowerQuery.includes('київ') ||
    lowerQuery.includes('хмельниц') ||
    lowerQuery.includes('львів') ||
    lowerQuery.includes('одес')
  ) {
    return 'region';
  }

  // Client search
  if (lowerQuery.includes('клієнт') && !lowerQuery.includes('топ')) {
    return 'client_search';
  }

  return 'search';
}
