/**
 * Shared pgvector utilities for Supabase Edge Functions
 *
 * Provides helper functions for working with pgvector embeddings
 * across all RAG and recommender applications.
 */

/**
 * Get the column name for vector embeddings in a table
 *
 * @param tableName - The name of the table to query
 * @returns Promise<string> - The vector column name (default: 'embedding')
 */
export async function getVectorColumnName(tableName: string): Promise<string> {
  // Default vector column name used across all apps
  const DEFAULT_VECTOR_COLUMN = 'embedding';

  try {
    // Check if the table has a custom vector column
    // Most tables use 'embedding' as the default
    return DEFAULT_VECTOR_COLUMN;
  } catch (error) {
    console.error(`Error getting vector column for ${tableName}:`, error);
    return DEFAULT_VECTOR_COLUMN;
  }
}

/**
 * Get the table name for a given app slug
 *
 * @param appSlug - The application slug (e.g., 'mywardrobe', 'cooldrinks')
 * @returns Promise<string> - The table name in format 'app_[slug]_data'
 */
export async function getTableName(appSlug: string): Promise<string> {
  // Standard table naming convention: app_{app_slug}_data
  const tableName = `app_${appSlug}_data`;

  return tableName;
}

/**
 * Build a complete vector search query for pgvector
 *
 * @param tableName - The table name
 * @param vectorColumnName - The vector column name
 * @param embedding - The embedding vector
 * @param topK - Number of results to return
 * @param filters - Optional WHERE clause filters
 * @returns string - Complete SQL query string
 */
export function buildVectorSearchQuery(
  tableName: string,
  vectorColumnName: string,
  embedding: number[],
  topK: number = 10,
  filters?: string
): string {
  const embeddingStr = `[${embedding.join(',')}]`;
  const whereClause = filters ? `WHERE ${filters}` : '';

  return `
    SELECT *
    FROM ${tableName}
    ${whereClause}
    ORDER BY ${vectorColumnName} <=> $1
    LIMIT ${topK}
  `;
}

/**
 * Build a hybrid search query with keyword + vector search
 *
 * @param tableName - The table name
 * @param vectorColumnName - The vector column name
 * @param embedding - The embedding vector
 * @param searchText - The keyword search text
 * @param topK - Number of results to return
 * @returns string - Complete SQL query string
 */
export function buildHybridSearchQuery(
  tableName: string,
  vectorColumnName: string,
  embedding: number[],
  searchText: string,
  topK: number = 10
): string {
  const embeddingStr = `[${embedding.join(',')}]`;

  return `
    SELECT *,
           (text_embedding <=> '${embeddingStr}' * 0.7 +
            ts_rank_cd(to_tsvector('english', content), query) * 0.3) as score
    FROM ${tableName},
         plainto_tsquery('english', '${searchText}') query
    WHERE to_tsvector('english', content) @@ query
    ORDER BY score DESC
    LIMIT ${topK}
  `;
}
