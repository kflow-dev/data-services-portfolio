-- pgvector utilities migration
-- Version: 1.0.0
-- Description: Create pgvector extension and utility functions for all recommenders and search apps

-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Function to get the table name for a given app slug
-- Usage: SELECT get_table_name('mywardrobe');
-- Returns: 'app_mywardrobe_data'
CREATE OR REPLACE FUNCTION get_table_name(app_slug TEXT)
RETURNS TEXT AS $$
BEGIN
  RETURN 'app_' || app_slug || '_data';
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to get the vector column name for a given table
-- Usage: SELECT get_vector_column_name('app_mywardrobe_data');
-- Returns: 'embedding'
CREATE OR REPLACE FUNCTION get_vector_column_name(table_name TEXT)
RETURNS TEXT AS $$
BEGIN
  RETURN 'embedding';
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to create a vector search index on a given table and column
-- Usage: SELECT create_vector_index('app_mywardrobe_data', 'embedding');
CREATE OR REPLACE FUNCTION create_vector_index(table_name TEXT, column_name TEXT)
RETURNS VOID AS $$
DECLARE
  index_name TEXT;
BEGIN
  index_name := 'vec_idx_' || table_name || '_' || column_name;
  EXECUTE format('CREATE INDEX IF NOT EXISTS %I ON %I USING ivfflat (%I vector_cosine_ops) WITH (lists=100)',
                 index_name, table_name, column_name);
END;
$$ LANGUAGE plpgsql;

-- Function to add cosine similarity score to a query
-- Usage: SELECT add_cosine_similarity('app_mywardrobe_data', 'embedding', '[0.1,0.2,0.3]');
CREATE OR REPLACE FUNCTION add_cosine_similarity(table_name TEXT, column_name TEXT, vector FLOAT8[])
RETURNS TABLE(id UUID, similarity FLOAT8, data JSONB) AS $$
BEGIN
  RETURN QUERY
  SELECT
    id,
    1 - (column_name <=> vector::vector) as similarity,
    row_to_json(t)::jsonb as data
  FROM (SELECT * FROM get_table_by_name(table_name)) t
  ORDER BY similarity DESC;
END;
$$ LANGUAGE plpgsql;

-- Function to create a new data table for an app
-- Usage: SELECT create_app_table('mywardrobe', 'Description of wardrobe items');
CREATE OR REPLACE FUNCTION create_app_table(app_slug TEXT, description TEXT)
RETURNS TEXT AS $$
DECLARE
  table_name TEXT;
BEGIN
  table_name := get_table_name(app_slug);

  EXECUTE format('
    CREATE TABLE IF NOT EXISTS %I (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      created_at TIMESTAMPTZ DEFAULT NOW(),
      updated_at TIMESTAMPTZ DEFAULT NOW(),
      item_data JSONB NOT NULL,
      embedding vector(1536),
      metadata JSONB DEFAULT ''{}''::jsonb
    )
  ', table_name);

  -- Create index on embedding for cosine similarity search
  EXECUTE format('
    CREATE INDEX IF NOT EXISTS vec_idx_%s ON %I USING ivfflat (embedding vector_cosine_ops) WITH (lists=100)
  ', table_name, table_name);

  RETURN table_name;
END;
$$ LANGUAGE plpgsql;

-- Function to search across all app tables with a given embedding
-- Usage: SELECT search_all_apps('[0.1,0.2,0.3]', 10);
CREATE OR REPLACE FUNCTION search_all_apps(embedding_vector FLOAT8[], top_k INTEGER DEFAULT 10)
RETURNS TABLE(
  app_slug TEXT,
  table_name TEXT,
  id UUID,
  similarity FLOAT8,
  item_data JSONB
) AS $$
DECLARE
  app_table TEXT;
  similarity_score FLOAT8;
BEGIN
  -- Search in each app table
  FOR app_table IN
    SELECT tablename FROM pg_tables WHERE tablename LIKE 'app_%_data'
  LOOP
    app_slug := REPLACE(REPLACE(app_table, 'app_', ''), '_data', '');

    EXECUTE format('
      SELECT 1 - (embedding <=> %L::vector) as sim
      FROM %I
      ORDER BY sim DESC
      LIMIT %L
    ', embedding_vector, app_table, top_k)
    USING similarity_score;

    RETURN QUERY
    EXECUTE format('
      SELECT
        %L as app_slug,
        %L as table_name,
        id,
        1 - (embedding <=> %L::vector) as similarity,
        item_data
      FROM %I
      ORDER BY embedding <=> %L::vector
      LIMIT %L
    ', app_slug, app_table, embedding_vector, app_table, embedding_vector, top_k);
  END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions on utility functions
GRANT EXECUTE ON FUNCTION get_table_name(TEXT) TO authenticated;
GRANT EXECUTE ON FUNCTION get_vector_column_name(TEXT) TO authenticated;
GRANT EXECUTE ON FUNCTION create_vector_index(TEXT, TEXT) TO authenticated;
GRANT EXECUTE ON FUNCTION add_cosine_similarity(TEXT, TEXT, FLOAT8[]) TO authenticated;
GRANT EXECUTE ON FUNCTION create_app_table(TEXT, TEXT) TO authenticated;
GRANT EXECUTE ON FUNCTION search_all_apps(FLOAT8[], INTEGER) TO authenticated;

-- Comment on functions for documentation
COMMENT ON FUNCTION get_table_name(TEXT) IS 'Get the standard table name for an app slug';
COMMENT ON FUNCTION get_vector_column_name(TEXT) IS 'Get the vector column name for a table';
COMMENT ON FUNCTION create_vector_index(TEXT, TEXT) IS 'Create IVFFlat index for cosine similarity search';
COMMENT ON FUNCTION add_cosine_similarity(TEXT, TEXT, FLOAT8[]) IS 'Add cosine similarity scoring to table queries';
COMMENT ON FUNCTION create_app_table(TEXT, TEXT) IS 'Create a new app data table with vector support';
COMMENT ON FUNCTION search_all_apps(FLOAT8[], INTEGER) IS 'Search across all app tables with vector similarity';
