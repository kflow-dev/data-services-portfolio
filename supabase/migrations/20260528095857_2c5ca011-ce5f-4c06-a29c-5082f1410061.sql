CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE public.rag_documents (
  id uuid primary key default gen_random_uuid(),
  source text not null default 'untitled',
  content text not null,
  embedding vector(1536) not null,
  created_at timestamptz not null default now()
);

GRANT SELECT, INSERT, DELETE ON public.rag_documents TO anon, authenticated;
GRANT ALL ON public.rag_documents TO service_role;

ALTER TABLE public.rag_documents ENABLE ROW LEVEL SECURITY;

CREATE POLICY "rag public read" ON public.rag_documents FOR SELECT TO anon, authenticated USING (true);
CREATE POLICY "rag public insert" ON public.rag_documents FOR INSERT TO anon, authenticated WITH CHECK (true);
CREATE POLICY "rag public delete" ON public.rag_documents FOR DELETE TO anon, authenticated USING (true);

CREATE INDEX rag_documents_embedding_idx ON public.rag_documents USING hnsw (embedding vector_cosine_ops);

CREATE OR REPLACE FUNCTION public.match_rag(query_embedding vector(1536), match_count int DEFAULT 5)
RETURNS TABLE (id uuid, source text, content text, similarity float)
LANGUAGE sql STABLE
AS $$
  SELECT id, source, content, 1 - (embedding <=> query_embedding) AS similarity
  FROM public.rag_documents
  ORDER BY embedding <=> query_embedding
  LIMIT match_count;
$$;

GRANT EXECUTE ON FUNCTION public.match_rag(vector, int) TO anon, authenticated;