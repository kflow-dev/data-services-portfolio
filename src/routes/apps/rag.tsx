import { createFileRoute, Link } from "@tanstack/react-router";
import { useState } from "react";
import ReactMarkdown from "react-markdown";
import { SiteHeader } from "@/components/SiteHeader";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Loader2 } from "lucide-react";

export const Route = createFileRoute("/apps/rag")({
  head: () => ({ meta: [{ title: "Multi-media RAG — search, NER & Q&A" }] }),
  component: RagApp,
});

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL as string;
const anonKey = import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY as string;

type Match = { id: string; source: string; content: string; similarity: number };

function RagApp() {
  const [source, setSource] = useState("");
  const [content, setContent] = useState("");
  const [ingestMsg, setIngestMsg] = useState<string | null>(null);
  const [ingesting, setIngesting] = useState(false);

  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<string | null>(null);
  const [matches, setMatches] = useState<Match[]>([]);
  const [querying, setQuerying] = useState(false);

  async function ingest() {
    if (!content.trim()) return;
    setIngesting(true);
    setIngestMsg(null);
    try {
      const r = await fetch(`${supabaseUrl}/functions/v1/rag-ingest`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${anonKey}`, apikey: anonKey },
        body: JSON.stringify({ source: source || "untitled", content }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || "Failed");
      setIngestMsg(`Indexed ${d.inserted} chunks from "${source || "untitled"}".`);
      setContent("");
    } catch (e) {
      setIngestMsg(`Error: ${String(e)}`);
    } finally {
      setIngesting(false);
    }
  }

  async function ask() {
    if (!question.trim()) return;
    setQuerying(true);
    setAnswer(null);
    setMatches([]);
    try {
      const r = await fetch(`${supabaseUrl}/functions/v1/rag-query`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${anonKey}`, apikey: anonKey },
        body: JSON.stringify({ question, k: 5 }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || "Failed");
      setAnswer(d.answer);
      setMatches(d.matches || []);
    } catch (e) {
      setAnswer(`_Error: ${String(e)}_`);
    } finally {
      setQuerying(false);
    }
  }

  return (
    <div className="min-h-screen bg-background">
      <SiteHeader />
      <main className="mx-auto max-w-4xl px-4 py-8">
        <Link to="/" className="text-sm text-muted-foreground hover:text-foreground">← Back to dashboard</Link>
        <h1 className="mt-2 text-3xl font-bold">Multi-media RAG</h1>
        <p className="mt-2 text-muted-foreground">
          Index any text into pgvector (1536-dim, cosine HNSW), then ask grounded questions with citations.
        </p>

        <div className="mt-6 grid gap-6 md:grid-cols-2">
          <Card>
            <CardHeader><CardTitle className="text-base">1. Ingest content</CardTitle></CardHeader>
            <CardContent className="space-y-3">
              <Input
                placeholder="Source name (e.g. product-catalog.pdf)"
                value={source}
                onChange={(e) => setSource(e.target.value)}
              />
              <Textarea
                placeholder="Paste any text — article, transcript, product catalog…"
                value={content}
                onChange={(e) => setContent(e.target.value)}
                rows={8}
              />
              <Button onClick={ingest} disabled={ingesting || !content.trim()}>
                {ingesting && <Loader2 className="mr-2 size-4 animate-spin" />}
                Embed & index
              </Button>
              {ingestMsg && <p className="text-sm text-muted-foreground">{ingestMsg}</p>}
            </CardContent>
          </Card>

          <Card>
            <CardHeader><CardTitle className="text-base">2. Ask a question</CardTitle></CardHeader>
            <CardContent className="space-y-3">
              <Textarea
                placeholder="What does the catalog say about…?"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                rows={3}
              />
              <Button onClick={ask} disabled={querying || !question.trim()}>
                {querying && <Loader2 className="mr-2 size-4 animate-spin" />}
                Ask
              </Button>
              {answer && (
                <div className="prose prose-sm dark:prose-invert max-w-none rounded-md border border-border bg-muted/40 p-3">
                  <ReactMarkdown>{answer}</ReactMarkdown>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {matches.length > 0 && (
          <section className="mt-8">
            <h2 className="mb-3 text-lg font-semibold">Retrieved chunks</h2>
            <div className="space-y-2">
              {matches.map((m, i) => (
                <Card key={m.id}>
                  <CardContent className="pt-4">
                    <div className="mb-1 flex items-center justify-between text-xs text-muted-foreground">
                      <span>[{i + 1}] {m.source}</span>
                      <span>sim {(m.similarity * 100).toFixed(1)}%</span>
                    </div>
                    <p className="text-sm">{m.content}</p>
                  </CardContent>
                </Card>
              ))}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
