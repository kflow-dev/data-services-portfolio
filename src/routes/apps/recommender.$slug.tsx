import { createFileRoute, Link, notFound } from "@tanstack/react-router";
import { useState } from "react";
import { SiteHeader } from "@/components/SiteHeader";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Loader2 } from "lucide-react";
import { RECOMMENDERS, type RecommenderSpec, type CatalogItem } from "@/lib/recommenderCatalogs";

export const Route = createFileRoute("/apps/recommender/$slug")({
  loader: ({ params }): RecommenderSpec => {
    const spec = RECOMMENDERS[params.slug];
    if (!spec) throw notFound();
    return spec;
  },
  head: ({ loaderData }) => ({
    meta: [{ title: `${loaderData?.name ?? "Recommender"} — MyDataSciencePortfolio` }],
  }),
  component: RecommenderPage,
  notFoundComponent: () => (
    <div className="min-h-screen bg-background">
      <SiteHeader />
      <main className="mx-auto max-w-4xl px-4 py-10">
        <Link to="/" className="text-sm text-muted-foreground">← Back</Link>
        <h1 className="mt-2 text-2xl font-bold">Recommender not found</h1>
      </main>
    </div>
  ),
  errorComponent: ({ error }) => (
    <div className="min-h-screen bg-background">
      <SiteHeader />
      <main className="mx-auto max-w-4xl px-4 py-10">
        <p className="text-sm text-destructive">{String(error)}</p>
      </main>
    </div>
  ),
});

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL as string;
const anonKey = import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY as string;

type Ranked = { id: string; score: number; reason: string };

function RecommenderPage() {
  const spec = Route.useLoaderData();
  const [context, setContext] = useState(spec.defaultContext);
  const [loading, setLoading] = useState(false);
  const [ranked, setRanked] = useState<Ranked[] | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function recommend() {
    setLoading(true); setErr(null); setRanked(null);
    try {
      const r = await fetch(`${supabaseUrl}/functions/v1/recommend`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${anonKey}`, apikey: anonKey },
        body: JSON.stringify({ domain: spec.domain, context, items: spec.items, topK: 5 }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || "Failed");
      setRanked(data.ranked || []);
    } catch (e) { setErr(String(e)); } finally { setLoading(false); }
  }

  return (
    <div className="min-h-screen bg-background">
      <SiteHeader />
      <main className="mx-auto max-w-4xl px-4 py-8">
        <Link to="/" className="text-sm text-muted-foreground hover:text-foreground">← Back to dashboard</Link>
        <h1 className="mt-2 text-3xl font-bold">{spec.name}</h1>
        <p className="mt-2 text-muted-foreground">{spec.tagline}</p>

        <Card className="mt-6">
          <CardHeader><CardTitle className="text-base">{spec.promptLabel}</CardTitle></CardHeader>
          <CardContent className="space-y-3">
            <Textarea value={context} onChange={(e) => setContext(e.target.value)} rows={3} />
            <Button onClick={recommend} disabled={loading}>
              {loading && <Loader2 className="mr-2 size-4 animate-spin" />}
              Recommend top 5
            </Button>
          </CardContent>
        </Card>

        {err && <p className="mt-4 text-sm text-destructive">{err}</p>}

        {ranked && (
          <div className="mt-6 space-y-3">
            <h2 className="text-lg font-semibold">Top picks</h2>
            {ranked.map((r, i) => {
              const it = spec.items.find((a: { id: string }) => a.id === r.id);
              if (!it) return null;
              return (
                <Card key={r.id}>
                  <CardContent className="pt-4">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <div className="text-xs text-muted-foreground">#{i + 1}{it.meta ? ` · ${it.meta}` : ""}</div>
                        <div className="font-medium">{it.title}</div>
                        <div className="mt-1 text-sm text-muted-foreground">{r.reason}</div>
                      </div>
                      <div className="text-sm tabular-nums text-primary">{(r.score * 100).toFixed(0)}%</div>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        )}

        <details className="mt-8 text-sm text-muted-foreground">
          <summary className="cursor-pointer">Catalog ({spec.items.length} items)</summary>
          <ul className="mt-2 list-disc pl-5">
            {spec.items.map((a: CatalogItem) => (
              <li key={String(a.id)}>{String(a.title)}{a.meta ? ` — ${String(a.meta)}` : ""}</li>
            ))}
          </ul>
        </details>
      </main>
    </div>
  );
}
