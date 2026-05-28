import { createFileRoute, Link } from "@tanstack/react-router";
import { useState } from "react";
import { SiteHeader } from "@/components/SiteHeader";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Loader2 } from "lucide-react";

export const Route = createFileRoute("/apps/hotnews4u")({
  head: () => ({ meta: [{ title: "HotNews4U — Personalized news recommender" }] }),
  component: HotNews,
});

const SAMPLE = [
  { id: "a1", title: "Apple announces M5 chip with on-device LLM acceleration", category: "tech" },
  { id: "a2", title: "ECB cuts interest rates by 25bps amid easing inflation", category: "finance" },
  { id: "a3", title: "New transformer architecture beats Gemini 2.5 on MMLU", category: "ai" },
  { id: "a4", title: "Real Madrid wins Club World Cup final 3-1", category: "sports" },
  { id: "a5", title: "EU AI Act compliance deadline arrives for general-purpose models", category: "policy" },
  { id: "a6", title: "Berlin startup raises $40M for vector DB infrastructure", category: "startup" },
  { id: "a7", title: "Mediterranean diet linked to longer cognitive lifespan in 30-yr study", category: "health" },
  { id: "a8", title: "SpaceX launches first crewed lunar flyby", category: "space" },
  { id: "a9", title: "Climate report: Arctic ice loss accelerates 18% YoY", category: "climate" },
  { id: "a10", title: "Open-source LLM gateway Lovable AI hits 1M daily requests", category: "ai" },
];

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL as string;
const anonKey = import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY as string;

type Ranked = { id: string; score: number; reason: string };

function HotNews() {
  const [interests, setInterests] = useState("AI infrastructure, machine learning research, European tech startups");
  const [loading, setLoading] = useState(false);
  const [ranked, setRanked] = useState<Ranked[] | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function recommend() {
    setLoading(true);
    setErr(null);
    setRanked(null);
    try {
      const r = await fetch(`${supabaseUrl}/functions/v1/news-recommend`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${anonKey}`, apikey: anonKey },
        body: JSON.stringify({ interests, articles: SAMPLE }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || "Failed");
      setRanked(data.ranked || []);
    } catch (e) {
      setErr(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-background">
      <SiteHeader />
      <main className="mx-auto max-w-4xl px-4 py-8">
        <Link to="/" className="text-sm text-muted-foreground hover:text-foreground">← Back to dashboard</Link>
        <h1 className="mt-2 text-3xl font-bold">HotNews4U</h1>
        <p className="mt-2 text-muted-foreground">
          Personalized news ranking. Tell us your interests, the LLM scores and explains each pick.
        </p>

        <Card className="mt-6">
          <CardHeader><CardTitle className="text-base">Your interests</CardTitle></CardHeader>
          <CardContent className="space-y-3">
            <Input value={interests} onChange={(e) => setInterests(e.target.value)} />
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
              const art = SAMPLE.find((a) => a.id === r.id);
              if (!art) return null;
              return (
                <Card key={r.id}>
                  <CardContent className="pt-4">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <div className="text-xs text-muted-foreground">#{i + 1} · {art.category}</div>
                        <div className="font-medium">{art.title}</div>
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
          <summary className="cursor-pointer">Sample article pool ({SAMPLE.length})</summary>
          <ul className="mt-2 list-disc pl-5">
            {SAMPLE.map((a) => <li key={a.id}>[{a.category}] {a.title}</li>)}
          </ul>
        </details>
      </main>
    </div>
  );
}
