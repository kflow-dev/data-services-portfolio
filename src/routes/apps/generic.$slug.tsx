import { createFileRoute, Link, notFound } from "@tanstack/react-router";
import { useState } from "react";
import { SiteHeader } from "@/components/SiteHeader";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Loader2 } from "lucide-react";
import { GENERIC_APPS, type GenericSpec } from "@/lib/genericApps";

export const Route = createFileRoute("/apps/generic/$slug")({
  loader: ({ params }): GenericSpec => {
    const spec = GENERIC_APPS[params.slug];
    if (!spec) throw notFound();
    return spec;
  },
  head: ({ loaderData }) => ({
    meta: [{ title: `${loaderData?.name ?? "App"} — MyDataSciencePortfolio` }],
  }),
  component: GenericPage,
  notFoundComponent: () => (
    <div className="min-h-screen bg-background">
      <SiteHeader />
      <main className="mx-auto max-w-4xl px-4 py-10">
        <Link to="/" className="text-sm text-muted-foreground">← Back</Link>
        <h1 className="mt-2 text-2xl font-bold">App not found</h1>
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

type Result = { summary: string; items: { title: string; detail: string }[] };

function GenericPage() {
  const spec = Route.useLoaderData();
  const [input, setInput] = useState(spec.defaultInput);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<Result | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function run() {
    setLoading(true); setErr(null); setResult(null);
    try {
      const r = await fetch(`${supabaseUrl}/functions/v1/generic-app`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${anonKey}`, apikey: anonKey },
        body: JSON.stringify({ slug: spec.slug, input }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || "Failed");
      setResult({ summary: data.summary ?? "", items: data.items ?? [] });
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
          <CardHeader><CardTitle className="text-base">{spec.inputLabel}</CardTitle></CardHeader>
          <CardContent className="space-y-3">
            <Textarea value={input} onChange={(e) => setInput(e.target.value)} rows={spec.rows ?? 4} />
            <Button onClick={run} disabled={loading}>
              {loading && <Loader2 className="mr-2 size-4 animate-spin" />}
              Run
            </Button>
          </CardContent>
        </Card>

        {err && <p className="mt-4 text-sm text-destructive">{err}</p>}

        {result && (
          <div className="mt-6 space-y-3">
            <Card>
              <CardHeader><CardTitle className="text-base">Summary</CardTitle></CardHeader>
              <CardContent><p className="whitespace-pre-wrap text-sm">{result.summary}</p></CardContent>
            </Card>
            {result.items.map((it, i) => (
              <Card key={i}>
                <CardContent className="pt-4">
                  <div className="text-xs uppercase tracking-wide text-muted-foreground">{it.title}</div>
                  <div className="mt-1 whitespace-pre-wrap text-sm">{it.detail}</div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
