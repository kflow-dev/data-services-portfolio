import { createFileRoute, Link } from "@tanstack/react-router";
import { APPS, CATEGORIES } from "@/lib/portfolio";
import { SiteHeader } from "@/components/SiteHeader";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "MyDataSciencePortfolio — RAG & ML showcase" },
      {
        name: "description",
        content:
          "Portfolio of 25+ data science apps: recommenders, RAG pipelines, chatbots, forecasters, agents, and more.",
      },
    ],
  }),
  component: Dashboard,
});

function StatusBadge({ status }: { status: "live" | "scaffold" | "planned" }) {
  const map = {
    live: { label: "Live demo", className: "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400" },
    scaffold: { label: "Scaffold", className: "bg-amber-500/15 text-amber-600 dark:text-amber-400" },
    planned: { label: "Planned", className: "bg-muted text-muted-foreground" },
  } as const;
  return <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${map[status].className}`}>{map[status].label}</span>;
}

function Dashboard() {
  return (
    <div className="min-h-screen bg-background">
      <SiteHeader />
      <main className="mx-auto max-w-6xl px-4 py-10">
        <section className="mb-12">
          <h1 className="text-4xl font-bold tracking-tight">MyDataSciencePortfolio</h1>
          <p className="mt-3 max-w-2xl text-muted-foreground">
            A full RAG + ML product portfolio: 25+ data science apps spanning recommenders, search, agents,
            forecasters, and optimization. Each app ships with a CLI entry point and a Streamlit UI, plus the
            dashboard you're looking at. Three demos are wired up live below.
          </p>
          <div className="mt-4 flex flex-wrap gap-2 text-xs text-muted-foreground">
            <Badge variant="outline">Next.js / TanStack Start :7000</Badge>
            <Badge variant="outline">Streamlit :3000</Badge>
            <Badge variant="outline">nginx reverse proxy</Badge>
            <Badge variant="outline">GitHub Actions → VPS</Badge>
            <Badge variant="outline">pgvector RAG</Badge>
          </div>
        </section>

        {CATEGORIES.map((cat) => {
          const items = APPS.filter((a) => a.category === cat);
          if (!items.length) return null;
          return (
            <section key={cat} className="mb-12">
              <h2 className="mb-4 text-xl font-semibold tracking-tight">{cat}</h2>
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {items.map((app) => {
                  const inner = (
                    <Card className="h-full transition-all hover:border-primary/40 hover:shadow-md">
                      <CardHeader>
                        <div className="mb-2 flex items-center justify-between">
                          <CardTitle className="text-base">{app.name}</CardTitle>
                          <StatusBadge status={app.status} />
                        </div>
                        <CardDescription>{app.tagline}</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="flex flex-wrap gap-1">
                          {app.stack.map((s) => (
                            <Badge key={s} variant="secondary" className="text-[10px] font-normal">
                              {s}
                            </Badge>
                          ))}
                        </div>
                      </CardContent>
                    </Card>
                  );
                  return app.route ? (
                    <Link key={app.slug} to={app.route} className="block">
                      {inner}
                    </Link>
                  ) : (
                    <div key={app.slug}>{inner}</div>
                  );
                })}
              </div>
            </section>
          );
        })}

        <footer className="mt-16 border-t border-border pt-6 text-sm text-muted-foreground">
          Built with TanStack Start, Tailwind, Lovable Cloud (Postgres + pgvector), and the Lovable AI Gateway.
          See <code className="rounded bg-muted px-1">README.md</code>, <code className="rounded bg-muted px-1">docker-compose.yml</code>,
          and <code className="rounded bg-muted px-1">.github/workflows/</code> for the VPS deployment pipeline.
        </footer>
      </main>
    </div>
  );
}
