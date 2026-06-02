import { createFileRoute, Link } from "@tanstack/react-router";
import { SiteHeader } from "@/components/SiteHeader";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

// This is a generic scaffold route that renders any app's Streamlit UI
// Each app should have its own route file with proper metadata
export function ScaffoldRoute({ appName, appTagline }: { appName: string; appTagline: string }) {
  return (
    <div className="min-h-screen bg-background">
      <SiteHeader />
      <main className="mx-auto max-w-4xl px-4 py-8">
        <Link to="/apps" className="text-sm text-muted-foreground hover:text-foreground">
          ← Back to dashboard
        </Link>
        <h1 className="mt-2 text-3xl font-bold">{appName}</h1>
        <p className="mt-2 text-muted-foreground">{appTagline}</p>

        <Card className="mt-6">
          <CardHeader>
            <CardTitle className="text-base">Application Status</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <Badge variant="secondary">Scaffold</Badge>
              <p className="text-sm text-muted-foreground">
                This app has a CLI and Streamlit UI. Run it locally with:
              </p>
              <div className="rounded-md bg-muted p-3 text-sm font-mono">
                streamlit run apps/[slug]/streamlit_app.py --server.port 3000
              </div>
              <div className="rounded-md bg-muted p-3 text-sm font-mono">
                python apps/[slug]/cli.py --help
              </div>
            </div>
          </CardContent>
        </Card>
      </main>
    </div>
  );
}
