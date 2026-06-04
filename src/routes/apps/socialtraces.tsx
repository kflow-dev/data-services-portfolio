import { createFileRoute, Link } from "@tanstack/react-router";
import { ScaffoldRoute } from "./-scaffold";

export const Route = createFileRoute("/apps/socialtraces")({
  head: () => ({ meta: [{ title: "SocialTraces — Fuzzy search detective" }] }),
  component: () => <ScaffoldRoute appName="SocialTraces" appTagline="Social media fuzzy search detective with graph analysis" />,
});
