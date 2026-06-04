import { createFileRoute, Link } from "@tanstack/react-router";
import { ScaffoldRoute } from "./-scaffold";

export const Route = createFileRoute("/apps/drift-monitor")({
  head: () => ({ meta: [{ title: "Drift Monitor — Model & data drift" }] }),
  component: () => <ScaffoldRoute appName="Drift Monitor" appTagline="Model and data drift monitoring asset with Evidently and Prometheus" />,
});
