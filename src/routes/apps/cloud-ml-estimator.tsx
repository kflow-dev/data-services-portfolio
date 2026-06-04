import { createFileRoute, Link } from "@tanstack/react-router";
import { ScaffoldRoute } from "./-scaffold";

export const Route = createFileRoute("/apps/cloud-ml-estimator")({
  head: () => ({ meta: [{ title: "Cloud ML Estimator — Pricing estimator" }] }),
  component: () => <ScaffoldRoute appName="Cloud ML Estimator" appTagline="Cloud ML solution and pricing estimator using pricing APIs and heuristics" />,
});
