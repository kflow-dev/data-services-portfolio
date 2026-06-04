import { createFileRoute, Link } from "@tanstack/react-router";
import { ScaffoldRoute } from "./-scaffold";

export const Route = createFileRoute("/apps/datalab-aas")({
  head: () => ({ meta: [{ title: "DataLab-as-a-Service — Jupyter workbench" }] }),
  component: () => <ScaffoldRoute appName="DataLab-as-a-Service" appTagline="Jupyter workbench for data science teams on Kubernetes" />,
});
