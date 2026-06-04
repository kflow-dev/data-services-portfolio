import { createFileRoute, Link } from "@tanstack/react-router";
import { ScaffoldRoute } from "./-scaffold";

export const Route = createFileRoute("/apps/segmentation")({
  head: () => ({ meta: [{ title: "Customer Segmentation — Persona creation" }] }),
  component: () => <ScaffoldRoute appName="Customer Segmentation" appTagline="Representative customers and persona creation using KMeans and UMAP" />,
});
