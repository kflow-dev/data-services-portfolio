import { createFileRoute, Link } from "@tanstack/react-router";
import { ScaffoldRoute } from "./-scaffold";

export const Route = createFileRoute("/apps/sku-forecast")({
  head: () => ({ meta: [{ title: "SKU Demand Forecaster — Foundation models" }] }),
  component: () => <ScaffoldRoute appName="SKU Demand Forecaster" appTagline="Foundation models (TimesFM, Chronos) for product SKU demand forecasting" />,
});
