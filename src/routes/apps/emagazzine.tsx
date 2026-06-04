import { createFileRoute, Link } from "@tanstack/react-router";
import { ScaffoldRoute } from "./-scaffold";

export const Route = createFileRoute("/apps/emagazzine")({
  head: () => ({ meta: [{ title: "EMagazzine — Price comparator" }] }),
  component: () => <ScaffoldRoute appName="EMagazzine" appTagline="Price comparator and multi-objective product tracker with Scrapy and Pareto optimization" />,
});
