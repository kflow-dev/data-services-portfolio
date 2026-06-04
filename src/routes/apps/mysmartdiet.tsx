import { createFileRoute, Link } from "@tanstack/react-router";
import { ScaffoldRoute } from "./-scaffold";

export const Route = createFileRoute("/apps/mysmartdiet")({
  head: () => ({ meta: [{ title: "MySmartDiet — Healthy diet recommender" }] }),
  component: () => <ScaffoldRoute appName="MySmartDiet" appTagline="Healthy diet recommender using constraint solver optimization" />,
});
