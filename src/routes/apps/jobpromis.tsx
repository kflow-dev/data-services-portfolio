import { createFileRoute, Link } from "@tanstack/react-router";
import { ScaffoldRoute } from "./-scaffold";

export const Route = createFileRoute("/apps/jobpromis")({
  head: () => ({ meta: [{ title: "JobPromis — Job recommender app" }] }),
  component: () => <ScaffoldRoute appName="JobPromis" appTagline="Job recommender app with skills gap analysis" />,
});
