import { createFileRoute, Link } from "@tanstack/react-router";
import { ScaffoldRoute } from "./-scaffold";

export const Route = createFileRoute("/apps/jobminder")({
  head: () => ({ meta: [{ title: "JobMinder — Job recommender chatbot" }] }),
  component: () => <ScaffoldRoute appName="JobMinder" appTagline="Job recommender chatbot with agentic flow and tools" />,
});
