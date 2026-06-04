import { createFileRoute, Link } from "@tanstack/react-router";
import { ScaffoldRoute } from "./-scaffold";

export const Route = createFileRoute("/apps/aifluent")({
  head: () => ({ meta: [{ title: "AIFluent — Skills acquisition platform" }] }),
  component: () => <ScaffoldRoute appName="AIFluent" appTagline="Skills acquisition platform with agentic tutor" />,
});
