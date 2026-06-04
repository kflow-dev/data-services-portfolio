import { createFileRoute, Link } from "@tanstack/react-router";
import { ScaffoldRoute } from "./-scaffold";

export const Route = createFileRoute("/apps/scitubbies")({
  head: () => ({ meta: [{ title: "SciTubbies — YouTube content recommender" }] }),
  component: () => <ScaffoldRoute appName="SciTubbies" appTagline="YouTube content recommender for science and tech education" />,
});
