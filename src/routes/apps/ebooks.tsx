import { createFileRoute, Link } from "@tanstack/react-router";
import { ScaffoldRoute } from "./-scaffold";

export const Route = createFileRoute("/apps/ebooks")({
  head: () => ({ meta: [{ title: "E-books/Audiobook RecSys — Content-based recommender" }] }),
  component: () => <ScaffoldRoute appName="E-books/Audiobook RecSys" appTagline="Content-based recommender for reading materials" />,
});
