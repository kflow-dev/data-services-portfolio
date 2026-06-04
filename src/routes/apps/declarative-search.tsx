import { createFileRoute, Link } from "@tanstack/react-router";
import { ScaffoldRoute } from "./-scaffold";

export const Route = createFileRoute("/apps/declarative-search")({
  head: () => ({ meta: [{ title: "Declarative Search — Multi-agent scrape" }] }),
  component: () => <ScaffoldRoute appName="Declarative Search" appTagline="Multi-agent search and scraping with LangGraph and Playwright" />,
});
