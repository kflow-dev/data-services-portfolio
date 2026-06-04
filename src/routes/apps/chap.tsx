import { createFileRoute, Link } from "@tanstack/react-router";
import { ScaffoldRoute } from "./-scaffold";

export const Route = createFileRoute("/apps/chap")({
  head: () => ({ meta: [{ title: "CHAP — Hybrid agent platform" }] }),
  component: () => <ScaffoldRoute appName="CHAP" appTagline="Common hybrid agent platform for knowledge exchange and multi-agent simulation" />,
});
