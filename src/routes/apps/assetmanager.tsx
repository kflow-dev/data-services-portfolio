import { createFileRoute, Link } from "@tanstack/react-router";
import { ScaffoldRoute } from "./-scaffold";

export const Route = createFileRoute("/apps/assetmanager")({
  head: () => ({ meta: [{ title: "AssetManager — Article summarizer + NER" }] }),
  component: () => <ScaffoldRoute appName="AssetManager" appTagline="Multi-media article summarizer with NER using spaCy and LLM" />,
});
