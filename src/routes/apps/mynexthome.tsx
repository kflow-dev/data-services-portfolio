import { createFileRoute, Link } from "@tanstack/react-router";
import { ScaffoldRoute } from "./-scaffold";

export const Route = createFileRoute("/apps/mynexthome")({
  head: () => ({ meta: [{ title: "MyNextHome — Real-estate recommender" }] }),
  component: () => <ScaffoldRoute appName="MyNextHome" appTagline="Real-estate recommender with HMM-based price forecasting" />,
});
