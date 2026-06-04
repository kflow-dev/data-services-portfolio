import { createFileRoute, Link } from "@tanstack/react-router";
import { ScaffoldRoute } from "./-scaffold";

export const Route = createFileRoute("/apps/mylocalradar")({
  head: () => ({ meta: [{ title: "MyLocalRadar — Location disambiguation" }] }),
  component: () => <ScaffoldRoute appName="MyLocalRadar" appTagline="Location mapping and disambiguation with Geo NER and OpenStreetMap" />,
});
