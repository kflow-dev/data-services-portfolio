import { createFileRoute, Link } from "@tanstack/react-router";
import { ScaffoldRoute } from "./-scaffold";

export const Route = createFileRoute("/apps/mymedicine")({
  head: () => ({ meta: [{ title: "MyMedicine — Travel medicine lookup" }] }),
  component: () => <ScaffoldRoute appName="MyMedicine" appTagline="Travel medicine lookup and drug availability checker" />,
});
