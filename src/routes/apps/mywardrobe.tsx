import { createFileRoute, Link } from "@tanstack/react-router";
import { ScaffoldRoute } from "./-scaffold";

export const Route = createFileRoute("/apps/mywardrobe")({
  head: () => ({ meta: [{ title: "MyWardrobe — ShopTheLook outfit recommender" }] }),
  component: () => <ScaffoldRoute appName="MyWardrobe" appTagline="ShopTheLook outfit recommender for B2B shops" />,
});
