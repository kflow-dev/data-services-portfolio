import { createFileRoute, Link } from "@tanstack/react-router";
import { ScaffoldRoute } from "./-scaffold";

export const Route = createFileRoute("/apps/auctionlab")({
  head: () => ({ meta: [{ title: "AuctionLab — Auction experimentation" }] }),
  component: () => <ScaffoldRoute appName="AuctionLab" appTagline="Auction experimentation platform for mechanism design research" />,
});
