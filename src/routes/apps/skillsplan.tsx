import { createFileRoute, Link } from "@tanstack/react-router";
import { ScaffoldRoute } from "./-scaffold";

export const Route = createFileRoute("/apps/skillsplan")({
  head: () => ({ meta: [{ title: "SkillsPlan — Curriculum optimizer" }] }),
  component: () => <ScaffoldRoute appName="SkillsPlan" appTagline="Curriculum builder optimizing cost, time and impact using MILP and OR-Tools" />,
});
