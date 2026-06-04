import { corsHeaders } from "../_shared/cors.ts";

const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");

// Per-app system prompts. Each instructs the LLM to produce strict JSON of shape
// {"summary": "...", "items": [{"title": "...", "detail": "..."}]}
const PROMPTS: Record<string, string> = {
  "drift-monitor":
    "You are a model-drift analyst. Given a description of an ML model and its recent input/output stats, return a JSON drift report with summary plus items where each item is a drift signal (PSI, KS, label drift, latency, etc.) with the metric value and recommended action.",
  "datalab-aas":
    "You are a DataLab-as-a-Service architect. Given a team description, return a JSON plan with summary plus items describing recommended JupyterHub config, kernels, storage, GPU/CPU sizing, and governance steps.",
  "segmentation":
    "You are a customer-segmentation analyst. Given a brief about a customer base, return JSON with summary and items where each item is one persona segment with size estimate, key traits, and a marketing hook.",
  "jobminder":
    "You are JobMinder, a friendly job-search chatbot. Given the user's situation, return JSON with a conversational summary and items where each item is a concrete next action (skill to learn, role to apply for, recruiter to contact) with rationale.",
  "sku-forecast":
    "You are a demand-forecasting expert using foundation time-series models (TimesFM, Chronos). Given a SKU description and recent demand notes, return JSON with summary plus items where each item is a forecast horizon (next week / month / quarter) with point forecast, range, and risk note.",
  "aifluent":
    "You are AIFluent, a skills-acquisition tutor. Given the learner goal, return JSON with summary plus items where each item is a learning module with topic, suggested resource type, time estimate, and assessment idea.",
  "chap":
    "You are CHAP, a common hybrid agent platform orchestrator. Given the user task, return JSON with summary plus items where each item is a sub-agent in the plan with role, tool calls, and expected output.",
  "auctionlab":
    "You are an auction-simulation expert. Given an auction setup (format, bidders, valuations), return JSON with summary plus items where each item is a simulated round or strategy with expected revenue, winner profile, and insight.",
  "emagazzine":
    "You are EMagazzine, a multi-objective price comparator. Given a product query and weights for price/quality/delivery, return JSON with summary plus items where each item is a candidate offer with merchant, price, score, and tradeoff note.",
  "skillsplan":
    "You are SkillsPlan, a curriculum optimizer balancing cost, time, and knowledge impact. Given a target role/skill, return JSON with summary plus items where each item is a curriculum step with course, cost, hours, and impact score.",
  "mysmartdiet":
    "You are MySmartDiet, a healthy-diet recommender. Given the user's goals, restrictions, and preferences, return JSON with summary plus items where each item is one meal with macros, ingredients, and rationale.",
  "cloud-ml-estimator":
    "You are a Cloud ML solution & pricing estimator. Given a workload description, return JSON with summary plus items where each item is one cloud option (AWS/GCP/Azure component) with monthly cost estimate, assumptions, and fit note.",
};

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return new Response(null, { headers: corsHeaders });
  try {
    const { slug, input } = await req.json();
    const system = PROMPTS[slug];
    if (!system) {
      return new Response(JSON.stringify({ error: `unknown slug: ${slug}` }), {
        status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
    const r = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
      method: "POST",
      headers: { Authorization: `Bearer ${LOVABLE_API_KEY}`, "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "google/gemini-2.5-flash",
        messages: [
          { role: "system", content: system + ' Always output ONLY JSON of shape {"summary":"...","items":[{"title":"...","detail":"..."}]}.' },
          { role: "user", content: String(input ?? "") },
        ],
        response_format: { type: "json_object" },
      }),
    });
    if (!r.ok) {
      return new Response(JSON.stringify({ error: await r.text() }), {
        status: r.status, headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
    const data = await r.json();
    const content = data.choices?.[0]?.message?.content ?? '{"summary":"","items":[]}';
    return new Response(content, { headers: { ...corsHeaders, "Content-Type": "application/json" } });
  } catch (e) {
    return new Response(JSON.stringify({ error: String(e) }), {
      status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});