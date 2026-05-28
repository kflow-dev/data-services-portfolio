import { corsHeaders } from "../_shared/cors.ts";

const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");

const PERSONAS: Record<string, string> = {
  papie:
    "You are PAPIE, a Personal Assistant for Personal Information Exchange. " +
    "You help the user organize, retrieve and exchange personal info (contacts, notes, IDs, preferences). " +
    "Be concise, structured (markdown), and privacy-aware. Never invent personal data.",
  jobminder:
    "You are JobMinder, a friendly job-search chatbot. Ask about role, seniority, location, " +
    "remote preference, and tech stack. Recommend job categories and next-step actions.",
  default: "You are a helpful AI assistant in the MyDataSciencePortfolio demo.",
};

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return new Response(null, { headers: corsHeaders });
  if (!LOVABLE_API_KEY) {
    return new Response(JSON.stringify({ error: "LOVABLE_API_KEY not configured" }), {
      status: 500,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }

  try {
    const { messages, persona = "default" } = await req.json();
    const system = PERSONAS[persona] ?? PERSONAS.default;

    const upstream = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${LOVABLE_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "google/gemini-2.5-flash",
        stream: true,
        messages: [{ role: "system", content: system }, ...messages],
      }),
    });

    if (upstream.status === 429)
      return new Response(JSON.stringify({ error: "Rate limited" }), {
        status: 429,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    if (upstream.status === 402)
      return new Response(JSON.stringify({ error: "AI credits exhausted" }), {
        status: 402,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    if (!upstream.ok) {
      const txt = await upstream.text();
      return new Response(JSON.stringify({ error: txt }), {
        status: upstream.status,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    return new Response(upstream.body, {
      headers: {
        ...corsHeaders,
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    });
  } catch (e) {
    return new Response(JSON.stringify({ error: String(e) }), {
      status: 500,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});
