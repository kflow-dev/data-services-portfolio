import { corsHeaders } from "../_shared/cors.ts";

const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return new Response(null, { headers: corsHeaders });
  try {
    const { domain, context, items, topK = 5 } = await req.json();
    if (!Array.isArray(items) || !items.length) {
      return new Response(JSON.stringify({ error: "items[] required" }), {
        status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
    const prompt =
      `Domain: ${domain ?? "generic"}\n` +
      `User context: ${context ?? ""}\n\n` +
      `Catalog items (JSON):\n${JSON.stringify(items)}\n\n` +
      `Rank the top ${topK} most relevant items for this user. ` +
      `Return strict JSON: {"ranked":[{"id":"...","score":0-1,"reason":"<one short sentence>"}]}`;

    const r = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
      method: "POST",
      headers: { Authorization: `Bearer ${LOVABLE_API_KEY}`, "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "google/gemini-2.5-flash",
        messages: [
          { role: "system", content: `You are an expert ${domain ?? ""} recommender. Output ONLY valid JSON.` },
          { role: "user", content: prompt },
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
    const content = data.choices?.[0]?.message?.content ?? "{}";
    return new Response(content, { headers: { ...corsHeaders, "Content-Type": "application/json" } });
  } catch (e) {
    return new Response(JSON.stringify({ error: String(e) }), {
      status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});