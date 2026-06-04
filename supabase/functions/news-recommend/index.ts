import { corsHeaders } from "../_shared/cors.ts";

const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return new Response(null, { headers: corsHeaders });
  try {
    const { interests, articles } = await req.json();
    const prompt =
      `User interests: ${interests}\n\n` +
      `Available articles (JSON):\n${JSON.stringify(articles)}\n\n` +
      `Rank the top 5 most relevant articles for this user. Return strict JSON: ` +
      `{"ranked":[{"id":"...","score":0-1,"reason":"<one short sentence>"}]}`;

    const r = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${LOVABLE_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "google/gemini-2.5-flash",
        messages: [
          { role: "system", content: "You are a personalized news recommender. Output ONLY valid JSON." },
          { role: "user", content: prompt },
        ],
        response_format: { type: "json_object" },
      }),
    });
    if (!r.ok) {
      return new Response(JSON.stringify({ error: await r.text() }), {
        status: r.status,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
    const data = await r.json();
    const content = data.choices?.[0]?.message?.content ?? "{}";
    const parsed = JSON.parse(content);
    return new Response(JSON.stringify(parsed), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (e) {
    return new Response(JSON.stringify({ error: String(e) }), {
      status: 500,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});