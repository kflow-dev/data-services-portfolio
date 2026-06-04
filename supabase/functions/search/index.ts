import { corsHeaders } from "../_shared/cors.ts";

const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");

type Mode = "declarative" | "socialtraces" | "assetmanager" | "localradar";

const SYSTEM: Record<Mode, string> = {
  declarative:
    "You are a multi-agent declarative search planner. Given a user query, produce a short research plan AND a synthesized answer using your own knowledge. Output ONLY valid JSON.",
  socialtraces:
    "You are SocialTraces, a social-media fuzzy-search detective. Given a query and a list of social profile snippets, find probable matches (handle variations, typos, transliteration, aliases). Output ONLY valid JSON.",
  assetmanager:
    "You are AssetManager, a multi-media article summarizer with NER. Summarize the input and extract named entities (PERSON, ORG, LOC, DATE, PRODUCT). Output ONLY valid JSON.",
  localradar:
    "You are MyLocalRadar, a location disambiguation engine. Given a query mentioning places, return candidate locations with country, region, and a confidence score. Output ONLY valid JSON.",
};

function userPrompt(mode: Mode, query: string, corpus?: string): string {
  switch (mode) {
    case "declarative":
      return `Query: ${query}\n\nReturn JSON: {"summary":"<one-paragraph answer>","items":[{"title":"<step or source>","detail":"<what was done / found>"}]}`;
    case "socialtraces":
      return `Query: ${query}\n\nProfile corpus:\n${corpus ?? ""}\n\nReturn JSON: {"summary":"<who you think this is>","items":[{"title":"<handle / name>","detail":"<why it matches, score 0-1>"}]}`;
    case "assetmanager":
      return `Article:\n${query}\n\nReturn JSON: {"summary":"<3-sentence summary>","items":[{"title":"<ENTITY_TYPE>","detail":"<entity text — short context>"}]}`;
    case "localradar":
      return `Query: ${query}\n\nReturn JSON: {"summary":"<best interpretation>","items":[{"title":"<place name>","detail":"<country, region — confidence 0-1>"}]}`;
  }
}

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return new Response(null, { headers: corsHeaders });
  try {
    const { mode, query, corpus } = (await req.json()) as { mode: Mode; query: string; corpus?: string };
    if (!mode || !query) {
      return new Response(JSON.stringify({ error: "mode and query required" }), {
        status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
    const r = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
      method: "POST",
      headers: { Authorization: `Bearer ${LOVABLE_API_KEY}`, "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "google/gemini-2.5-flash",
        messages: [
          { role: "system", content: SYSTEM[mode] },
          { role: "user", content: userPrompt(mode, query, corpus) },
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
    return new Response(data.choices?.[0]?.message?.content ?? "{}", {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (e) {
    return new Response(JSON.stringify({ error: String(e) }), {
      status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});