import { createClient } from "https://esm.sh/@supabase/supabase-js@2.45.0";
import { corsHeaders } from "../_shared/cors.ts";

const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY")!;
const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;

async function embed(input: string): Promise<number[]> {
  const r = await fetch("https://ai.gateway.lovable.dev/v1/embeddings", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${LOVABLE_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "openai/text-embedding-3-small",
      input,
    }),
  });
  if (!r.ok) throw new Error(`Embed failed: ${await r.text()}`);
  const data = await r.json();
  return data.data[0].embedding;
}

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return new Response(null, { headers: corsHeaders });
  try {
    const { question, k = 5 } = await req.json();
    const qEmb = await embed(question);
    const sb = createClient(SUPABASE_URL, SERVICE_KEY);
    const { data: matches, error } = await sb.rpc("match_rag", {
      query_embedding: qEmb,
      match_count: k,
    });
    if (error) throw error;

    const context = (matches || [])
      .map((m: { source: string; content: string }, i: number) => `[${i + 1}] (${m.source})\n${m.content}`)
      .join("\n\n");

    const r = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${LOVABLE_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "google/gemini-2.5-flash",
        messages: [
          {
            role: "system",
            content:
              "You are a Multi-media RAG assistant. Answer ONLY from the provided context. " +
              "Cite sources inline as [1], [2], etc. If the answer is not in the context, say so.",
          },
          { role: "user", content: `Context:\n${context}\n\nQuestion: ${question}` },
        ],
      }),
    });
    if (!r.ok) {
      return new Response(JSON.stringify({ error: await r.text() }), {
        status: r.status,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
    const data = await r.json();
    const answer = data.choices?.[0]?.message?.content ?? "";
    return new Response(JSON.stringify({ answer, matches }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (e) {
    return new Response(JSON.stringify({ error: String(e) }), {
      status: 500,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});
