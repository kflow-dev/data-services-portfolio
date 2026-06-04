import { createClient } from "https://esm.sh/@supabase/supabase-js@2.45.0";
import { corsHeaders } from "../_shared/cors.ts";

const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY")!;
const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;

function chunk(text: string, size = 800, overlap = 100): string[] {
  const out: string[] = [];
  let i = 0;
  while (i < text.length) {
    out.push(text.slice(i, i + size));
    i += size - overlap;
  }
  return out;
}

async function embed(inputs: string[]): Promise<number[][]> {
  const r = await fetch("https://ai.gateway.lovable.dev/v1/embeddings", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${LOVABLE_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "openai/text-embedding-3-small",
      input: inputs,
    }),
  });
  if (!r.ok) throw new Error(`Embed failed: ${await r.text()}`);
  const data = await r.json();
  return data.data.map((d: { embedding: number[] }) => d.embedding);
}

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return new Response(null, { headers: corsHeaders });
  try {
    const { source, content } = await req.json();
    if (!content || typeof content !== "string")
      return new Response(JSON.stringify({ error: "content required" }), {
        status: 400,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });

    const chunks = chunk(content);
    const embeddings = await embed(chunks);
    const sb = createClient(SUPABASE_URL, SERVICE_KEY);
    const rows = chunks.map((c, i) => ({
      source: source || "untitled",
      content: c,
      embedding: embeddings[i],
    }));
    const { error } = await sb.from("rag_documents").insert(rows);
    if (error) throw error;

    return new Response(JSON.stringify({ inserted: rows.length }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (e) {
    return new Response(JSON.stringify({ error: String(e) }), {
      status: 500,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});