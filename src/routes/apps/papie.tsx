import { createFileRoute, Link } from "@tanstack/react-router";
import { useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { SiteHeader } from "@/components/SiteHeader";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Loader2, Send } from "lucide-react";

export const Route = createFileRoute("/apps/papie")({
  head: () => ({ meta: [{ title: "PAPIE — Personal AI assistant" }] }),
  component: Papie,
});

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL as string;
const anonKey = import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY as string;

type Msg = { role: "user" | "assistant"; content: string };

function Papie() {
  const [messages, setMessages] = useState<Msg[]>([
    { role: "assistant", content: "Hi, I'm **PAPIE** — your personal information assistant. How can I help today?" },
  ]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  async function send() {
    if (!input.trim() || streaming) return;
    const next: Msg[] = [...messages, { role: "user", content: input.trim() }];
    setMessages(next);
    setInput("");
    setStreaming(true);
    const ac = new AbortController();
    abortRef.current = ac;

    try {
      const r = await fetch(`${supabaseUrl}/functions/v1/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${anonKey}`, apikey: anonKey },
        body: JSON.stringify({ persona: "papie", messages: next }),
        signal: ac.signal,
      });
      if (!r.ok || !r.body) {
        const t = await r.text();
        setMessages((m) => [...m, { role: "assistant", content: `_Error: ${t}_` }]);
        return;
      }
      const reader = r.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      let acc = "";
      setMessages((m) => [...m, { role: "assistant", content: "" }]);
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.startsWith("data:")) continue;
          const payload = line.slice(5).trim();
          if (payload === "[DONE]") continue;
          try {
            const j = JSON.parse(payload);
            const delta = j.choices?.[0]?.delta?.content;
            if (delta) {
              acc += delta;
              setMessages((m) => {
                const copy = m.slice();
                copy[copy.length - 1] = { role: "assistant", content: acc };
                return copy;
              });
            }
          } catch {
            // skip
          }
        }
      }
    } catch (e) {
      setMessages((m) => [...m, { role: "assistant", content: `_Error: ${String(e)}_` }]);
    } finally {
      setStreaming(false);
    }
  }

  return (
    <div className="min-h-screen bg-background">
      <SiteHeader />
      <main className="mx-auto flex max-w-3xl flex-col px-4 py-6" style={{ minHeight: "calc(100vh - 60px)" }}>
        <Link to="/" className="text-sm text-muted-foreground hover:text-foreground">← Back to dashboard</Link>
        <h1 className="mt-2 text-2xl font-bold">PAPIE</h1>
        <p className="text-sm text-muted-foreground">Personal Assistant for Personal Information Exchange.</p>

        <Card className="mt-4 flex-1 overflow-y-auto p-4">
          <div className="space-y-4">
            {messages.map((m, i) => (
              <div key={i} className={m.role === "user" ? "text-right" : ""}>
                <div className={`inline-block max-w-[85%] rounded-lg px-3 py-2 text-sm ${
                  m.role === "user" ? "bg-primary text-primary-foreground" : "bg-muted"
                }`}>
                  <div className="prose prose-sm dark:prose-invert max-w-none [&_p]:my-1">
                    <ReactMarkdown>{m.content || "…"}</ReactMarkdown>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Card>

        <form
          onSubmit={(e) => { e.preventDefault(); send(); }}
          className="mt-3 flex gap-2"
        >
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask PAPIE anything…"
            disabled={streaming}
          />
          <Button type="submit" disabled={streaming || !input.trim()}>
            {streaming ? <Loader2 className="size-4 animate-spin" /> : <Send className="size-4" />}
          </Button>
        </form>
      </main>
    </div>
  );
}
