import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useSessionStore } from "@/store/sessionStore";

export function DraftViewer() {
  const { draft, status, critique } = useSessionStore();

  if (!draft && status === "idle") {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
        Submit a query to begin analysis.
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-4 py-2 border-b border-border shrink-0">
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
          Report Draft
        </span>
        {critique && (
          <span
            className={`text-xs px-2 py-0.5 rounded-full font-mono ${
              critique.verdict === "pass"
                ? "bg-green-500/20 text-green-400"
                : critique.verdict === "revise"
                  ? "bg-yellow-500/20 text-yellow-400"
                  : "bg-red-500/20 text-red-400"
            }`}
          >
            critic: {critique.verdict}
          </span>
        )}
      </div>

      <div className="flex-1 overflow-y-auto p-6">
        {draft ? (
          <article className="prose prose-sm dark:prose-invert max-w-none">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{draft}</ReactMarkdown>
            {status === "running" && (
              <span className="inline-block w-2 h-4 bg-primary animate-pulse ml-0.5 align-text-bottom" />
            )}
          </article>
        ) : (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <span className="inline-block w-2 h-2 rounded-full bg-blue-400 animate-pulse" />
            Agents working…
          </div>
        )}
      </div>

      {critique?.failing_claims.length ? (
        <div className="px-4 py-3 border-t border-border bg-yellow-500/5 shrink-0">
          <p className="text-xs font-medium text-yellow-400 mb-1">Critic flagged claims</p>
          {critique.failing_claims.map((c, i) => (
            <p key={i} className="text-xs text-muted-foreground">
              • {c.claim} — {c.reason}
            </p>
          ))}
        </div>
      ) : null}
    </div>
  );
}
