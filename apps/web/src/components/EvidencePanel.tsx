import { useSessionStore } from "@/store/sessionStore";
import type { Evidence } from "@/types/agent";

export function EvidencePanel() {
  const evidence = useSessionStore((s) => s.evidence);

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-2 border-b border-border shrink-0">
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
          Evidence ({evidence.length})
        </span>
      </div>
      <div className="flex-1 overflow-y-auto divide-y divide-border">
        {evidence.length === 0 ? (
          <p className="px-4 py-6 text-xs text-muted-foreground">No evidence gathered yet.</p>
        ) : (
          evidence.map((e) => <EvidenceCard key={e.id} evidence={e} />)
        )}
      </div>
    </div>
  );
}

function EvidenceCard({ evidence: e }: { evidence: Evidence }) {
  const sourceLabel =
    e.filing_type && e.ticker
      ? `${e.ticker} ${e.filing_type}`
      : e.source_type === "web"
        ? "Web"
        : e.source_type;

  return (
    <div className="px-4 py-3 hover:bg-muted/40 transition-colors group">
      <div className="flex items-center gap-1.5 mb-1">
        <span className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-mono">
          {sourceLabel}
        </span>
        {e.section && (
          <span className="text-[10px] text-muted-foreground">{e.section}</span>
        )}
        {e.relevance_score != null && (
          <span className="ml-auto text-[10px] text-muted-foreground">
            {(e.relevance_score * 100).toFixed(0)}%
          </span>
        )}
      </div>
      <p className="text-xs text-foreground/80 line-clamp-3 leading-relaxed">{e.text}</p>
      {e.source_url && (
        <a
          href={e.source_url}
          target="_blank"
          rel="noopener noreferrer"
          className="mt-1 text-[10px] text-primary hover:underline truncate block opacity-0 group-hover:opacity-100 transition-opacity"
        >
          {e.source_url}
        </a>
      )}
    </div>
  );
}
