import { useSessionStore } from "@/store/sessionStore";
import { QueryPanel } from "@/components/QueryPanel";
import { AgentDAG } from "@/components/AgentDAG";
import { DraftViewer } from "@/components/DraftViewer";
import { EvidencePanel } from "@/components/EvidencePanel";

export default function App() {
  const status = useSessionStore((s) => s.status);

  return (
    <div className="flex flex-col h-screen bg-background text-foreground overflow-hidden">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-3 border-b border-border shrink-0">
        <div className="flex items-center gap-3">
          <span className="text-lg font-semibold tracking-tight">Market Intelligence Agent</span>
          <span className="text-xs px-2 py-0.5 rounded-full bg-muted text-muted-foreground font-mono">
            v0.1
          </span>
        </div>
        <StatusBadge status={status} />
      </header>

      {/* Main layout */}
      <div className="flex flex-1 min-h-0">
        {/* Left: query + agent DAG */}
        <div className="flex flex-col w-80 shrink-0 border-r border-border">
          <QueryPanel />
          <div className="flex-1 min-h-0">
            <AgentDAG />
          </div>
        </div>

        {/* Center: streamed draft */}
        <div className="flex-1 min-w-0 border-r border-border">
          <DraftViewer />
        </div>

        {/* Right: evidence / citations */}
        <div className="w-72 shrink-0">
          <EvidencePanel />
        </div>
      </div>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    idle: "bg-muted text-muted-foreground",
    connecting: "bg-yellow-500/20 text-yellow-400",
    running: "bg-blue-500/20 text-blue-400 animate-pulse",
    done: "bg-green-500/20 text-green-400",
    error: "bg-red-500/20 text-red-400",
  };
  return (
    <span className={`text-xs px-2 py-0.5 rounded-full font-mono ${colors[status] ?? colors.idle}`}>
      {status}
    </span>
  );
}
