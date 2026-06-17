import { useEffect, useState } from "react";
import { Moon, Sun, AlertCircle } from "lucide-react";
import { useSessionStore } from "@/store/sessionStore";
import { QueryPanel } from "@/components/QueryPanel";
import { AgentDAG } from "@/components/AgentDAG";
import { DraftViewer } from "@/components/DraftViewer";
import { EvidencePanel } from "@/components/EvidencePanel";
import { EventLog } from "@/components/EventLog";
import { HumanApprovalBanner } from "@/components/HumanApprovalBanner";

// ── Dark mode hook ────────────────────────────────────────────────────────────

function useDarkMode(): [boolean, () => void] {
  const [dark, setDark] = useState(() => {
    if (typeof window === "undefined") return false;
    return (
      localStorage.getItem("theme") === "dark" ||
      (!localStorage.getItem("theme") &&
        window.matchMedia("(prefers-color-scheme: dark)").matches)
    );
  });

  useEffect(() => {
    const root = document.documentElement;
    if (dark) {
      root.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      root.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
  }, [dark]);

  return [dark, () => setDark((d) => !d)];
}

// ── Right panel tabs ──────────────────────────────────────────────────────────

type RightTab = "evidence" | "events";

function RightPanel() {
  const [tab, setTab] = useState<RightTab>("evidence");

  return (
    <div className="w-80 shrink-0 flex flex-col border-l border-border">
      {/* Tab bar */}
      <div className="flex shrink-0 border-b border-border">
        {(["evidence", "events"] as RightTab[]).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`flex-1 py-2 text-xs font-medium capitalize transition-colors ${
              tab === t
                ? "text-foreground border-b-2 border-primary"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {/* Tab content — keep both mounted so Evidence list doesn't reset */}
      <div className={`flex-1 min-h-0 ${tab === "evidence" ? "flex flex-col" : "hidden"}`}>
        <EvidencePanel />
      </div>
      <div className={`flex-1 min-h-0 ${tab === "events" ? "flex flex-col" : "hidden"}`}>
        <EventLog />
      </div>
    </div>
  );
}

// ── App ───────────────────────────────────────────────────────────────────────

export default function App() {
  const { status, error } = useSessionStore();
  const [dark, toggleDark] = useDarkMode();

  return (
    <div className="flex flex-col h-screen bg-background text-foreground overflow-hidden">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-3 border-b border-border shrink-0">
        <div className="flex items-center gap-3">
          <span className="text-lg font-semibold tracking-tight">Market Intelligence Agent</span>
          <span className="text-xs px-2 py-0.5 rounded-full bg-muted text-muted-foreground font-mono">
            v0.7
          </span>
        </div>
        <div className="flex items-center gap-3">
          <StatusBadge status={status} />
          <button
            onClick={toggleDark}
            aria-label="Toggle dark mode"
            className="p-1.5 rounded-md hover:bg-muted transition-colors text-muted-foreground hover:text-foreground"
          >
            {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </button>
        </div>
      </header>

      {/* Error banner */}
      {status === "error" && error && (
        <div className="flex items-center gap-2 px-6 py-2 bg-red-500/10 border-b border-red-500/20 text-red-400 text-sm shrink-0">
          <AlertCircle className="h-4 w-4 shrink-0" />
          <span className="truncate">{error}</span>
        </div>
      )}

      {/* Human approval banner (shown inside main area) */}
      <HumanApprovalBanner />

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

        {/* Right: evidence / event log (tabbed) */}
        <RightPanel />
      </div>
    </div>
  );
}

// ── Status badge ──────────────────────────────────────────────────────────────

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    idle:       "bg-muted text-muted-foreground",
    connecting: "bg-yellow-500/20 text-yellow-400",
    running:    "bg-blue-500/20 text-blue-400 animate-pulse",
    done:       "bg-green-500/20 text-green-400",
    error:      "bg-red-500/20 text-red-400",
  };
  return (
    <span className={`text-xs px-2 py-0.5 rounded-full font-mono ${colors[status] ?? colors.idle}`}>
      {status}
    </span>
  );
}
