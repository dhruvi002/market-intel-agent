import { AlertTriangle } from "lucide-react";
import { useSessionStore } from "@/store/sessionStore";

export function HumanApprovalBanner() {
  const { humanApprovalRequired, sessionId } = useSessionStore();

  if (!humanApprovalRequired) return null;

  return (
    <div className="mx-4 my-2 flex items-start gap-3 rounded-lg border border-yellow-500/40 bg-yellow-500/10 px-4 py-3 shrink-0">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-yellow-400" />
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-yellow-300">Human approval required</p>
        <p className="text-xs text-yellow-400/80 mt-0.5">
          The agent has paused and is waiting for a human decision before
          continuing.{" "}
          {sessionId && (
            <span className="font-mono opacity-70">session: {sessionId.slice(0, 8)}…</span>
          )}
        </p>
      </div>
    </div>
  );
}
