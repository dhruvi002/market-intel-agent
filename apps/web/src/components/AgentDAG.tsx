import ReactFlow, {
  type Node,
  type Edge,
  Background,
  BackgroundVariant,
  Handle,
  Position,
} from "reactflow";
import "reactflow/dist/style.css";
import { useSessionStore } from "@/store/sessionStore";
import type { AgentName, AgentNodeData, AgentNodeStatus } from "@/types/agent";

const STATUS_COLORS: Record<AgentNodeStatus, string> = {
  idle:   "border-[hsl(var(--agent-idle))] text-[hsl(var(--agent-idle))]",
  active: "border-[hsl(var(--agent-active))] text-[hsl(var(--agent-active))] shadow-[0_0_12px_hsl(var(--agent-active)/0.5)]",
  done:   "border-[hsl(var(--agent-done))] text-[hsl(var(--agent-done))]",
  error:  "border-[hsl(var(--agent-error))] text-[hsl(var(--agent-error))]",
};

function AgentNode({ data }: { data: AgentNodeData }) {
  return (
    <div
      className={`px-3 py-2 rounded-lg border-2 bg-card text-xs font-medium transition-all duration-300 ${STATUS_COLORS[data.status]}`}
    >
      <Handle type="target" position={Position.Top} className="opacity-0" />
      <div>{data.label}</div>
      {data.status === "active" && (
        <div className="mt-1 text-[10px] opacity-70 animate-pulse">running…</div>
      )}
      <Handle type="source" position={Position.Bottom} className="opacity-0" />
    </div>
  );
}

const nodeTypes = { agent: AgentNode };

const AGENT_LABELS: Record<AgentName, string> = {
  supervisor:    "Supervisor",
  web_search:    "Web Search",
  edgar_parser:  "EDGAR Parser",
  retrieval:     "RAG Retrieval",
  sql_generator: "SQL Generator",
  summarizer:    "Summarizer",
  critic:        "Critic",
};

// Fixed DAG layout positions
const NODE_POSITIONS: Record<AgentName, { x: number; y: number }> = {
  supervisor:    { x: 90,  y: 0   },
  web_search:    { x: 0,   y: 90  },
  edgar_parser:  { x: 120, y: 90  },
  retrieval:     { x: 0,   y: 180 },
  sql_generator: { x: 120, y: 180 },
  summarizer:    { x: 60,  y: 270 },
  critic:        { x: 60,  y: 360 },
};

// Static edge definitions — animated flag and color are derived at render time
// from agentStatuses so the DAG is live.
const STATIC_EDGES: { id: string; source: AgentName; target: AgentName; dashed?: boolean }[] = [
  { id: "s-ws",  source: "supervisor",    target: "web_search"    },
  { id: "s-ep",  source: "supervisor",    target: "edgar_parser"  },
  { id: "ws-r",  source: "web_search",    target: "retrieval"     },
  { id: "ep-sq", source: "edgar_parser",  target: "sql_generator" },
  { id: "r-sum", source: "retrieval",     target: "summarizer"    },
  { id: "sq-sum",source: "sql_generator", target: "summarizer"    },
  { id: "sum-c", source: "summarizer",    target: "critic"        },
  { id: "c-s",   source: "critic",        target: "supervisor", dashed: true },
];

export function AgentDAG() {
  const agentStatuses = useSessionStore((s) => s.agentStatuses);

  const nodes: Node<AgentNodeData>[] = (Object.keys(AGENT_LABELS) as AgentName[]).map((agent) => ({
    id: agent,
    type: "agent",
    position: NODE_POSITIONS[agent],
    data: {
      label: AGENT_LABELS[agent],
      agent,
      status: agentStatuses[agent],
    },
    draggable: false,
  }));

  // Derive live edge styles from current agent statuses:
  // an edge lights up (animated + orange) when its source or target is active.
  const edges: Edge[] = STATIC_EDGES.map(({ id, source, target, dashed }) => {
    const lit =
      agentStatuses[source] === "active" || agentStatuses[target] === "active";
    return {
      id,
      source,
      target,
      animated: lit,
      style: {
        stroke: lit ? "hsl(var(--agent-active))" : "hsl(var(--border))",
        strokeWidth: lit ? 2 : 1,
        strokeDasharray: dashed ? "4 2" : undefined,
        transition: "stroke 0.3s, stroke-width 0.3s",
      },
    };
  });

  return (
    <div className="h-full w-full">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        proOptions={{ hideAttribution: true }}
        panOnDrag={false}
        zoomOnScroll={false}
        zoomOnPinch={false}
        nodesDraggable={false}
      >
        <Background variant={BackgroundVariant.Dots} gap={16} size={1} color="hsl(var(--border))" />
      </ReactFlow>
    </div>
  );
}
