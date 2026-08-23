from typing import TypedDict, List, Dict, Any, NotRequired
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

from .agent import RetailAgent


class AgentState(TypedDict):
    query: str
    answer: str
    context_chunks: List[Dict[str, Any]]
    tool_info: str
    guardrail: NotRequired[Dict[str, Any]]


def run_agent_node(state: AgentState, agent: RetailAgent) -> AgentState:
    result = agent.handle_query(state["query"])
    return {
        **state,
        "answer": result["answer"],
        "context_chunks": result["context_chunks"],
        "tool_info": result["tool_info"],
        "guardrail": result.get("guardrail") or {"blocked": False},
    }


def build_graph(agent: RetailAgent):
    graph = StateGraph(AgentState)

    # Single main node that wraps your existing pipeline
    def agent_node(state: AgentState):
        return run_agent_node(state, agent)

    graph.add_node("agent", agent_node)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)

    checkpointer = InMemorySaver()
    return graph.compile(checkpointer=checkpointer)
