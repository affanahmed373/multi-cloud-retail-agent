"""
LangGraph agent graph with LangChain guardrail nodes.

Flow:
  input_guardrails -> (blocked?) -> END
                   -> agent -> output_guardrails -> END
"""

from typing import TypedDict, List, Dict, Any, NotRequired, Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

from .agent import RetailAgent
from .langchain_guardrails import get_guardrail_pipeline


class AgentState(TypedDict):
    query: str
    answer: str
    context_chunks: List[Dict[str, Any]]
    tool_info: str
    guardrail: NotRequired[Dict[str, Any]]
    blocked: NotRequired[bool]


def input_guardrails_node(state: AgentState) -> AgentState:
    """LangChain scope + PII middleware on user input."""
    pipeline = get_guardrail_pipeline()
    outcome = pipeline.process_input(state["query"])

    if not outcome.allowed:
        guardrail = outcome.as_dict()
        return {
            **state,
            "answer": outcome.message or "Request blocked by guardrails.",
            "context_chunks": [],
            "tool_info": "",
            "blocked": True,
            "guardrail": guardrail,
        }

    guardrail: Dict[str, Any] = outcome.as_dict()
    if outcome.pii_detected:
        guardrail["pii_redacted_input"] = True

    return {
        **state,
        "query": outcome.text,
        "blocked": False,
        "guardrail": guardrail,
    }


def run_agent_node(state: AgentState, agent: RetailAgent) -> AgentState:
    """Core RAG agent (no guardrails here — handled by graph nodes)."""
    result = agent.handle_query(state["query"])
    return {
        **state,
        "answer": result["answer"],
        "context_chunks": result["context_chunks"],
        "tool_info": result["tool_info"],
    }


def output_guardrails_node(state: AgentState) -> AgentState:
    """LangChain PII middleware on model output."""
    pipeline = get_guardrail_pipeline()
    outcome = pipeline.process_output(state["answer"])

    guardrail = dict(state.get("guardrail") or {"blocked": False})
    guardrail.update(outcome.as_dict())

    if not outcome.allowed:
        return {
            **state,
            "answer": outcome.message or "Request blocked by guardrails.",
            "guardrail": guardrail,
        }

    if outcome.pii_detected:
        guardrail["pii_redacted_output"] = True

    return {
        **state,
        "answer": outcome.text,
        "guardrail": guardrail,
    }


def route_after_input(state: AgentState) -> Literal["agent", "end"]:
    if state.get("blocked"):
        return "end"
    return "agent"


def build_graph(agent: RetailAgent):
    graph = StateGraph(AgentState)

    graph.add_node("input_guardrails", input_guardrails_node)

    def agent_node(state: AgentState):
        return run_agent_node(state, agent)

    graph.add_node("agent", agent_node)
    graph.add_node("output_guardrails", output_guardrails_node)

    graph.set_entry_point("input_guardrails")
    graph.add_conditional_edges(
        "input_guardrails",
        route_after_input,
        {"agent": "agent", "end": END},
    )
    graph.add_edge("agent", "output_guardrails")
    graph.add_edge("output_guardrails", END)

    checkpointer = InMemorySaver()
    return graph.compile(checkpointer=checkpointer)
