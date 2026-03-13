# lab4/graph.py
"""LangGraph wiring for planner, researcher, tools, and answerer nodes."""

import os

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

from config import (
    PLANNER_MODEL,
    RESEARCHER_MODEL,
    ANSWERER_MODEL,
    MAILER_MODEL,
    NUM_CTX,
    PLANNER_NUM_CTX,
    RESEARCHER_NUM_CTX,
    ANSWERER_NUM_CTX,
    MAILER_NUM_CTX,
    NUM_THREAD,
    NUM_PREDICT,
    KEEP_ALIVE,
    DISABLE_STREAMING,
    TEMPERATURE,
    TOP_P,
    TOP_K,
    REPEAT_PENALTY,
    PLANNER_REASONING,
    RESEARCHER_REASONING,
    ANSWERER_REASONING,
    MAILER_REASONING,
    GROUNDEDNESS_MODEL,
    GROUNDEDNESS_NUM_CTX,
    GROUNDEDNESS_TEMPERATURE,
)
from state import AgentState
from planner_agent import create_supervisor
from researcher_agent import create_researcher
from answer_agent import create_answerer
from mailer_agent import create_mailer


def get_llm(model_name: str, reasoning: bool = False, num_ctx: int | None = None):
    """Create a ChatOllama model tuned for laptop-friendly latency."""
    kwargs = {
        "model": model_name,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "repeat_penalty": REPEAT_PENALTY,
        "num_ctx": num_ctx if num_ctx is not None else NUM_CTX,
        "num_thread": NUM_THREAD,
        "num_predict": NUM_PREDICT,
        "keep_alive": KEEP_ALIVE,
        "disable_streaming": DISABLE_STREAMING,
        "reasoning": reasoning,
    }
    # Keep provider defaults for unspecified knobs.
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return ChatOllama(**kwargs)


def build_app(
    planner_model: str = PLANNER_MODEL,
    researcher_model: str = RESEARCHER_MODEL,
    answerer_model: str = ANSWERER_MODEL,
    mailer_model: str = MAILER_MODEL,
):
    """Build the LangGraph app with explicit role-to-model mapping."""
    planner_llm = get_llm(planner_model, reasoning=PLANNER_REASONING, num_ctx=PLANNER_NUM_CTX)
    researcher_llm = get_llm(researcher_model, reasoning=RESEARCHER_REASONING, num_ctx=RESEARCHER_NUM_CTX)
    answerer_llm = get_llm(answerer_model, reasoning=ANSWERER_REASONING, num_ctx=ANSWERER_NUM_CTX)
    mailer_llm = get_llm(mailer_model, reasoning=MAILER_REASONING, num_ctx=MAILER_NUM_CTX)

    # Groundedness judge: small context window, low temperature, reuses VRAM via keep_alive.
    # When GROUNDEDNESS_MODEL is empty, reuse the answerer model to avoid VRAM swap.
    judge_model = GROUNDEDNESS_MODEL if GROUNDEDNESS_MODEL else answerer_model
    judge_is_shared = (judge_model == answerer_model)
    groundedness_llm = get_llm(
        judge_model,
        num_ctx=GROUNDEDNESS_NUM_CTX if not judge_is_shared else min(GROUNDEDNESS_NUM_CTX, ANSWERER_NUM_CTX),
    )
    # Override temperature for deterministic judging.
    groundedness_llm.temperature = GROUNDEDNESS_TEMPERATURE
    # Prevent judge tokens from leaking into the answerer's stream.
    groundedness_llm.disable_streaming = True

    supervisor = create_supervisor(planner_llm)
    researcher, research_tools = create_researcher(researcher_llm)
    # `create_answerer/create_mailer` already return node callables.
    answerer_node = create_answerer(answerer_llm)
    mailer_node = create_mailer(mailer_llm)

    # Keep topology compact: a single supervisor decides each transition.
    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("researcher", researcher)
    workflow.add_node("research_tools", research_tools)
    workflow.add_node("answerer", answerer_node)
    workflow.add_node("mailer", mailer_node)

    # Tools always return control to supervisor for the next decision.
    workflow.add_edge("research_tools", "supervisor")

    # Researcher emits tool calls; tool node executes them.
    workflow.add_edge("researcher", "research_tools")

    # Answerer and mailer both terminate the run with a user-facing message.
    workflow.add_edge("answerer", END)
    workflow.add_edge("mailer", END)

    # Supervisor decides where to go next from its `state["next"]` value.
    def supervisor_router(state):
        """Read supervisor decision from state and return next node id."""
        decision = state["next"]
        if os.getenv("ROUTER_DEBUG", "0") == "1":
            print(f"Supervisor decided: '{decision}'")
        return decision

    workflow.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "researcher": "researcher",
            "answerer": "answerer",
            "mailer": "mailer",
            "FINISH": END,
        },
    )

    # Every run starts at supervisor.
    workflow.set_entry_point("supervisor")
    app = workflow.compile()

    if os.getenv("GRAPH_DEBUG", "0") == "1":
        print("\nGRAPH STRUCTURE:")
        try:
            print(app.get_graph().draw_ascii())
        except ImportError:
            print("(Graph visualization skipped; install 'grandalf' to enable.)")

    return app, groundedness_llm
