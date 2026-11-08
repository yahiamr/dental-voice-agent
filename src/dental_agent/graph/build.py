"""Wires the graph together. Stage: 1.

What this file is:
    Connects the nodes and edges into a working LangGraph graph.

What it contains:
    - build_graph(deps: GraphDeps), which adds nodes and edges and compiles with the
      checkpointer.
    - graph_diagram: LangGraph's own Mermaid export, for checking the wiring.

How to understand it:
    Read this for the shape of a turn in one place. No node has a retry policy, so a commit can
    never run twice by accident. The README diagrams are drawn by hand in assets/diagrams/, so
    after changing the wiring, compare graph_diagram's output with turn.mmd.

Uses:
    graph/nodes.py, graph/routing.py, state.py, deps.py

Used by:
    session.py
"""

from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from dental_agent.deps import GraphDeps
from dental_agent.graph import nodes, routing
from dental_agent.state import CallState

# Every node, in the order a turn meets them.
NODES = {
    "start_turn": nodes.start_turn,
    "check_emergency": nodes.check_emergency,
    "check_confirmation": nodes.check_confirmation,
    "await_emergency": nodes.await_emergency,
    "write_to_calendar": nodes.write_to_calendar,
    "call_model": nodes.call_model,
    "run_tools": nodes.run_tools,
    "ask_confirmation": nodes.ask_confirmation,
    "check_model_reply": nodes.check_model_reply,
    "use_fixed_line": nodes.use_fixed_line,
    "end_turn": nodes.end_turn,
}


def build_graph(deps: GraphDeps) -> Any:
    """Wire the nodes and edges, and compile the graph with the checkpointer.

    No node has a retry policy, so a commit can never run twice by accident.
    """
    graph = StateGraph(CallState)
    for name, node in NODES.items():
        # partial gives each node its dependencies, so nodes stay plain functions.
        graph.add_node(name, partial(node, deps=deps))

    # The fixed path into every turn.
    graph.add_edge(START, "start_turn")
    graph.add_edge("start_turn", "check_emergency")

    # The branches. The lists name each possible target, for the diagram.
    graph.add_conditional_edges(
        "check_emergency", routing.after_check_emergency, ["use_fixed_line", "check_confirmation"]
    )
    graph.add_conditional_edges(
        "check_confirmation", routing.after_check_confirmation, ["await_emergency", "call_model"]
    )
    # The emergency check may still be running, so every model call and every write waits here.
    graph.add_edge("call_model", "await_emergency")
    graph.add_conditional_edges(
        "await_emergency",
        routing.after_await_emergency,
        ["use_fixed_line", "write_to_calendar", "run_tools", "check_model_reply"],
    )
    graph.add_conditional_edges(
        "write_to_calendar", routing.after_write_to_calendar, ["use_fixed_line", "end_turn"]
    )
    graph.add_conditional_edges(
        "run_tools",
        routing.after_run_tools,
        ["call_model", "ask_confirmation", "use_fixed_line", "end_turn"],
    )

    # Every other node leads to the end of the turn.
    graph.add_edge("ask_confirmation", "end_turn")
    graph.add_edge("check_model_reply", "end_turn")
    graph.add_edge("use_fixed_line", "end_turn")
    graph.add_edge("end_turn", END)
    return graph.compile(checkpointer=deps.checkpointer)


def graph_diagram(deps: GraphDeps) -> str:
    """Return LangGraph's Mermaid diagram of the real graph, to check turn.mmd against."""
    return build_graph(deps).get_graph().draw_mermaid()
