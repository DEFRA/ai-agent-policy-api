"""
Search Agent for LangGraph Semantic Search Bot

This module contains the search node implementation that performs
semantic search using an external API.
"""

from typing import Any

from langchain_core.messages import HumanMessage


def search_node(state: dict[str, Any]):
    """
    LangGraph Node #1: Semantic Search

    LangGraph Node Characteristics:
    - Takes current state as input
    - Returns dictionary with state updates
    - Updates are automatically merged with existing state
    - State passing is handled by LangGraph framework

    This node extracts user question and performs external API search.
    """
    # Import dependencies at runtime to avoid circular imports
    from simple_langgraph_semantic_bot import DISPLAY_SEARCH_RESULTS, search_tool
    from utils import display_search_results

    # Extract the latest user message from conversation history
    # LangGraph automatically manages message list via add_messages
    user_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    # Perform external API call
    results = search_tool.search(user_message)

    # Display search results to terminal (if enabled)
    if DISPLAY_SEARCH_RESULTS:
        display_search_results(results, user_message)

    # Return state update - LangGraph merges this with existing state
    return {"search_results": results}
