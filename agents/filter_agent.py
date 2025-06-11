"""
Filter Agent for LangGraph Semantic Search Bot

This module contains the filter node implementation that uses an LLM
to assess the relevance of search results beyond semantic similarity.
"""

from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


def filter_node(state: dict[str, Any]):
    """
    LangGraph Node #2: Relevance Assessment Filter

    This node demonstrates:
    - State access: Reading search results and user question
    - LLM-based filtering: Using AI to assess relevance beyond semantic similarity
    - State updates: Creating filtered_results for downstream processing
    - Quality control: Ensuring only relevant results proceed to response generation

    The node takes raw search results and filters them for actual relevance
    to the user's question, addressing the problem of high semantic similarity
    but low contextual relevance.
    """
    # Import dependencies at runtime to avoid circular imports
#from simple_langgraph_semantic_bot import DISPLAY_SEARCH_RESULTS, llm
    from simple_langgraph_semantic_bot import DISPLAY_SEARCH_RESULTS

    llm = ChatOpenAI(model="o4-mini")

    # Access search results from state (populated by search_node)
    search_results = state.get("search_results", [])

    # Extract user question from message history
    user_question = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_question = msg.content
            break

    # If no search results, return empty filtered results
    if not search_results:
        return {"filtered_results": []}

    # Load filter prompt template
    with open("prompts/filter_prompt.md", encoding="utf-8") as file:
        filter_prompt_template = file.read().strip()

    # Prepare search results for evaluation
    search_context = ""
    for i, result in enumerate(search_results, 1):
        if isinstance(result, dict):
            search_context = create_search_context(result, i, search_context)
            """
            pq_id = result.get("id", f"ID-{i}")
            ask_date = result.get("ask_date", result.get("source", ""))
            question = result.get("question", "")
            answer = result.get("answer", "")
            score = result.get("score", result.get("similarity", ""))

            # Format date for display
            date_display = f" - {ask_date}" if ask_date else ""

            search_context += f"\nResult {i} (PQ {pq_id}{date_display}):\n"
            if question:
                search_context += f"Question: {question}\n"
            if answer:
                search_context += f"Answer: {answer}\n"
            if score:
                search_context += f"Similarity Score: {score}\n"
            """
    # Build filtering prompt
    filter_prompt = f"""{filter_prompt_template}

**User Question:** {user_question}

**Search Results to Evaluate:**
{search_context}

**Your Assessment:**"""

    # Get relevance assessment from LLM
    filter_response = llm.invoke([HumanMessage(content=filter_prompt)])
    assessment = filter_response.content

    # Simplified parsing - look for RELEVANT/IRRELEVANT keywords
    filtered_results = []
    lines = assessment.split("\n")


    # Extract all RELEVANT/IRRELEVANT decisions in order
    result_decisions = assemble_result_decisions(lines)

    # Apply decisions to results in order
    for i, decision in enumerate(result_decisions):
        if decision == "RELEVANT" and i < len(search_results):
            filtered_results.append(search_results[i])

    # Limit to top 5 results even after filtering
    filtered_results = filtered_results[:5]

    # Display filtering results if enabled
    if DISPLAY_SEARCH_RESULTS:
        display_search_results(search_results, filtered_results)
        """
        print("\n🔽 RELEVANCE FILTERING RESULTS:")
        print("=" * 60)
        print(f"Original results: {len(search_results)}")
        print(f"Filtered to: {len(filtered_results)} relevant results")
        print("=" * 60)
        """
    # Return state update with filtered results
    return {"filtered_results": filtered_results}


def assemble_result_decisions(lines):
    result_decisions = []
    for line in lines:
        line = line.strip().upper()
        if "RELEVANT" in line and "IRRELEVANT" not in line:
            result_decisions.append("RELEVANT")
        elif "IRRELEVANT" in line:
            result_decisions.append("IRRELEVANT")
    return result_decisions


def create_search_context(result, i, search_context):
    pq_id = result.get("id", f"ID-{i}")
    ask_date = result.get("ask_date", result.get("source", ""))
    question = result.get("question", "")
    answer = result.get("answer", "")
    score = result.get("score", result.get("similarity", ""))

    # Format date for display
    date_display = f" - {ask_date}" if ask_date else ""

    search_context += f"\nResult {i} (PQ {pq_id}{date_display}):\n"
    if question:
        search_context += f"Question: {question}\n"
    if answer:
        search_context += f"Answer: {answer}\n"
    if score:
        search_context += f"Similarity Score: {score}\n"
    return search_context


def display_search_results(search_results, filtered_results):
    print("\n🔽 RELEVANCE FILTERING RESULTS:")
    print("=" * 60)
    print(f"Original results: {len(search_results)}")
    print(f"Filtered to: {len(filtered_results)} relevant results")
    print("=" * 60)
