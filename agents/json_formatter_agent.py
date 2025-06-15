"""
JSON Formatter Agent for LangGraph Semantic Search Bot

This module contains the JSON formatter node implementation that creates
structured JSON output for API consumption.
"""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI


def json_formatter_node(state: dict[str, Any]):
    """
    LangGraph Node #5: JSON Output Formatter

    This node takes the AI response, filtered search results, and review assessment
    to create a structured JSON output for API consumption. Uses a separate LLM call
    with specialized formatting prompt.
    """
    # Import dependencies at runtime to avoid circular imports
#    from simple_langgraph_semantic_bot import llm

    llm = ChatOpenAI(model="o4-mini")

    # Extract data from state - use filtered_results instead of search_results
    filtered_results = state.get("filtered_results", [])

    # Get the user's original question
    user_question = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_question = msg.content
            break

    # Get the AI response
    ai_response = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            ai_response = msg.content
            break

    # Get review information
    review_assessment = state.get(
        "review_assessment", "No review assessment available")
    review_passed = state.get("review_passed", False)
    review_attempts = state.get("review_attempts", 0)

    # Load JSON formatting prompt template
    with open("prompts/json_formatter_prompt.md", encoding="utf-8") as file:
        prompt_template = file.read().strip()

    # Prepare search results data for formatting
    search_data = f"Search Results ({len(filtered_results)} total):\n"
    for i, result in enumerate(filtered_results, 1):
        if isinstance(result, dict):
            search_data = create_search_data(result, i, search_data)
            """
            pq_id = result.get("id", f"ID-{i}")
            ask_date = result.get("ask_date", result.get("source", ""))
            question = result.get("question", "")
            answer = result.get("answer", "")
            score = result.get("score", "")
            source = result.get("source", "")

            # Format date for display
            date_display = f" - {ask_date}" if ask_date else ""

            search_data += f"\nResult {i} (PQ {pq_id}{date_display}):\n"
            if question:
                search_data += f"  Question: {question}\n"
            if answer:
                search_data += f"  Answer: {answer}\n"
            if score:
                search_data += f"  Score: {score}\n"
            if source:
                search_data += f"  Source: {source}\n"
            """
    # Prepare review data for formatting
    review_data = f"""Parliamentary Review Assessment:
- Review Attempts: {review_attempts}
- Review Result: {'PASSED' if review_passed else 'FAILED'}
- Full Assessment: {review_assessment}"""

    # Build formatting prompt
    formatting_prompt = f"""{prompt_template}

**User Query:** {user_question}

**Search Results Data:**
{search_data}

**AI Response to Format:**
{ai_response}

**Review Assessment Data:**
{review_data}

**Generate JSON Output (include review information in the JSON):**"""

    # Get JSON formatting from LLM
    json_response = llm.invoke([HumanMessage(content=formatting_prompt)])
    json_output = json_response.content

    # Return state update with JSON output
    return {"json_output": json_output}


def create_search_data(result, i, search_data):
    pq_id = result.get("id", f"ID-{i}")
    ask_date = result.get("ask_date", result.get("source", ""))
    uin = result.get("uin", "")
    question = result.get("question", "")
    answer = result.get("answer", "")
    score = result.get("score", "")
    source = result.get("source", "")

    # Format date for display
    date_display = f" - {ask_date}" if ask_date else ""

    search_data += f"\nResult {i} (PQ {pq_id}{date_display}):\n"
    if question:
        search_data += f"  Question: {question}\n"
    if uin:
        search_data += f"  UIN: {uin}\n"
    if answer:
        search_data += f"  Answer: {answer}\n"
    if score:
        search_data += f"  Score: {score}\n"
    if source:
        search_data += f"  Source: {source}\n"
    return search_data
