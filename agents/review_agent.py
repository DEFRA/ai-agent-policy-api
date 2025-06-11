"""
Review Agent for LangGraph Semantic Search Bot

This module contains the Parliamentary review node implementation that assesses
responses against Parliamentary Question standards and the conditional routing logic.
"""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage


def review_node(state: dict[str, Any]):
    """
    LangGraph Node #4: Parliamentary Question Review Agent

    This node demonstrates:
    - Quality assessment: Evaluating response against Parliamentary standards
    - Conditional routing: Determining whether to proceed or loop back for improvement
    - Feedback generation: Providing specific guidance for response improvement
    - Loop prevention: Tracking attempts to avoid infinite cycles

    The node assesses responses against 6 criteria:
    1. Question answered appropriately
    2. Information currency (up-to-date facts)
    3. Parliamentary language appropriateness
    4. Response length (150-200 words)
    5. Sensitive topics handling
    6. Readability and quality
    """
    # Import dependencies at runtime to avoid circular imports
#    from simple_langgraph_semantic_bot import DISPLAY_SEARCH_RESULTS, llm
    from simple_langgraph_semantic_bot import DISPLAY_SEARCH_RESULTS, get_llm
    from utils import extract_search_content
    llm = get_llm()

    # Get the latest AI response to review
    ai_response = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            ai_response = msg.content
            break

    # Get the user's original question for context
    user_question = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_question = msg.content
            break

    # Get current review attempts to track loops
    review_attempts = state.get("review_attempts", 0)

    # Get filtered search results to verify accuracy
    filtered_results = state.get("filtered_results", [])

    # Format search results for review context
    search_context = ""
    if filtered_results:
        search_context = "\n**Available Source Material:**\n"
        search_context += extract_search_content(filtered_results)
    else:
        search_context = "\n**No search results available for verification**"

    # Load review prompt template
    with open("prompts/review_prompt.md", encoding="utf-8") as file:
        review_prompt_template = file.read().strip()

    # Build review prompt
    review_prompt = f"""{review_prompt_template}

**Original User Question:** {user_question}

{search_context}

**Response to Review:**
{ai_response}

**Previous Review Feedback (if any):** {state.get("review_feedback", "None - this is the first review")}

**Current Attempt:** {review_attempts + 1}/3

Please assess this response:"""

    # Get review assessment from LLM
    review_response = llm.invoke([HumanMessage(content=review_prompt)])
    assessment = review_response.content

    # Parse review result - look for PASS/FAIL in the assessment
    review_passed = "OVERALL RESULT: PASS" in assessment.upper()

    # Extract feedback section for potential improvements
    feedback = ""
    if "**FEEDBACK:**" in assessment:
        feedback = assessment.split("**FEEDBACK:**")[1].strip()
    elif not review_passed:
        feedback = "The response did not meet all Parliamentary Question standards. Please review the assessment above and improve accordingly."

    # Display review results if enabled
    if DISPLAY_SEARCH_RESULTS:
        print("\n📋 PARLIAMENTARY REVIEW ASSESSMENT:")
        print("=" * 70)
        print(f"Attempt: {review_attempts + 1}/3")
        print("-" * 70)
        print(assessment)  # Print the full assessment output
        print("-" * 70)
        print(f"Overall Result: {'PASSED' if review_passed else 'FAILED'}")
        if not review_passed:
            print("\n🔄 Feedback provided to response agent for improvement")
        print("=" * 70)

    # Update state with review results
    return {
        "review_passed": review_passed,
        "review_feedback": feedback if not review_passed else None,
        "review_attempts": review_attempts + 1,
        "review_assessment": assessment  # Store full assessment for JSON output
    }


def review_decision(state: dict[str, Any]) -> str:
    """
    LangGraph Conditional Edge Function: Route based on review outcome.

    This function implements LangGraph's conditional routing:
    - Returns string keys that map to destination nodes
    - Enables dynamic workflow based on state conditions
    - Prevents infinite loops with attempt limits

    Returns:
        "json_formatter": If review passed or max attempts reached
        "response": If review failed and attempts < 3 (loop back for improvement)
    """
    review_passed = state.get("review_passed", False)
    review_attempts = state.get("review_attempts", 0)

    if review_passed:
        return "json_formatter"  # Continue to final output
    if review_attempts >= 3:
        return "json_formatter"  # Prevent infinite loops - proceed anyway
    return "response"        # Loop back for improvement
