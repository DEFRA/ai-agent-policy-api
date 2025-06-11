"""
Key Elements Agent for LangGraph Semantic Search Bot

This module contains the key elements extraction node that identifies
and structures important information from filtered search results.
"""

from typing import Any

from langchain_core.messages import HumanMessage


def key_elements_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph Node #3: Key Elements Extraction

    This node:
    - Analyzes filtered search results
    - Extracts key facts, figures, and policy positions
    - Categorizes information by relevance and type
    - Structures data for the response generation node

    The extracted elements are stored in state for use by the response node,
    creating a clear separation of concerns between information extraction
    and response composition.
    """
    # Import dependencies at runtime to avoid circular imports
#from simple_langgraph_semantic_bot import DISPLAY_SEARCH_RESULTS, llm
    from simple_langgraph_semantic_bot import DISPLAY_SEARCH_RESULTS
    from utils import extract_search_content

    llm = ChatOpenAI(model="o4-mini")

    # Access filtered search results from state
    filtered_results = state.get("filtered_results", [])

    # Extract user question from message history
    user_question = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_question = msg.content
            break

    # Extract key elements if we have filtered results
    if not filtered_results:
        key_elements = {
            "elements": [],
            "user_question": user_question,
            "extraction_status": "No relevant search results found"
        }
    else:
        # Load the key elements extraction prompt
        with open("prompts/key_elements_prompt.md", encoding="utf-8") as file:
            extraction_prompt = file.read().strip()

        # Prepare search context
        search_context = extract_search_content(filtered_results)

        # Build complete prompt
        prompt = f"""{extraction_prompt}

**New Question:** {user_question}

**Search Results:**
{search_context}

**Extract Key Elements:**"""

        # Get LLM to extract key elements
        extraction_response = llm.invoke([HumanMessage(content=prompt)])
        extracted_content = extraction_response.content

        # Structure the key elements
        key_elements = {
            "elements": extracted_content,
            "user_question": user_question,
            "extraction_status": "success"
        }

        # Display extracted elements for monitoring
        if DISPLAY_SEARCH_RESULTS:
            print("\n🔑 KEY ELEMENTS EXTRACTED:")
            print("=" * 70)
            print(extracted_content)
            print("=" * 70)

    # Return state update with key elements
    return {"key_elements": key_elements}
