"""
LangGraph Semantic Search Service - Full Pipeline

Complete implementation matching simple_langgraph_semantic_bot.py for FastAPI integration.
Uses all existing agents with internal search service and disabled printing.

Features:
- Complete 6-node workflow with conditional feedback loops
- Parliamentary Question review with 7 quality criteria
- Key elements extraction for consistent information structuring
- LLM-based relevance filtering
- Internal search service using DataFrame
- JSON-only output (no terminal printing)

Workflow: Search → Filter → Key Elements → Response → Review → [Loop if needed] → JSON Output
"""

import json
import sys
from typing import Annotated, Any, Optional

# Internal API imports
from langchain_core.messages import HumanMessage

# LLM import
from langchain_openai import ChatOpenAI

# LangGraph imports
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from agents import (
    filter_node,
    json_formatter_node,
    key_elements_node,
    response_node,
    review_decision,
    review_node,
)

# =============================================================================
# CONFIGURATION - API Optimized
# =============================================================================

# Disable all printing for API usage
DISPLAY_SEARCH_RESULTS = False
DISPLAY_JSON_OUTPUT = False

"""
# LLM Configuration
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="o4-mini", openai_api_key=api_key)
"""

# Global LangGraph workflow variable
semantic_chat_graph = None

# =============================================================================
# GENERAL FUNCTIONS
# =============================================================================
def get_semantic_graph():
    return semantic_chat_graph


# =============================================================================
# UTILITY FUNCTIONS - For Agent Compatibility
# =============================================================================


def display_search_results(results: list[dict[str, Any]], user_message: str):
    """API-compatible display function that respects DISPLAY_SEARCH_RESULTS flag."""
    if not DISPLAY_SEARCH_RESULTS:
        return

    if not results:
        print(f"\n❌ No search results found for: {user_message}")
        return

    print(f"\n🔍 SEMANTIC SEARCH RESULTS ({len(results)} found):")
    print("=" * 60)

    for i, result in enumerate(results, 1):
        if isinstance(result, dict):
            pq_id = result.get("id", f"ID-{i}")
            ask_date = result.get("ask_date", result.get("source", ""))
            question = result.get("question", "")
            answer = result.get("answer", "")
            score = result.get("score", result.get("similarity", ""))

            date_display = f" - {ask_date}" if ask_date else ""
            print(f"\nResult {i} (PQ {pq_id}{date_display}):")
            if question:
                print(f"Question: {question}")
            if answer:
                print(f"Answer: {answer}")
            if score:
                print(f"Score: {score}")

    print("=" * 60)


def extract_search_content(results: list[dict[str, Any]]) -> str:
    """Format search results for LLM consumption. Shared utility used by multiple agents."""
    if not results:
        return "No search results available."

    content_parts = []
    for i, result in enumerate(results, 1):
        pq_id = result.get("id", f"ID-{i}")
        ask_date = result.get("ask_date", result.get("source", ""))

        # Format date for display
        date_display = f" - {ask_date}" if ask_date else ""

        part = f"Result {i} (PQ {pq_id}{date_display}):"
        part += f"\nQuestion: {result.get('question', '')}"
        part += f"\nAnswer: {result.get('answer', '')}"
        part += f"\n(Relevance Score: {result.get('score', result.get('similarity', ''))})"
        part += f"\n(Source: {result.get('source', '')})"
        content_parts.append(part)

    return "\n\n".join(content_parts)

# =============================================================================
# AGENT DEPENDENCY INJECTION
# =============================================================================


def setup_agent_dependencies():
    """Set up the dependencies that agents expect to import."""

    # Create a mock utils module if it doesn't exist
    if "utils" not in sys.modules:
        import types
        utils_module = types.ModuleType("utils")
        utils_module.display_search_results = display_search_results
        utils_module.extract_search_content = extract_search_content
        sys.modules["utils"] = utils_module

    # Create a mock simple_langgraph_semantic_bot module for API context
    if "simple_langgraph_semantic_bot" not in sys.modules:
        import types
        bot_module = types.ModuleType("simple_langgraph_semantic_bot")
#        bot_module.llm = llm
        bot_module.llm = ChatOpenAI(model="o4-mini")
        bot_module.DISPLAY_SEARCH_RESULTS = DISPLAY_SEARCH_RESULTS
        bot_module.DISPLAY_JSON_OUTPUT = DISPLAY_JSON_OUTPUT
        # We'll set search_tool when we create the internal service
        sys.modules["simple_langgraph_semantic_bot"] = bot_module


# Set up dependencies immediately
setup_agent_dependencies()

# =============================================================================
# STATE DEFINITION - Complete Pipeline State
# =============================================================================


class ApiSemanticState(TypedDict):
    """Complete LangGraph State Schema matching main pipeline."""
    messages: Annotated[list,
                        add_messages]  # Auto-managed conversation history
    # Raw search results from internal service
    search_results: list[dict[str, Any]]
    # Relevance-filtered search results
    filtered_results: list[dict[str, Any]]
    key_elements: Optional[dict[str, Any]]   # Extracted key information
    json_output: Optional[str]               # Final JSON formatted output
    # Review agent fields
    review_feedback: Optional[str]           # Feedback from review agent
    review_attempts: int                     # Counter to prevent infinite loops
    review_passed: bool                      # Whether review criteria were met
    review_assessment: Optional[str]         # Full review assessment output
    # History tracking for all attempts
    # All responses and their review results
    response_history: list[dict[str, Any]]
    # All feedback provided across attempts
    feedback_history: list[str]

# =============================================================================
# INTERNAL SEARCH SERVICE
# =============================================================================


class InternalSemanticSearchService:
    """Internal search service using direct Vector Store operations."""

    def __init__(self, question_store, num_results: int = 10, split_string: str = "Affairs,"):
        self.question_store = question_store
        self.num_results = num_results
        self.split_string = split_string

    def search(self, question: str) -> list[dict[str, Any]]:
        """Perform semantic search using internal Vector Store operations."""
        try:
            similarity_results = self.question_store.get_question_match(question, self.num_results)
 #           similarity_results = cosine_similarity_search(
 #               question, self.df, 'question_embedding'
 #           )

            # Format results to match the external API structure
            results = []
            for row in similarity_results:
                result = {
                    "id": str(row["id"]),
                    "question": row["question"],
                    "answer": row["answer"],
                    "score": row["score"],
                    "source": row.get("date_tabled", ""),
                    "ask_date": row.get("date_tabled", ""),
                    "similarity": row["score"]
                }
                results.append(result)

            return results
        except Exception:
            return []

# =============================================================================
# API-SPECIFIC SEARCH NODE
# =============================================================================


def create_api_search_node(search_service: InternalSemanticSearchService):
    """Factory function to create API-optimized search node."""

    def api_search_node(state: ApiSemanticState):
        """LangGraph Node #1: Internal Semantic Search (API Version)"""
        # Extract the latest user message from conversation history
        user_message = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break

        # Perform internal search (no external API call)
        results = search_service.search(user_message)

        # Display results if enabled (will be False in API mode)
        display_search_results(results, user_message)

        return {"search_results": results}

    return api_search_node

# =============================================================================
# HISTORY-TRACKING WRAPPER FUNCTIONS
# =============================================================================


def create_history_tracking_response_node():
    """Create a wrapper around response_node that tracks response history."""

    def history_tracking_response_node(state: ApiSemanticState):
        """Wrapper that captures each response attempt for history tracking."""

        # Call the original response agent
        result = response_node(state)

        # Extract the generated response
        # response_node returns one new message
        new_message = result["messages"][0]
        response_text = new_message.content

        # Get current attempt number
        attempt_number = state.get("review_attempts", 0) + 1

        # Create history entry
        history_entry = {
            "attempt": attempt_number,
            "response": response_text,
            "timestamp": "generated",  # Could add actual timestamp if needed
            "word_count": len(response_text.split()) if response_text else 0
        }

        # Update response history
        response_history = state.get("response_history", [])
        response_history.append(history_entry)

        # Return the original result plus history update
        result["response_history"] = response_history
        return result

    return history_tracking_response_node


def create_history_tracking_review_node():
    """Create a wrapper around review_node that tracks review history."""

    def history_tracking_review_node(state: ApiSemanticState):
        """Wrapper that captures each review assessment and feedback for history tracking."""

        # Call the original review agent
        result = review_node(state)

        # Extract review information
        review_assessment = result.get("review_assessment", "")
        review_feedback = result.get("review_feedback", "")
        review_passed = result.get("review_passed", False)
        attempt_number = result.get("review_attempts", 1)

        # Update response history with review results
        response_history = state.get("response_history", [])
        if response_history and len(response_history) >= attempt_number:
            # Update the latest response entry with review results
            response_history[attempt_number - 1].update({
                "review_passed": review_passed,
                "review_assessment": review_assessment,
                "review_feedback": review_feedback if review_feedback else None
            })

        # Update feedback history
        feedback_history = state.get("feedback_history", [])
        if review_feedback:  # Only add if there's actual feedback
            feedback_entry = f"Attempt {attempt_number}: {review_feedback}"
            feedback_history.append(feedback_entry)

        # Return the original result plus history updates
        result["response_history"] = response_history
        result["feedback_history"] = feedback_history
        return result

    return history_tracking_review_node


def create_history_aware_json_formatter():
    """Create a wrapper around json_formatter_node that includes response history."""

    def history_aware_json_formatter(state: ApiSemanticState):
        """Wrapper that ensures response history is included in JSON output."""

        # Call the original JSON formatter
        result = json_formatter_node(state)

        # Get the JSON output and parse it to add history
        json_output = result.get("json_output", "{}")

        try:
            # Parse the existing JSON
            if json_output.startswith("```json"):
                json_output = json_output[7:]
            if json_output.endswith("```"):
                json_output = json_output[:-3]
            json_output = json_output.strip()

            parsed_json = json.loads(json_output)

            # Add response and feedback history
            response_history = state.get("response_history", [])
            feedback_history = state.get("feedback_history", [])

            # Update the JSON structure
            parsed_json["response_attempts"] = response_history
            parsed_json["feedback_history"] = feedback_history

            # Replace the parliamentary_review section with more detailed info
            if response_history:
                parsed_json["parliamentary_review"] = {
                    "total_attempts": len(response_history),
                    "final_status": "PASSED" if state.get("review_passed", False) else "FAILED",
                    "all_attempts": response_history,
                    "final_assessment": state.get("review_assessment", "")
                }

            # Convert back to JSON string
            updated_json = json.dumps(parsed_json, indent=2)
            result["json_output"] = updated_json

        except (json.JSONDecodeError, Exception):
            # If JSON parsing fails, at least add history as a fallback
            fallback_addition = f'\n\n"response_attempts": {json.dumps(response_history)},\n"feedback_history": {json.dumps(feedback_history)}'
            result["json_output"] = json_output + fallback_addition

        return result

    return history_aware_json_formatter

# =============================================================================
# IMPORT AGENTS AFTER DEPENDENCIES ARE SET UP
# =============================================================================


# Now we can safely import the agents since dependencies are available

# Create history-tracking versions AFTER imports
history_response_node = create_history_tracking_response_node()
history_review_node = create_history_tracking_review_node()
history_json_formatter = create_history_aware_json_formatter()

# =============================================================================
# LANGGRAPH WORKFLOW BUILDER
# =============================================================================


def build_semantic_chat_graph(question_store):
    """Build complete LangGraph workflow matching main pipeline."""
    global semantic_chat_graph
    # Initialize internal search service
    search_service = InternalSemanticSearchService(question_store)

    # Make search service available to mock module (for compatibility)
    sys.modules["simple_langgraph_semantic_bot"].search_tool = search_service

    # Create graph with complete state schema
    graph_builder = StateGraph(ApiSemanticState)

    # Add all nodes using existing agents
    graph_builder.add_node("search", create_api_search_node(search_service))
    graph_builder.add_node("filter", filter_node)
    graph_builder.add_node("extract_key_elements", key_elements_node)
    graph_builder.add_node("response", history_response_node)
    graph_builder.add_node("review", history_review_node)
    graph_builder.add_node("json_formatter", history_json_formatter)

    # Define complete workflow with all edges
    graph_builder.add_edge(START, "search")
    graph_builder.add_edge("search", "filter")
    graph_builder.add_edge("filter", "extract_key_elements")
    graph_builder.add_edge("extract_key_elements", "response")
    graph_builder.add_edge("response", "review")

    # Add conditional edges for feedback loops
    graph_builder.add_conditional_edges(
        "review",
        review_decision,
        {
            "response": "response",           # If review fails, loop back
            "json_formatter": "json_formatter"  # If review passes, continue
        }
    )

    graph_builder.add_edge("json_formatter", END)

    # Compile and return complete workflow
    semantic_chat_graph =  graph_builder.compile()
    return semantic_chat_graph

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================


def run_semantic_chat(graph, user_input: str) -> dict[str, Any]:
    """Execute complete LangGraph workflow and return structured response."""
    # Create initial state with all required fields
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "search_results": [],
        "filtered_results": [],
        "key_elements": None,
        "json_output": None,
        "review_feedback": None,
        "review_attempts": 0,
        "review_passed": False,
        "review_assessment": None,
        "response_history": [],
        "feedback_history": []
    }

    # Execute complete graph workflow
    final_state = graph.invoke(initial_state)

    # Extract all results for API response
    final_message = final_state["messages"][-1]
    search_results = final_state.get("search_results", [])
    filtered_results = final_state.get("filtered_results", [])
    key_elements = final_state.get("key_elements", {})
    json_output = final_state.get("json_output", None)
    review_attempts = final_state.get("review_attempts", 0)
    review_passed = final_state.get("review_passed", False)
    review_assessment = final_state.get("review_assessment", "")
    response_history = final_state.get("response_history", [])
    feedback_history = final_state.get("feedback_history", [])

    return {
        "response": final_message.content,
        "search_results": search_results,
        "filtered_results": filtered_results,
        "key_elements": key_elements,
        "json_output": json_output,
        "review_attempts": review_attempts,
        "review_passed": review_passed,
        "review_assessment": review_assessment,
        "response_history": response_history,
        "feedback_history": feedback_history
    }
