"""
Response Agent for LangGraph Semantic Search Bot

This module contains the response node implementation that generates
LLM-based responses using pre-extracted key elements.
"""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI


def response_node(state: dict[str, Any]):
    """
    LangGraph Node #4: LLM Response Generation

    This node demonstrates:
    - Using pre-extracted key elements for response generation
    - Simplified prompt focused only on composition
    - Review feedback integration for response improvement
    - Clean separation of concerns (extraction vs composition)

    The node takes key elements extracted by the previous node and uses
    them to compose a parliamentary-standard response.
    """
    # Import dependencies at runtime to avoid circular imports
#    from simple_langgraph_semantic_bot import DISPLAY_SEARCH_RESULTS, llm
    from simple_langgraph_semantic_bot import DISPLAY_SEARCH_RESULTS

    llm = ChatOpenAI(model="o4-mini")

    # Access key elements from state (populated by key_elements_node)
    key_elements = state.get("key_elements", {})
    extracted_elements = key_elements.get("elements", "")
    user_question = key_elements.get("user_question", "")
    extraction_status = key_elements.get("extraction_status", "")

    # Check for review feedback (indicates this is a regeneration attempt)
    review_feedback = state.get("review_feedback")
    review_attempts = state.get("review_attempts", 0)
    review_assessment = state.get("review_assessment", "")

    # Get the previous response if this is a regeneration
    previous_response = ""
    if review_attempts > 0:
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage):
                previous_response = msg.content
                break

    # Generate response using LLM
    if extraction_status != "success" or not extracted_elements:
        response_content = "I couldn't find any relevant information to answer your question."
    else:
        # Load response generation prompt
        with open("prompts/response_generation_prompt.md", encoding="utf-8") as file:
            response_prompt = file.read().strip()

        prompt = create_prompt(response_prompt,
                               previous_response,
                               user_question,
                               extracted_elements,
                               review_feedback,
                               review_attempts,
                               review_assessment)

        # Get LLM response
        ai_response = llm.invoke([HumanMessage(content=prompt)])
        response_content = ai_response.content

    # Create AI message
    ai_message = AIMessage(content=response_content)

    # Display the response for monitoring
    if DISPLAY_SEARCH_RESULTS:
        attempt_info = f" (Attempt {review_attempts + 1})" if review_attempts > 0 else ""
        print(f"\n🤖 RESPONSE GENERATED{attempt_info}:")
        print("=" * 70)
        print(response_content)
        print("=" * 70)

    # Return state update with new message
    return {"messages": [ai_message]}


def create_prompt(response_prompt,
                  previous_response,
                  user_question,
                  extracted_elements,
                  review_feedback,
                  review_attempts,
                  review_assessment):
    # Build the base prompt
    base_prompt = f"""{response_prompt}

**New Question:** {user_question}

**Key Elements:**
{extracted_elements}"""

    # Add review feedback if this is a regeneration attempt
    if review_feedback and review_attempts > 0:
        # Extract specific failure points from the assessment
        failure_points = []
        passed_criteria = []
        if review_assessment:
            lines = review_assessment.split("\n")
            for line in lines:
                if ": NO" in line:
                    failure_points.append(line.strip())
                elif ": YES" in line:
                    passed_criteria.append(line.strip())

        # Count how many criteria passed vs failed
        num_passed = len(passed_criteria)
        num_failed = len(failure_points)

        prompt = f"""{base_prompt}

**⚠️ RESPONSE IMPROVEMENT REQUIRED (Attempt {review_attempts + 1}/3)**

**YOUR PREVIOUS RESPONSE:**
{previous_response}

**Review Results:**
✅ Passed Criteria: {num_passed}/7
❌ Failed Criteria: {num_failed}/7

**Failed Items:**
{chr(10).join(failure_points) if failure_points else "No specific criteria listed"}

**SPECIFIC IMPROVEMENTS NEEDED:**
{review_feedback}

**CRITICAL INSTRUCTIONS FOR EDITING:**
1. Your previous response was MOSTLY GOOD - it passed {num_passed} out of 7 criteria
2. Make MINIMAL changes - only fix what's specifically mentioned in the feedback
3. For word count issues:
   - If over by less than 20%, remove only redundant phrases or combine sentences
   - Do NOT delete entire paragraphs unless explicitly told to
   - Preserve ALL key information and facts
4. For other issues:
   - Make the smallest possible change to fix the issue
   - Keep 90%+ of your original text intact
5. Think of this as light editing, not rewriting

**Example of good editing for word count:**
- BEFORE: "The Government remains committed to fostering evidence-based innovation in residual waste reduction through continued collaboration with WRAP and sector stakeholders."
- AFTER: [Delete this sentence as suggested]
- SAVINGS: ~20 words

**Your LIGHTLY EDITED Response (minimal changes only):**"""
    else:
        prompt = f"""{base_prompt}

**Your Response:**"""

    return prompt
