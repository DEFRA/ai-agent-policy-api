# Prompt Templates

This folder contains all the prompt templates used by the LangGraph Semantic Search Bot agents.

## Prompt Files

### `filter_prompt.md`
Used by the **Filter Agent** to assess the relevance of search results beyond semantic similarity. The LLM evaluates each search result and determines if it's RELEVANT or IRRELEVANT to the user's question.

### `prompt_template.md`
Used by the **Response Agent** to generate comprehensive, summarized responses based on filtered search results. This is the main prompt for creating the Parliamentary Question response.

### `review_prompt.md`
Used by the **Review Agent** to assess responses against Parliamentary Question standards. Evaluates 6 criteria:
1. Question answered appropriately
2. Information currency (up-to-date facts)
3. Parliamentary language appropriateness
4. Response length (150-200 words)
5. Sensitive topics handling
6. Readability and quality

### `json_formatter_prompt.md`
Used by the **JSON Formatter Agent** to create structured JSON output for API consumption. Takes the AI response, search results, and review assessment to generate a properly formatted JSON response.

## Usage

Each agent loads its corresponding prompt template at runtime:
```python
with open("prompts/[prompt_name].md", 'r', encoding='utf-8') as file:
    prompt_template = file.read().strip()
```

## Editing Prompts

You can modify these prompts without changing any code. The agents will automatically use the updated prompts on the next execution. 