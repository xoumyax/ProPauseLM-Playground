# Test Data Field Description

This directory contains cleaned outputs from the 68-query data collection run.

## query_template.jsonl

One JSON object per original query. This file is the final organized answer set.

Fields:

- `id`: Query id, for example `q001`.
- `original_query`: The original query before any refinement.
- `type`: `clear` if the original query was satisfied in the first agent run; `vague` if it required refinement or remained unsatisfied after the maximum iterations.
- `clarified_query`: The final query used for the final answer. For `clear` examples this is usually the same as `original_query`; for `vague` examples this is the refined query.
- `final_response`: The final answer saved for this query.
- `number_of_tokens`: Sum of `agent_response.trace.total_tokens` over all agent iterations for this query.
- `iterations`: Number of agent iterations run for this query.
- `satisfied`: Whether the final saved answer is satisfied.
- `final_response_tokens`: Token count of `final_response` only, computed with the tokenizer listed above.

## llm_responses.jsonl

One JSON object per agent iteration. This file is the complete agent trajectory.

Fields:

- `id`: Query id matching `query_template.jsonl`.
- `iteration`: Iteration number for this query, starting from 1.
- `query`: The exact query sent to the agent in this iteration.
- `agent_url`: Local endpoint used for the agent run.
- `number_of_tokens`: `agent_response.trace.total_tokens` for this single agent iteration.
- `agent_response`: Full agent response object for this iteration.
- `agent_response.response`: Final visible answer for this iteration.
- `agent_response.metadata`: Agent metadata such as tool-call counts and browsed/searched links.
- `agent_response.messages`: Chat messages sent to the workflow.
- `agent_response.trace`: Full trace captured from the workflow.
- `agent_response.trace.generated_text`: Raw generated text from the agent trace. This may include thinking text and tool-call markup if returned by the workflow.
- `agent_response.trace.tool_calls`: List of tool calls made during this iteration.
- `agent_response.trace.tool_calls[].tool_name`: Tool name.
- `agent_response.trace.tool_calls[].input_params`: Tool input parameters when available.
- `agent_response.trace.tool_calls[].output`: Processed tool output.
- `agent_response.trace.tool_calls[].raw_output`: Raw tool output.
- `agent_response.trace.tool_calls[].documents`: Retrieved or browsed documents attached to the tool call.
- `agent_response.trace.tool_calls[].runtime`: Tool runtime in seconds.
- `agent_response.trace.tool_calls[].error`: Tool error string, empty when successful.
- `response_tokens`: Token count of `agent_response.response` for this iteration only, computed with the tokenizer listed above.
- `generated_text_tokens`: Token count of `agent_response.trace.generated_text` for this iteration, computed with the tokenizer listed above.

## Where To Find The Think Process

The LLM think/original trajectory text is stored in:

- `llm_responses.jsonl -> agent_response -> trace -> generated_text`

This field may contain raw trace markup such as:

- `<think>...</think>`
- `<call_tool name="...">...</call_tool>`
- `<tool_output>...</tool_output>`
- `<answer>...</answer>`

The structured tool-call records are stored in:

- `llm_responses.jsonl -> agent_response -> trace -> tool_calls`

The final visible answer for one agent iteration is stored in:

- `llm_responses.jsonl -> agent_response -> response`

The final saved answer for each query is stored in:

- `query_template.jsonl -> final_response`

## Example: Query Refinement Process

For a `vague` query, each rewritten query appears as a new iteration in `llm_responses.jsonl`.
The final rewritten query is also copied into `query_template.jsonl -> clarified_query`.

Example `q012`:

- Initial query:
  - Location: `query_template.jsonl -> original_query`
  - Also: `llm_responses.jsonl -> id=q012 -> iteration=1 -> query`
  - Text: `Can a 3B-parameter model fine-tuned on rec data match GPT-4's ranking performance, or is the big model's scale irreplaceable?`

- First rewritten query:
  - Location: `llm_responses.jsonl -> id=q012 -> iteration=2 -> query`
  - Text: `Based on empirical studies, can a 3B-parameter model fine-tuned on recommendation data match GPT-4's ranking performance, and does task-specific fine-tuning close the performance gap?`

- Final clarified query:
  - Location: `query_template.jsonl -> clarified_query`
  - Also: `llm_responses.jsonl -> id=q012 -> iteration=3 -> query`
  - Text: `Based on empirical research, what evidence exists that a 3B-parameter model fine-tuned on recommendation data can match GPT-4's ranking performance, and does such fine-tuning close the gap between small and large models in this context?`

- Final answer:
  - Location: `query_template.jsonl -> final_response`
