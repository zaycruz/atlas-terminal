# Atlas AI Mode Design

Goal: provide an optional chat-first workflow that lets users talk to an LLM
running locally via Ollama while still executing trading actions safely through
our existing Alpaca broker layer.

## Components

1. **Ollama client wrapper (`atlas/ai/client.py`)**
   - Thin helper around the Ollama HTTP API (default `http://localhost:11434`).
   - Supports configurable model name via `ATLAS_AI_MODEL` (default: `llama3.2`).
   - Exposes `generate(messages, tools)` returning a response object with
     optional tool call payloads.

2. **Tool registry (`atlas/ai/tools.py`)**
   - Wraps broker operations (account snapshot, list positions/orders, submit
     buy/sell, cancel, quote).
   - Each tool is defined with a schema (name, description, expected args) and a
     callable that receives the broker instance.
   - Returns structured data (dict) ready to render back to the user.

3. **Conversation loop (`atlas/ai/chat.py`)**
   - Maintains conversation history, provides the system prompt, and enforces
     tool-handling contract.
   - On each turn:
     1. Send user + history to Ollama with tool metadata.
     2. If the model returns tool call JSON, execute the mapped tool through the
        broker, append the result as an assistant/tool message, and re-query the
        model for the final answer.
     3. Otherwise stream/print the assistant reply.
   - Uses Rich for output, reuses CLI render helpers when returning structured
     tool results.

4. **CLI integration**
   - New subcommand `atlas ai` (and terminal command `AI` in REPL) launches the
     chat loop.
   - Options: `--model` to override model name, `--env` to reuse paper/live
     selection.

## Tool invocation contract

- System prompt instructs the model to emit tool calls in JSON form wrapped in a
  fenced block tagged `atlas_tool`:

  ```
  ```atlas_tool
  {"tool": "account", "args": {}}
  ```
  ```

- Chat loop parses fenced blocks; multiple tool calls are executed sequentially.
- Tool responses are appended as assistant messages with role `tool` so the
  model receives structured feedback before composing the final reply.

## Error handling & safety

- Invalid/missing tools => inform the user and continue the conversation.
- Broker errors bubble up as rich-formatted error messages and are also passed
  back to the model for context.
- Ollama connectivity issues surface immediately with actionable hints.

## Future extensions

- Add analytics/logging for tool invocations.
- Support streaming responses once the baseline flow is stable.
- Introduce additional analysis tools (news, technical indicators, etc.).
