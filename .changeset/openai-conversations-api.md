---
'@livekit/agents-plugin-openai': minor
---

Add OpenAI Conversations API support to the Responses LLM (WebSocket and HTTP) via a new `conversation` option. Each turn sends only the items the server conversation does not already hold, system instructions go through the request `instructions` field instead of being persisted into the conversation, and function calls left without an output are answered with a placeholder so OpenAI does not reject the request.
