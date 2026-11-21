"""v2 provider layer: real LLMs behind the MOCK_LLM switch, through one generic OpenAI-compatible endpoint.

`settings`  which provider (mock | ollama | openai_compatible), read from the environment on demand.
`errors`    typed, user-safe errors (unreachable, timeout, rate limit, auth, unknown model, bad response).
`metering`  reserve-then-settle charging of the per-request token budget on the provider's reported usage.
`crew_llm`  CrewAI's own OpenAICompletion class, metered.
`autogen_client`  AutoGen's OpenAIChatCompletionClient, metered.
With no environment variables set the provider is "mock" and nothing here is used.
"""
