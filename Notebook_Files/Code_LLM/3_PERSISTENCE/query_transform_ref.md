# Quick Reference: Query Transformation in RAG

## Troubleshooting

**Issue**: Query not being transformed
- Check if chat_history is being passed correctly
- Verify CONTEXTUALIZE_QUERY_PROMPT is properly formatted
- Ensure LLM (simpler_llm) is initialized

**Issue**: Over-contextualization
- Adjust number of history messages (currently last 6)
- Refine system prompt in CONTEXTUALIZE_QUERY_PROMPT

**Issue**: Performance concerns
- Query transformation adds one extra LLM call per user question
- Consider caching for repeated similar questions
- Use faster LLM model for contextualization