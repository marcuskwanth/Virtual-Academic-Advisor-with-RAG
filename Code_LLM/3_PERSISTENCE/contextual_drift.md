# Contextual Drift in RAG Systems

When implementing chat memory in RAG (Retrieval-Augmented Generation) systems, a critical issue arises with follow-up questions: **contextual drift**.

**Example of the Problem:**
```
User: "What is the BEng Scheme in IAIE about?"
System: [Retrieves IAIE programme documents] ✓
System: "The BEng in Industrial and Systems Engineering..."

User: "What are the career paths for that programme?"
System: [Retrieves random programme documents] ✗
System: [Returns information about a DIFFERENT programme]
```

The retriever doesn't understand that "that programme" refers to IAIE, leading to incorrect document retrieval.

## Solution: Query Transformation

### What is Query Transformation?

Query transformation (also called query rewriting) converts ambiguous follow-up questions into standalone, context-rich queries BEFORE they hit the retriever.

**Flow:**
```
User Question → [Query Transformation] → Contextualized Query → RAG-Fusion → Retrieve → Answer
```

## How It Works

### Step-by-Step Process

1. **User asks initial question**: "What is the BEng Scheme in IAIE about?"
   - No chat history → Question passed through as-is
   - Retrieves IAIE documents
   - Response stored in chat history

2. **User asks follow-up**: "What are the career paths for that programme?"
   - **Query Transformation** kicks in:
     - Analyzes chat history
     - Identifies "that programme" = IAIE
     - Transforms to: "What are the career paths for the BEng Scheme in IAIE?"
   
3. **RAG-Fusion** uses contextualized query:
   - Generates variations of the contextualized query
   - All queries now contain "IAIE" context
   
4. **Retrieval**:
   - Uses contextualized queries
   - Retrieves IAIE-specific career documents ✓
   
5. **Answer Generation**:
   - Uses correctly retrieved documents
   - Provides accurate, context-aware answer

### Before vs After Comparison

| Aspect | Without Transformation | With Transformation |
|--------|----------------------|-------------------|
| Follow-up Query | "What are the career paths for that programme?" | → "What are the career paths for BEng in IAIE?" |
| Documents Retrieved | Random programme docs ✗ | IAIE programme docs ✓ |
| Answer Quality | Incorrect/Generic | Specific & Accurate |
| Context Preservation | Lost | Maintained |

## Benefits

1. **Prevents Contextual Drift**: Follow-up questions maintain context from previous exchanges
2. **Better Document Retrieval**: Retriever gets complete, standalone queries
3. **Improved Answer Accuracy**: Correct documents lead to accurate answers
4. **Natural Conversations**: Users can ask follow-ups naturally without repeating context
5. **Programme-Specific Accuracy**: Critical for multi-programme databases

## Testing the Implementation

Run the demonstration cells in the notebook to see:
- Query transformation in action
- Contextualized queries printed for verification
- Comparison of retrieval quality

## Technical Notes

- Preserves last 6 messages (3 exchanges) for context window management
- Query transformation happens BEFORE RAG-Fusion for optimal results
- Works seamlessly with existing memory persistence

## Files Modified

- `/Code_LLM/3_PERSISTENCE/Memory_Store/3_1_inMemory.ipynb`
  - Added query contextualization prompt
  - Added `contextualize_question()` function
  - Updated state to include `contextualized_question`
  - Modified graph to include query transformation step
  - Added demonstration cells

## Future Enhancements

1. **Advanced Contextualization**: Use more sophisticated prompt engineering
2. **Selective Transformation**: Only transform when ambiguity detected
3. **Multi-turn Context**: Extend beyond last 3 exchanges for complex conversations
4. **Caching**: Cache contextualized queries to reduce LLM calls
5. **Metrics**: Track transformation effectiveness and retrieval accuracy
