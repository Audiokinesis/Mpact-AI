import os
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
# Community integration for local models
from langchain_ollama import ChatOllama

from retriever import RAGRetriever


# ----------------------------------------------------------------------
# 1. System Prompt (Grounding & Guardrails)
# ----------------------------------------------------------------------
SYSTEM_PROMPT = """You are an official organizational AI assistant for Future Horizons Youth Foundation.

Your task is to answer user questions accurately using ONLY the provided context snippets.

STRICT RULES YOU MUST FOLLOW:
1. USE RETRIEVED INFORMATION: Answer using ONLY facts directly mentioned in the provided Context. Do NOT use outside knowledge or make assumptions.
2. SAY WHEN INFORMATION ISN'T AVAILABLE: If the context does not contain enough information to answer the question, state exactly:
   "I cannot answer this question based on the provided organizational documents."
3. PROVIDE SOURCE REFERENCES: Every claim or fact in your answer MUST be immediately followed by a citation referencing the source document and page number in brackets, for example: [Source: youth_programs.pdf, Page 1].

Context:
{context}
"""


# ----------------------------------------------------------------------
# 2. RAG Pipeline Class
# ----------------------------------------------------------------------
class RAGPipeline:
    """
    End-to-end local RAG Pipeline: 
    Question -> Retriever -> Context Prompt -> Ollama LLM -> Guardrailed Answer
    """

    def __init__(self, top_k: int = 3, model_name: str = "llama3.2"):
        self.retriever = RAGRetriever()
        self.top_k = top_k
        
        # Initialize local open-source LLM via Ollama
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.0  # Zero temperature for deterministic, factual output
        )
        
        # Build Chat Prompt Template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}")
        ])

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Executes the full RAG pipeline for an input question.
        """
        # Step 1: Retrieve Relevant Context Documents
        retrieval_output = self.retriever.get_context_for_llm(question, top_k=self.top_k)
        raw_context = retrieval_output["raw_context"]
        
        # Step 2: Format Prompt with Context & Question
        prompt = self.prompt_template.format_messages(
            context=raw_context,
            question=question
        )
        
        # Step 3: Pass to Ollama LLM
        response = self.llm.invoke(prompt)
        
        return {
            "question": question,
            "answer": response.content,
            "retrieved_sources": [
                f"{doc['source']} (Page {doc['page']})" 
                for doc in retrieval_output["retrieved_documents"]
            ],
            "raw_context_used": raw_context
        }


# ----------------------------------------------------------------------
# 3. Execution & Verification
# ----------------------------------------------------------------------
def main():
    print("ℹ️ Running local Ollama RAG Pipeline. Ensure Ollama is running in the background.")

    rag_chain = RAGPipeline(top_k=3, model_name="llama3.2")

    # Test Case 1: Answerable Question (Checks Rules #1 & #3)
    q1 = "What programs support middle-school students?"
    print("\n" + "=" * 80)
    print(f"❓ QUESTION 1: {q1}")
    print("=" * 80)
    
    res1 = rag_chain.answer_question(q1)
    print(f"\n💡 ANSWER:\n{res1['answer']}\n")
    print(f"📚 SOURCES RETRIEVED: {res1['retrieved_sources']}")

    # Test Case 2: Out-of-Scope Question (Checks Rule #2)
    q2 = "What is the foundation's budget for international expansion in Europe?"
    print("\n" + "=" * 80)
    print(f"❓ QUESTION 2: {q2}")
    print("=" * 80)
    
    res2 = rag_chain.answer_question(q2)
    print(f"\n💡 ANSWER:\n{res2['answer']}\n")
    print(f"📚 SOURCES RETRIEVED: {res2['retrieved_sources']}")


if __name__ == "__main__":
    main()