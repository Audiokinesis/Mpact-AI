import os
from typing import Dict, Any
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

import sys
import pathlib

# Automatically locate and add the project root directory to Python's search path
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Now import your RAGRetriever module cleanly
from rag.retriever import RAGRetriever


# ----------------------------------------------------------------------
# 1. Tool Definitions
# ----------------------------------------------------------------------

_retriever_instance = None

def get_retriever():
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = RAGRetriever()
    return _retriever_instance


@tool
def search_knowledge_base(query: str) -> str:
    """Searches the organizational document knowledge base using RAG vector search."""
    retriever = get_retriever()
    retrieved = retriever.get_context_for_llm(query=query, top_k=3)
    return retrieved["raw_context"]


@tool
def get_program_information(program_name: str) -> str:
    """Retrieves operational details, target goals, and curriculum for a program."""
    retriever = get_retriever()
    retrieved = retriever.get_context_for_llm(query=f"Details and goals for {program_name}", top_k=2)
    return f"Program Details for '{program_name}':\n{retrieved['raw_context']}"


@tool
def get_organization_statistics() -> str:
    """Fetches verified organizational impact metrics and demographic outcomes."""
    retriever = get_retriever()
    retrieved = retriever.get_context_for_llm(query="organization metrics impact stats demographic served", top_k=2)
    return f"Verified Organizational Impact Stats:\n{retrieved['raw_context']}"


@tool
def draft_grant_section(section_name: str, key_points: str) -> str:
    """Formats key factual points into a structured grant section draft."""
    return (
        f"\n=== SECTION: {section_name.upper()} ===\n"
        f"{key_points.strip()}\n"
        f"====================================\n"
    )


# ----------------------------------------------------------------------
# 2. System Prompt & Grant Agent Class
# ----------------------------------------------------------------------

SYSTEM_AGENT_PROMPT = """You are an expert Grant Writer AI Agent for Future Horizons Youth Foundation.

Your objective is to produce a complete, compelling, and fully grounded grant proposal based on user requests.

WORKFLOW RULES:
1. GATHER FACTS FIRST: Always call `get_program_information` and `get_organization_statistics` or `search_knowledge_base` to gather necessary facts BEFORE writing. Do NOT invent stats or facts.
2. CITATION REQUIREMENT: Ensure all claims cite the retrieved document source and page number provided by the tools.
3. STRUCTURED DRAFTING: Use `draft_grant_section` to construct key proposal sections (Executive Summary, Statement of Need, Methodology, Expected Impact).
4. NO HALLUCINATIONS: If the tools return insufficient information for a section, state that clearly in the draft.
"""


class GrantAgent:
    """
    Autonomous Grant Agent built with LangGraph ReAct agent architecture.
    """

    def __init__(self, model_name: str = "llama3.2"):
        self.tools = [
            search_knowledge_base,
            get_program_information,
            get_organization_statistics,
            draft_grant_section,
        ]

        # Initialize local Ollama model
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.1
        )

        # Create the LangGraph agent executor directly
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=SYSTEM_AGENT_PROMPT
        )

    def generate_proposal(self, request_prompt: str) -> str:
        """Runs the LangGraph agent loop and extracts the final text response."""
        response = self.agent.invoke({
            "messages": [("user", request_prompt)]
        })
        
        # Extract the content from the last AI message
        final_answer = response["messages"][-1].content
        return final_answer


# ----------------------------------------------------------------------
# 3. Execution Example
# ----------------------------------------------------------------------

def main():
    print("\n🚀 INITIALIZING GRANT AGENT (LangGraph)...")
    agent = GrantAgent(model_name="llama3.2")

    user_request = (
        "Draft a $50,000 grant proposal section for the 'STEM Innovators' program targeting middle school students. "
        "Include an Executive Summary, Statement of Need, and Expected Impact using our verified stats."
    )

    print("\n" + "=" * 80)
    print(f"📋 USER REQUEST: {user_request}")
    print("=" * 80 + "\n")

    proposal = agent.generate_proposal(user_request)

    print("\n" + "=" * 80)
    print("📄 FINAL GENERATED PROPOSAL")
    print("=" * 80)
    print(proposal)


if __name__ == "__main__":
    main()