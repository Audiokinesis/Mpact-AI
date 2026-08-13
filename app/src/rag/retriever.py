import os
from typing import List, Tuple, Dict, Any

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ----------------------------------------------------------------------
# Configuration Constants
# ----------------------------------------------------------------------
VECTOR_STORE_PATH = r"C:\Users\mdesc\Documents\Projects\MpactAI\app\src\rag\vector_store\faiss_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Default Retrieval Settings
DEFAULT_TOP_K = 3  # Number of relevant chunks to retrieve


class RAGRetriever:
    """
    Handles vector store loading and semantic similarity retrieval for RAG pipelines.
    """

    def __init__(self, vector_store_path: str = VECTOR_STORE_PATH):
        self.vector_store_path = vector_store_path
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.vector_store = self._load_vector_store()

    def _load_vector_store(self) -> FAISS:
        """Loads the persisted FAISS vector index from disk."""
        if not os.path.exists(self.vector_store_path):
            raise FileNotFoundError(
                f"Vector store index not found at '{self.vector_store_path}'. "
                "Please run 'python src/rag/ingest.py' first."
            )
        
        # allow_dangerous_deserialization=True is required by LangChain to load 
        # locally saved pickle files (.pkl) generated during ingestion.
        vector_store = FAISS.load_local(
            folder_path=self.vector_store_path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_store

    def retrieve_with_scores(
        self, query: str, top_k: int = DEFAULT_TOP_K
    ) -> List[Tuple[Document, float]]:
        """
        Retrieves top-k document chunks closest to the query, along with distance scores.
        
        Note: FAISS uses L2 (Euclidean) distance by default with this model.
        Lower score = Closer vector distance = Higher semantic similarity.
        """
        results = self.vector_store.similarity_search_with_score(query, k=top_k)
        return results

    def get_context_for_llm(
        self, query: str, top_k: int = DEFAULT_TOP_K
    ) -> Dict[str, Any]:
        """
        Executes search and formats retrieved chunks into a clean context block 
        for injection into LLM prompts.
        """
        results = self.retrieve_with_scores(query, top_k=top_k)
        
        formatted_chunks = []
        context_str_list = []

        for rank, (doc, score) in enumerate(results, start=1):
            source_file = doc.metadata.get("file_name", "Unknown Source")
            page_num = doc.metadata.get("page", 0) + 1  # 1-indexed for display
            chunk_id = doc.metadata.get("chunk_id", "N/A")
            content = doc.page_content.strip()

            # Structured citation format
            formatted_entry = (
                f"--- [SOURCE {rank}]: {source_file} (Page {page_num}) ---\n"
                f"{content}\n"
            )
            context_str_list.append(formatted_entry)

            formatted_chunks.append({
                "rank": rank,
                "score": round(score, 4),
                "source": source_file,
                "page": page_num,
                "chunk_id": chunk_id,
                "content": content
            })

        # Combined string ready for standard prompt templates
        raw_context = "\n".join(context_str_list)

        return {
            "query": query,
            "raw_context": raw_context,
            "retrieved_documents": formatted_chunks
        }


# ----------------------------------------------------------------------
# Demonstration Execution
# ----------------------------------------------------------------------
def main():
    # Sample Test Query
    test_query = "What programs support middle-school students?"

    print("\n" + "=" * 70)
    print(f"🔎 INITIALIZING RETRIEVER FOR QUERY: '{test_query}'")
    print("=" * 70)

    # Initialize Retriever
    retriever = RAGRetriever()

    # Execute Search & Format
    output = retriever.get_context_for_llm(test_query, top_k=3)

    # Display Breakdown of Retrieved Context
    print(f"\n📊 RETRIEVED {len(output['retrieved_documents'])} MATCHING CHUNKS:\n")

    for item in output["retrieved_documents"]:
        print(f"• Rank {item['rank']} | L2 Distance Score: {item['score']}")
        print(f"  Source: {item['source']} (Page {item['page']})")
        print(f"  Chunk ID: {item['chunk_id']}")
        print(f"  Snippet: {item['content'][:150]}...\n")

    print("=" * 70)
    print("📝 PROMPT-READY CONTEXT BLOCK:")
    print("=" * 70)
    print(output["raw_context"])


if __name__ == "__main__":
    main()