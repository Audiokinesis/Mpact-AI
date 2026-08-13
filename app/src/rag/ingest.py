import os
import glob
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ----------------------------------------------------------------------
# Pipeline Configuration & Core Concepts
# ----------------------------------------------------------------------
DOCS_DIR = r"C:\Users\mdesc\Documents\Projects\MpactAI\app\documents"
VECTOR_STORE_PATH = r"C:\Users\mdesc\Documents\Projects\MpactAI\app\src\rag\vector_store\faiss_index"

# 1. CHUNK SIZE: The maximum length (in characters/tokens) of each text block.
#    - Smaller chunks (200-500) preserve precise, granular facts.
#    - Larger chunks (1000+) maintain broader thematic context but may dilute specifics.
CHUNK_SIZE = 500

# 2. OVERLAP: The number of characters shared between consecutive chunks.
#    - Prevents cutting critical information midway through a sentence or paragraph.
#    - Ensures continuity across boundary transitions.
CHUNK_OVERLAP = 100

# 3. EMBEDDINGS MODEL: Dense vector representations of semantic text.
#    - Translates human text into high-dimensional numerical vectors (384 dimensions here).
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ----------------------------------------------------------------------
# Step 1: PDF Loading & Text Extraction
# ----------------------------------------------------------------------
def load_pdf_documents(docs_dir: str) -> List[Document]:
    """
    Scans the target directory, extracts text page-by-page from PDFs,
    and captures basic metadata (file source, page number).
    """
    pdf_files = glob.glob(os.path.join(docs_dir, "*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in directory: '{docs_dir}'")

    print(f"📄 Found {len(pdf_files)} PDF documents in '{docs_dir}/'. Loading...")
    
    all_docs = []
    for pdf_path in pdf_files:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        # 4. METADATA ENRICHMENT: Adding custom context tags to each raw document page
        for page in pages:
            file_name = os.path.basename(pdf_path)
            doc_type = file_name.replace(".pdf", "").replace("_", " ").title()
            
            # Enrich existing metadata dictionary
            page.metadata.update({
                "file_name": file_name,
                "document_type": doc_type,
                "total_pages": len(pages)
            })
            all_docs.append(page)
            
    print(f"✅ Extracted {len(all_docs)} total pages from PDFs.")
    return all_docs


# ----------------------------------------------------------------------
# Step 2: Intelligent Chunking
# ----------------------------------------------------------------------
def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Splits continuous document text into smaller, overlapping chunks while 
    preserving metadata across every generated split.
    """
    print(f"\n✂️  Chunking documents (Size={CHUNK_SIZE}, Overlap={CHUNK_OVERLAP})...")
    
    # RecursiveCharacterTextSplitter attempts to split along natural boundaries:
    # Double newlines ("\n\n"), single newlines ("\n"), spaces (" "), and characters ("")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Assign a unique chunk ID in metadata for auditability
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{chunk.metadata['file_name']}_p{chunk.metadata.get('page', 0)}_c{idx}"

    print(f"✅ Created {len(chunks)} text chunks.")
    return chunks


# ----------------------------------------------------------------------
# Step 3 & 4: Embedding & FAISS Vector Indexing
# ----------------------------------------------------------------------
def create_and_save_vector_store(chunks: List[Document]):
    """
    Converts text chunks into dense embeddings and indexes them inside 
    a FAISS vector database.
    """
    print(f"\n🧠 Initializing Embedding Model ('{EMBEDDING_MODEL_NAME}')...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    print("⚡ Generating embeddings and building FAISS Index...")
    # FAISS computes vector representations for every chunk and creates a spatial lookup tree
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Save index locally to avoid re-generating embeddings every run
    os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
    vector_store.save_local(VECTOR_STORE_PATH)
    print(f"💾 FAISS Index successfully saved to '{VECTOR_STORE_PATH}/'")
    
    return vector_store, embeddings


# ----------------------------------------------------------------------
# Step 5: Vector Similarity Verification (Sanity Check)
# ----------------------------------------------------------------------
def verify_similarity_search(vector_store: FAISS, query: str):
    """
    Demonstrates 5. VECTOR SIMILARITY by searching the index for 
    chunks geometrically closest to an input query.
    """
    print("\n" + "="*70)
    print(f"🔍 SIMILARITY SEARCH VERIFICATION: '{query}'")
    print("="*70)
    
    # Performs Cosine / L2 distance search in embedding space
    results_with_scores = vector_store.similarity_search_with_score(query, k=2)
    
    for rank, (doc, score) in enumerate(results_with_scores, start=1):
        print(f"\n--- Result #{rank} (L2 Distance Score: {score:.4f}) ---")
        print(f"Source: {doc.metadata.get('file_name')} | Page: {doc.metadata.get('page', 0)}")
        print(f"Chunk ID: {doc.metadata.get('chunk_id')}")
        print(f"Content:\n{doc.page_content.strip()}")


# ----------------------------------------------------------------------
# Pipeline Execution
# ----------------------------------------------------------------------
def main():
    # Execute Pipeline Steps
    raw_docs = load_pdf_documents(DOCS_DIR)
    chunks = chunk_documents(raw_docs)
    vector_store, _ = create_and_save_vector_store(chunks)
    
    # Run a sample query to demonstrate vector retrieval
    verify_similarity_search(
        vector_store=vector_store, 
        query="What are the main requirements and training hours for mentors?"
    )

if __name__ == "__main__":
    main()