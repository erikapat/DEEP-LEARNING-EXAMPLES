import os
import sys
import json
import numpy as np
import torch
import faiss

from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

# --- Modern LangChain split packages ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore

# PDF extraction (pdfminer.six)
try:
    from pdfminer.high_level import extract_text as pdf_extract_text
except Exception as e:
    pdf_extract_text = None


# ===============================
# Retrieval evaluation metrics
# ===============================
def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())

def hit_rate_at_k(retrieved_docs, ground_truth_texts, k: int) -> bool:
    for doc in retrieved_docs[:k]:
        doc_norm = normalize_text(doc.page_content)
        if any(normalize_text(gt) in doc_norm or doc_norm in normalize_text(gt) for gt in ground_truth_texts):
            return True
    return False

def precision_at_k(retrieved_docs, ground_truth_texts, k: int) -> float:
    hits = 0
    for doc in retrieved_docs[:k]:
        doc_norm = normalize_text(doc.page_content)
        if any(normalize_text(gt) in doc_norm or doc_norm in normalize_text(gt) for gt in ground_truth_texts):
            hits += 1
    return hits / max(1, k)

def recall_at_k(retrieved_docs, ground_truth_texts, k: int) -> float:
    if not ground_truth_texts:
        return 0.0
    matched = set()
    for i, gt in enumerate(ground_truth_texts):
        gt_norm = normalize_text(gt)
        for doc in retrieved_docs[:k]:
            doc_norm = normalize_text(doc.page_content)
            if gt_norm in doc_norm or doc_norm in gt_norm:
                matched.add(i)
                break
    return len(matched) / len(ground_truth_texts)

def f1_at_k(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


# ===============================
# Ground truth loader
# ===============================
def load_ground_truth_from_file(path: str = "ground_truth.json") -> dict:
    """
    Load a dict mapping normalized queries -> list of acceptable ground-truth strings.
    Example ground_truth.json:
    {
      "who is anna pávlovna?": [
        "Anna Pavlovna Scherer is maid of honour and confidante to Empress Maria Feodorovna."
      ],
      "what information is required on an auto claim form?": [
        "date of loss", "vehicle year make model", "VIN", "police report number", "location of accident"
      ]
    }
    """
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {normalize_text(k): v for (k, v) in data.items()}
        except Exception as e:
            sys.stderr.write(f"WARNING: Failed to load {path}: {e}\n")
    return {}

# Inline fallback ground truth (edit as you like)
GROUND_TRUTH_INLINE = {
    # ========== Auto Claim Form (OneMain / Yosemite) ==========
    "what information is required on an auto claim form?": [
        "date of loss", "time of loss", "location of loss",
        "policy number", "claim number",
        "insured name", "phone number", "email",
        "vehicle year make model", "vin",
        "driver’s license", "license number",
        "police report number", "reporting agency",
        "description of accident", "photos of damage",
        "other party information", "repair shop information",
        "signature", "date"
    ],
    "list fields on the auto claim form": [
        "date of loss", "policy number", "vin", "police report number",
        "description of accident", "vehicle year make model"
    ],

    # ========== Kansas Auto Shopper’s Guide ==========
    # (Kansas commonly: 25/50/25 + PIP; UM/UIM required.)
    "what are the minimum auto liability limits in kansas?": [
        "25/50/25",                          # 25k BI per person / 50k BI per accident / 25k PD
        "bodily injury", "property damage",
        "personal injury protection", "pip",
        "uninsured motorist", "underinsured motorist"
    ],
    "is uninsured motorist coverage required in kansas?": [
        "uninsured motorist coverage is required",
        "underinsured motorist coverage is required"
    ],
    "what is pip in kansas auto insurance?": [
        "personal injury protection", "pip benefits", "no-fault"
    ],

    # ========== Shelter HO-4 Renters (OK) ==========
    "what does an ho-4 renters policy generally cover?": [
        "personal property", "loss of use", "personal liability",
        "medical payments to others"
    ],
    "what does loss of use mean in renters insurance?": [
        "additional living expense", "fair rental value"
    ],
    "what personal liability covers in ho-4?": [
        "bodily injury", "property damage", "defense costs"
    ],

    # ========== Maryland Homeowners Insurance Guide ==========
    "what perils are typically excluded in homeowners policies?": [
        "flood is not covered",
        "earthquake is excluded",
        "wear and tear", "maintenance",
        "neglect", "war", "nuclear hazard"
    ],
    "how do i file a homeowners claim in maryland?": [
        "contact your insurer", "protect property from further damage",
        "keep receipts", "document the damage", "proof of loss"
    ]
}
GROUND_TRUTH = {**GROUND_TRUTH_INLINE, **load_ground_truth_from_file()}

def get_ground_truth_texts(query: str):
    return GROUND_TRUTH.get(normalize_text(query))


# ===============================
# Load API key from .env
# ===============================
# .env should contain: OPENAI_API_KEY=sk-xxxx
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    sys.stderr.write(
        "ERROR: OPENAI_API_KEY not found. Create a `.env` file with:\n"
        "OPENAI_API_KEY=sk-yourkey\n"
    )
    sys.exit(1)


# ===============================
# Initialize models
# ===============================
# OpenAI chat model (modern client via langchain-openai)
llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0.3)

# Cross-encoder for reranking (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2", device=device)

# Embeddings
embeddings = OpenAIEmbeddings(api_key=api_key)


# ===============================
# Helpers
# ===============================
def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize a 2D array of vectors, returned as float32."""
    vectors = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return vectors / norms

def rerank_with_cross_encoder(query: str, relevant_docs):
    """Rerank docs using a cross-encoder; returns (ranked_docs, ranked_scores)."""
    if not relevant_docs:
        return [], []
    pairs = [(query, doc.page_content) for doc in relevant_docs]
    scores = cross_encoder.predict(pairs)  # higher is better
    ranked_indices = np.argsort(scores)[::-1]
    ranked_docs = [relevant_docs[i] for i in ranked_indices]
    ranked_scores = [float(scores[i]) for i in ranked_indices]
    return ranked_docs, ranked_scores

def extract_text_from_pdf(path: str) -> str:
    """Extract text from a PDF. Requires pdfminer.six."""
    if pdf_extract_text is None:
        raise RuntimeError(
            "pdfminer.six not available. Install it with: pip install pdfminer.six"
        )
    try:
        return pdf_extract_text(path)
    except Exception as e:
        sys.stderr.write(f"WARNING: Could not extract PDF text from {path}: {e}\n")
        return ""


# ===============================
# Load and split documents (PDF + TXT)
# ===============================
text_folder = "RAG files"

if not os.path.isdir(text_folder):
    sys.stderr.write(
        f"WARNING: Folder '{text_folder}' not found. Create it and add .pdf or .txt files for RAG.\n"
    )

documents = []
if os.path.isdir(text_folder):
    for filename in os.listdir(text_folder):
        fpath = os.path.join(text_folder, filename)
        low = filename.lower()

        if low.endswith(".txt"):
            # Use TextLoader to read the file
            try:
                loader = TextLoader(fpath, encoding="utf-8")
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                sys.stderr.write(f"WARNING: Failed to load {fpath}: {e}\n")

        elif low.endswith(".pdf"):
            # Extract text from PDF directly (no intermediate .txt required)
            text = extract_text_from_pdf(fpath)
            if text.strip():
                documents.append(Document(page_content=text, metadata={"source": filename}))
            else:
                sys.stderr.write(f"WARNING: Empty text extracted from {fpath}\n")

        else:
            # Ignore other file types
            continue

# Chunking
# For policies/guides, slightly larger chunks improve recall; overlap helps stitching references
splitter = RecursiveCharacterTextSplitter(chunk_size=1400, chunk_overlap=160)
split_docs = []
for doc in documents:
    for chunk in splitter.split_text(doc.page_content):
        split_docs.append(Document(page_content=chunk, metadata=doc.metadata if hasattr(doc, "metadata") else {}))

documents = split_docs

if len(documents) == 0:
    sys.stderr.write(
        f"WARNING: No text available after loading. Ensure 'RAG files/' contains readable PDFs/TXTs.\n"
    )


# ===============================
# Build FAISS index (cosine via normalized IP)
# ===============================
doc_texts = [doc.page_content for doc in documents]
if len(doc_texts) > 0:
    doc_embeddings = embeddings.embed_documents(doc_texts)                       # list[list[float]]
    doc_embeddings = l2_normalize(np.array(doc_embeddings, dtype=np.float32))    # (N, d) float32

    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # inner product == cosine if normalized
    index.add(doc_embeddings)

    docstore_dict = {i: doc for i, doc in enumerate(documents)}
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(docstore_dict),
        index_to_docstore_id={i: i for i in range(len(documents))}
    )
else:
    index = None
    vector_store = None


# ===============================
# Main loop
# ===============================
def main():
    print("Welcome to the RAG Assistant. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Exiting…")
            break

        # If no documents, just answer directly
        if index is None or vector_store is None or len(documents) == 0:
            print("\n(No documents indexed. Add PDFs/TXTs to 'RAG files/' and restart for retrieval.)\n")
            messages = [
                {"role": "system", "content": "You are a helpful assistant with no external context."},
                {"role": "user", "content": user_input},
            ]
            response = llm.invoke(messages)
            print(f"\nAssistant: {response.content.strip()}\n")
            continue

        # Embed & normalize query
        query_embedding = embeddings.embed_query(user_input)
        query_embedding = l2_normalize(np.array([query_embedding], dtype=np.float32))  # shape (1, d)

        # Search FAISS
        k_ = min(10, len(documents))
        D, I = index.search(query_embedding, k=k_)

        # Retrieve docs
        docdict = vector_store.docstore._dict  # InMemoryDocstore internal dict
        relevant_docs = [docdict[i] for i in I[0] if i in docdict]

        # Rerank
        reranked_docs, reranked_scores = rerank_with_cross_encoder(user_input, relevant_docs)

        # ---------------- Metrics (only if ground truth exists) ----------------
        ground_truth_texts = get_ground_truth_texts(user_input)
        if ground_truth_texts:
            top_k_eval = min(k_, len(reranked_docs))
            top_k_docs = reranked_docs[:top_k_eval]

            precision = precision_at_k(top_k_docs, ground_truth_texts, k=top_k_eval)
            recall = recall_at_k(top_k_docs, ground_truth_texts, k=top_k_eval)
            f1 = f1_at_k(precision, recall)
            hit = hit_rate_at_k(top_k_docs, ground_truth_texts, k=top_k_eval)

            print("\n--- Retrieval Evaluation Metrics ---")
            print(f"Hit@{top_k_eval}: {hit}")
            print(f"Precision@{top_k_eval}: {precision:.2f}")
            print(f"Recall@{top_k_eval}: {recall:.2f}")
            print(f"F1@{top_k_eval}: {f1:.2f}")
            print("-" * 40)
        else:
            print("\n(No ground truth defined for this query; skipping retrieval metrics.)\n")
        # ----------------------------------------------------------------------

        # Build context from top reranked chunks
        top_k_context = 5
        retrieved_context = "\n\n".join([doc.page_content for doc in reranked_docs[:top_k_context]])

        # Show scored chunks (pre-rerank cosine similarities)
        print("\nTop chunks and their cosine similarity scores (pre-rerank):\n")
        for rank, (idx, score) in enumerate(zip(I[0], D[0]), start=1):
            if idx in docdict:
                meta = docdict[idx].metadata if hasattr(docdict[idx], "metadata") else {}
                source = meta.get("source", "unknown")
                content = docdict[idx].page_content
                print(f"Chunk {rank} (source: {source}):")
                print(f"Cosine similarity: {float(score):.4f}")
                print(f"Content:\n{content}\n{'-'*40}")

        # System prompt with retrieved context
        system_prompt = (
            "You are a helpful assistant. "
            "Use ONLY the following knowledge base context to answer the user. "
            "If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{retrieved_context}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        response = llm.invoke(messages)
        print(f"\nAssistant: {response.content.strip()}\n")


if __name__ == "__main__":
    main()
