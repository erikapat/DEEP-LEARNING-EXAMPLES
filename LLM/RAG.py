import os
import sys
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

#%% retrieval evaluation metrics

# Function to normalize text
def normalize_text(text):
    return " ".join(text.lower().split())

# Hit Rate @ K
def hit_rate_at_k(retrieved_docs, ground_truth_texts, k):
    for doc in retrieved_docs[:k]:
        doc_norm = normalize_text(doc.page_content)
        if any(normalize_text(gt) in doc_norm or doc_norm in normalize_text(gt) for gt in ground_truth_texts):
            return True
    return False

# Precision @ k
def precision_at_k(retrieved_docs, ground_truth_texts, k):
    hits = 0
    for doc in retrieved_docs[:k]:
        doc_norm = normalize_text(doc.page_content)
        if any(normalize_text(gt) in doc_norm or doc_norm in normalize_text(gt) for gt in ground_truth_texts):
            hits += 1
    return hits / k

# Recall @ k
def recall_at_k(retrieved_docs, ground_truth_texts, k):
    matched = set()
    for i, gt in enumerate(ground_truth_texts):
        gt_norm = normalize_text(gt)
        for doc in retrieved_docs[:k]:
            doc_norm = normalize_text(doc.page_content)
            if gt_norm in doc_norm or doc_norm in gt_norm:
                matched.add(i)
                break
    return len(matched) / len(ground_truth_texts) if ground_truth_texts else 0

# F1 @ K
def f1_at_k(precision, recall):
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0


query = "Who is Anna Pávlovna?"

ground_truth_texts = [
    "It was in July, 1805, and the speaker was the well-known Anna Pávlovna Schérer, maid of honor and favorite of the Empress Márya Fëdorovna. With these words she greeted Prince Vasíli Kurágin, a man of high rank and importance, who was the first to arrive at her reception. Anna Pávlovna had had a cough for some days. She was, as she said, suffering from la grippe; grippe being then a new word in St. Petersburg, used only by the elite. All her invitations without exception, written in French, and delivered by a scarlet-liveried footman that morning, ran as follows: “If you have nothing better to do, Count (or Prince), and if the prospect of spending an evening with a poor invalid is not too terrible, I shall be very charmed to see you tonight between 7 and 10—Annette Schérer.”",

    "Anna Pávlovna’s “At Home” was like the former one, only the novelty she offered her guests this time was not Mortemart, but a diplomatist fresh from Berlin with the very latest details of the Emperor Alexander’s visit to Potsdam, and of how the two august friends had pledged themselves in an indissoluble alliance to uphold the cause of justice against the enemy of the human race. Anna Pávlovna received Pierre with a shade of melancholy, evidently relating to the young man’s recent loss by the death of Count Bezúkhov (everyone constantly considered it a duty to assure Pierre that he was greatly afflicted by the death of the father he had hardly known), and her melancholy was just like the august melancholy she showed at the mention of her most august Majesty the Empress Márya Fëdorovna. Pierre felt flattered by this. Anna Pávlovna arranged the different groups in her drawing room with her habitual skill. The large group, in which were",

    "drawing room with her habitual skill. The large group, in which were Prince Vasíli and the generals, had the benefit of the diplomat. Another group was at the tea table. Pierre wished to join the former, but Anna Pávlovna—who was in the excited condition of a commander on a battlefield to whom thousands of new and brilliant ideas occur which there is hardly time to put in action—seeing Pierre, touched his sleeve with her finger, saying:"
]

# ===============================
# Load API key from .env
# ===============================
# .env should contain: OPENAI_API_KEY=sk-xxxx
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    sys.stderr.write(
        "ERROR: OPENAI_API_KEY not found. Create a `.env` file in this directory with a line:\n"
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
def normalize(vectors: np.ndarray) -> np.ndarray:
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


# ===============================
# Load and split documents
# ===============================
text_folder = "RAG files"

if not os.path.isdir(text_folder):
    sys.stderr.write(
        f"WARNING: Folder '{text_folder}' not found. Create it and add .txt files for RAG.\n"
    )

documents = []
if os.path.isdir(text_folder):
    for filename in os.listdir(text_folder):
        if filename.lower().endswith(".txt"):
            file_path = os.path.join(text_folder, filename)
            loader = TextLoader(file_path)
            documents.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = []
for doc in documents:
    for chunk in splitter.split_text(doc.page_content):
        split_docs.append(Document(page_content=chunk))

documents = split_docs

if len(documents) == 0:
    sys.stderr.write(
        f"WARNING: No .txt documents found in '{text_folder}'. The assistant will run but retrieval will be empty.\n"
    )


# ===============================
# Build FAISS index (cosine via normalized IP)
# ===============================
doc_texts = [doc.page_content for doc in documents]
if len(doc_texts) > 0:
    doc_embeddings = embeddings.embed_documents(doc_texts)                 # list[list[float]]
    doc_embeddings = normalize(np.array(doc_embeddings, dtype=np.float32)) # (N, d) float32

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
            print("\n(No documents indexed. Add .txt files to 'RAG files/' and restart for retrieval.)\n")
            messages = [
                {"role": "system", "content": "You are a helpful assistant with no external context."},
                {"role": "user", "content": user_input},
            ]
            response = llm.invoke(messages)
            print(f"\nAssistant: {response.content.strip()}\n")
            continue

        # Embed & normalize query
        query_embedding = embeddings.embed_query(user_input)
        query_embedding = normalize(np.array([query_embedding], dtype=np.float32))  # shape (1, d)

        # Search FAISS
        k_ = min(10, len(documents))
        D, I = index.search(query_embedding, k=k_)

        # Retrieve docs
        docdict = vector_store.docstore._dict  # InMemoryDocstore internal dict
        relevant_docs = [docdict[i] for i in I[0] if i in docdict]

        # Rerank
        reranked_docs, reranked_scores = rerank_with_cross_encoder(user_input, relevant_docs)

        # -----------------------------------------------------------------------------------
        # Evaluate reranked docs using metrics
        top_k_docs = reranked_docs[:k_]  # or change `k` as needed
        precision = precision_at_k(top_k_docs, ground_truth_texts, k=k_)
        recall = recall_at_k(top_k_docs, ground_truth_texts, k=k_)
        f1 = f1_at_k(precision, recall)
        hit = hit_rate_at_k(top_k_docs, ground_truth_texts, k=k_)

        print("\n--- Retrieval Evaluation Metrics ---")
        print(f"Hit@6: {hit}")
        print(f"Precision@6: {precision:.2f}")
        print(f"Recall@6: {recall:.2f}")
        print(f"F1@6: {f1:.2f}")
        print("-" * 40)

        # -- NEW SECTION --

        # get top reranked chunks
        #retrieved_context = "\n\n".join([doc.page_content for doc in reranked_docs[:2]])


        # -----------------------------------------------------------------------------------

        # Build context from top reranked chunks
        top_k_context = 5
        retrieved_context = "\n\n".join([doc.page_content for doc in reranked_docs[:top_k_context]])

        # Show scored chunks (pre-rerank cosine similarities)
        print("\nTop chunks and their cosine similarity scores (pre-rerank):\n")
        for rank, (idx, score) in enumerate(zip(I[0], D[0]), start=1):
            if idx in docdict:
                content = docdict[idx].page_content
                print(f"Chunk {rank}:")
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


