import glob
import os

import chromadb
from langchain_mistralai import MistralAIEmbeddings

import config
from sentence_window import chunk_document


def load_corpus(corpus_dir: str) -> dict:
    docs = {}
    paths = sorted(
        glob.glob(os.path.join(corpus_dir, "*.md"))
        + glob.glob(os.path.join(corpus_dir, "*.txt"))
    )
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            docs[os.path.basename(path)] = f.read()
    return docs


def build_chunks(docs: dict) -> list:
    all_chunks = []
    for source_doc, text in docs.items():
        all_chunks.extend(chunk_document(text, source_doc, window_size=config.WINDOW_SIZE))
    return all_chunks


def embed_and_store(chunks: list) -> None:
    embeddings_model = MistralAIEmbeddings(model=config.MISTRAL_EMBED_MODEL)

    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    collection = client.get_or_create_collection(name=config.COLLECTION_NAME)

    sentences = [c["sentence"] for c in chunks]
    print(f"Embedding {len(sentences)} sentences with {config.MISTRAL_EMBED_MODEL}...")
    vectors = embeddings_model.embed_documents(sentences)

    ids = [c["id"] for c in chunks]
    metadatas = [
        {
            "source_doc": c["source_doc"],
            "section": c["section"],
            "window_text": c["window_text"],
            "sentence_index": c["sentence_index"],
        }
        for c in chunks
    ]

    collection.upsert(
        ids=ids,
        embeddings=vectors,
        documents=sentences,
        metadatas=metadatas,
    )
    print(
        f"Stored {len(ids)} chunks in Chroma collection "
        f"'{config.COLLECTION_NAME}' at '{config.CHROMA_PATH}'"
    )


def main():
    docs = load_corpus(config.CORPUS_DIR)
    if not docs:
        print(f"No .md/.txt files found in {config.CORPUS_DIR}")
        return
    print(f"Loaded {len(docs)} documents: {list(docs.keys())}")

    chunks = build_chunks(docs)
    print(f"Built {len(chunks)} sentence-window chunks (window_size={config.WINDOW_SIZE})")

    embed_and_store(chunks)


if __name__ == "__main__":
    main()
