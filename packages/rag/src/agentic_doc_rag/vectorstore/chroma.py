from pathlib import Path

import chromadb

from agentic_doc_rag.models import DocumentChunk, SearchResult
from agentic_doc_rag.observability.tracing import (
    get_tracer,
    mark_chain_span,
    mark_retriever_span,
    record_search_results,
)


class ChromaVectorStore:
    def __init__(self, persist_dir: Path, collection_name: str) -> None:
        self._persist_dir = persist_dir
        self._collection_name = collection_name

        self._client = chromadb.PersistentClient(str(self._persist_dir))
        self._collection = self._client.get_or_create_collection(self._collection_name)

    def upsert(self, chunks: list[DocumentChunk]) -> None:
        with get_tracer(__name__).start_as_current_span("vectorstore.upsert") as span:
            mark_chain_span(span)
            span.set_attribute("collection_name", self._collection_name)
            span.set_attribute("chunk_count", len(chunks))

            if not chunks:
                return

            self._collection.upsert(
                ids=[c.id for c in chunks],
                documents=[c.text for c in chunks],
                metadatas=[c.metadata for c in chunks],
            )

    def search(self, query: str, k: int) -> list[SearchResult]:
        with get_tracer(__name__).start_as_current_span("vectorstore.search") as span:
            span.set_attribute("collection_name", self._collection_name)
            mark_retriever_span(span, query=query, top_k=k)

            results: chromadb.QueryResult = self._collection.query(query_texts=[query], n_results=k)

            ids = results["ids"][0] if results["ids"] else []
            documents = results["documents"][0] if results["documents"] else []
            metadatas = results["metadatas"][0] if results["metadatas"] else []
            distances = results["distances"][0] if results["distances"] else []

            search_results = [
                SearchResult(
                    chunk=DocumentChunk(
                        id=id_,
                        text=document,
                        metadata=dict(metadata) if metadata else {},
                    ),
                    score=distance,
                )
                for id_, document, metadata, distance in zip(
                    ids, documents, metadatas, distances, strict=True
                )
            ]
            record_search_results(span, search_results)
            return search_results

    def delete(self, ids: list[str]) -> None:
        self._collection.delete(ids=ids)

    def count(self) -> int:
        return self._collection.count()
