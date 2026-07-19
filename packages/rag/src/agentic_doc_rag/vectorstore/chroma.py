from pathlib import Path
from typing import Any, cast

import chromadb

from agentic_doc_rag.embeddings.errors import EmbeddingModelMismatchError
from agentic_doc_rag.embeddings.protocols import Embeddings
from agentic_doc_rag.models import DocumentChunk, SearchResult
from agentic_doc_rag.observability.tracing import (
    get_tracer,
    mark_chain_span,
    mark_retriever_span,
    record_search_results,
)

_EMBEDDING_MODEL_KEY = "embedding_model"
_EMBEDDING_DIMENSIONS_KEY = "embedding_dimensions"


class ChromaVectorStore:
    def __init__(
        self,
        persist_dir: Path,
        collection_name: str,
        embeddings: Embeddings,
    ) -> None:
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._embeddings = embeddings

        self._client = chromadb.PersistentClient(str(self._persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={
                _EMBEDDING_MODEL_KEY: embeddings.model_name,
                _EMBEDDING_DIMENSIONS_KEY: embeddings.dimensions,
            },
        )
        self._validate_embedding_metadata()

    def upsert(self, chunks: list[DocumentChunk]) -> None:
        with get_tracer(__name__).start_as_current_span("vectorstore.upsert") as span:
            mark_chain_span(span)
            span.set_attribute("collection_name", self._collection_name)
            span.set_attribute("chunk_count", len(chunks))
            span.set_attribute("embedding_model", self._embeddings.model_name)

            if not chunks:
                return

            vectors = self._embeddings.embed_documents([chunk.text for chunk in chunks])
            self._collection.upsert(
                ids=[chunk.id for chunk in chunks],
                embeddings=cast(Any, vectors),
                documents=[chunk.text for chunk in chunks],
                metadatas=[chunk.metadata for chunk in chunks],
            )

    def search(self, query: str, k: int) -> list[SearchResult]:
        with get_tracer(__name__).start_as_current_span("vectorstore.search") as span:
            span.set_attribute("collection_name", self._collection_name)
            span.set_attribute("embedding_model", self._embeddings.model_name)
            mark_retriever_span(span, query=query, top_k=k)

            query_vector = self._embeddings.embed_query(query)
            results: chromadb.QueryResult = self._collection.query(
                query_embeddings=[query_vector],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )

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

    def _validate_embedding_metadata(self) -> None:
        metadata = self._collection.metadata or {}
        stored_model = metadata.get(_EMBEDDING_MODEL_KEY)
        if stored_model is None:
            return
        if stored_model != self._embeddings.model_name:
            msg = (
                f"Collection '{self._collection_name}' was built with embedding model "
                f"'{stored_model}', but settings specify '{self._embeddings.model_name}'. "
                "Re-run ingest after changing embedding settings."
            )
            raise EmbeddingModelMismatchError(msg)
