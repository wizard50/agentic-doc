from agentic_doc_rag.retrieval import (
    MetadataFilterStage,
    PipelineRetriever,
    RerankStage,
    RetrieveStage,
    TopKStage,
)
from agentic_doc_rag.sparse.bm25 import Bm25Index
from agentic_doc_rag.vectorstore.chroma import ChromaVectorStore

from .fakes import PassthroughReranker, StubVectorStore, UnusedSparseIndex


def semantic_pipeline_retriever(store: StubVectorStore) -> PipelineRetriever:
    return PipelineRetriever(
        stages=[
            RetrieveStage(store, UnusedSparseIndex()),
            MetadataFilterStage(),
            RerankStage(PassthroughReranker(), default_enabled=False),
            TopKStage(),
        ],
        vectorstore=store,
    )


def indexed_pipeline_retriever(store: ChromaVectorStore, sparse: Bm25Index) -> PipelineRetriever:
    return PipelineRetriever(
        stages=[
            RetrieveStage(store, sparse),
            MetadataFilterStage(),
            RerankStage(PassthroughReranker(), default_enabled=False),
            TopKStage(),
        ],
        vectorstore=store,
    )