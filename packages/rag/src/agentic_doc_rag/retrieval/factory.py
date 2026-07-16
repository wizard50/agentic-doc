from agentic_doc_rag.config import RagSettings
from agentic_doc_rag.retrieval.filters import MetadataFilterStage
from agentic_doc_rag.retrieval.pipeline import PipelineRetriever
from agentic_doc_rag.retrieval.protocols import Retriever
from agentic_doc_rag.retrieval.rerank import RerankStage, create_reranker
from agentic_doc_rag.retrieval.retrieve import RetrieveStage
from agentic_doc_rag.retrieval.topk import TopKStage
from agentic_doc_rag.sparse.factory import create_sparse_index
from agentic_doc_rag.vectorstore.factory import create_vector_store


def create_retriever(settings: RagSettings) -> Retriever:
    vectorstore = create_vector_store(settings)
    sparse_index = create_sparse_index(settings)
    reranker = create_reranker(settings)
    stages = [
        RetrieveStage(vectorstore, sparse_index),
        MetadataFilterStage(),
        RerankStage(reranker, default_enabled=settings.rerank_enabled),
        TopKStage(),
    ]
    return PipelineRetriever(stages=stages, vectorstore=vectorstore)
