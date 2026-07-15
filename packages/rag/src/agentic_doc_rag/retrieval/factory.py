from agentic_doc_rag.config import RagSettings
from agentic_doc_rag.retrieval.pipeline import PipelineRetriever
from agentic_doc_rag.retrieval.protocols import Retriever
from agentic_doc_rag.retrieval.retrieve import RetrieveStage
from agentic_doc_rag.sparse.factory import create_sparse_index
from agentic_doc_rag.vectorstore.factory import create_vector_store


def create_retriever(settings: RagSettings) -> Retriever:
    vectorstore = create_vector_store(settings)
    sparse_index = create_sparse_index(settings)
    stages = [RetrieveStage(vectorstore, sparse_index)]
    return PipelineRetriever(stages=stages, vectorstore=vectorstore)