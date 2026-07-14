from agentic_doc_rag.config import RagSettings
from agentic_doc_rag.retrieval.pipeline import PipelineRetriever
from agentic_doc_rag.retrieval.protocols import Retriever
from agentic_doc_rag.retrieval.semantic import SemanticStage
from agentic_doc_rag.vectorstore.factory import create_vector_store


def create_retriever(settings: RagSettings) -> Retriever:
    vectorstore = create_vector_store(settings)
    stages = [SemanticStage(vectorstore)]
    return PipelineRetriever(stages=stages, vectorstore=vectorstore)