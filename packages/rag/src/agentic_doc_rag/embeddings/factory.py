from agentic_doc_rag.config import EmbeddingType, RagSettings
from agentic_doc_rag.embeddings.chroma_default import ChromaDefaultEmbeddings
from agentic_doc_rag.embeddings.protocols import Embeddings
from agentic_doc_rag.embeddings.sentence_transformer import SentenceTransformerEmbeddings


def create_embeddings(settings: RagSettings) -> Embeddings:
    match settings.embedding_type:
        case EmbeddingType.CHROMA_DEFAULT:
            return ChromaDefaultEmbeddings()
        case EmbeddingType.SENTENCE_TRANSFORMERS:
            return SentenceTransformerEmbeddings(settings.embedding_model)