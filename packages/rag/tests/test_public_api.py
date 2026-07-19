import agentic_doc_rag as rag


def test_public_api_exports_core_symbols() -> None:
    assert rag.SearchMode.SEMANTIC.value == "semantic"
    assert rag.DocumentChunk is not None
    assert rag.SearchResult is not None
    assert rag.RagSettings is not None
    assert callable(rag.get_rag_settings)
    assert callable(rag.create_retriever)
    assert callable(rag.create_vector_store)
    assert callable(rag.create_sparse_index)
    assert callable(rag.create_embeddings)
    assert callable(rag.register_tracing)
    assert callable(rag.load_eval_dataset)
    assert callable(rag.run_retrieval_eval)
    assert rag.RetrievalRequest is not None
    assert rag.MetadataFilter is not None
    assert rag.IngestSettings is not None
    assert callable(rag.resolve_ingest_settings)
    assert callable(rag.ingest_settings_from_rag)


def test_public_api_run_ingestion_is_lazy_export() -> None:
    assert "run_ingestion" in rag.__all__
    assert callable(rag.run_ingestion)


def test_public_api_all_names_are_importable() -> None:
    for name in rag.__all__:
        assert hasattr(rag, name), name
        assert getattr(rag, name) is not None
