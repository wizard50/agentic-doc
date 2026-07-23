class IngestSourceNotFoundError(Exception):
    """Raised when the configured ingest source directory does not exist."""


class IngestEmptyCorpusError(Exception):
    """Raised when the ingest source directory contains no indexable markdown files."""
