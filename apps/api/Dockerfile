# Build from the monorepo root:
#   docker build -f apps/api/Dockerfile -t agentic-doc-api .
FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY packages/core packages/core
COPY apps/api apps/api

# Install only the API workspace member and its deps (no full dev group).
RUN uv sync --package api --no-dev --frozen

ENV PATH="/app/.venv/bin:$PATH"
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV ENVIRONMENT=prod

EXPOSE 8000

CMD ["uvicorn", "agentic_doc_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
