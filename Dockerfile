FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY vial_classifier ./vial_classifier
COPY models/model ./models/model
COPY artifacts/drift_baseline_stats.json ./artifacts/drift_baseline_stats.json

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 5001
CMD ["python", "-m", "vial_classifier.serve"]