# The Fusha layer for Saleem: foreign and Saudi dialect words -> Modern Standard Arabic (src/merged_api.py).
# CPU only. The two model folders are not in the image: mount them at /models (see docs/DEPLOY_SALEEM.md).
# The language model (gemma-4-12B-it) is Saleem's own server, reached at FUSHA_LLM_URL.
FROM python:3.11-slim

WORKDIR /app

COPY deploy/requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src ./src
COPY data ./data

ENV FUSHA_TAGGER_DIR=/models/tagger_merged \
    FOREIGN_DIR=/models/foreign_to_msa \
    FUSHA_LLM_API=openai \
    FUSHA_LLM_URL=http://gemma:8000 \
    FUSHA_LLM_MODEL=gemma-4-12B-it \
    PYTHONIOENCODING=utf-8

EXPOSE 8010
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8010/health')"
CMD ["python", "-m", "uvicorn", "merged_api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8010"]
