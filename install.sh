#!/bin/bash
set -e

echo "Installing Legal Paraphrase Mining project..."
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed."
    echo "Please install uv first: https://docs.astral.sh/uv/"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed."
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi

# Install Python dependencies
echo "1. Installing Python dependencies..."
uv sync

# Download spaCy models
echo ""
echo "2. Downloading spaCy language models..."
uv run python -m spacy download en_core_web_sm
uv run python -m spacy download de_core_news_sm

# Start Qdrant
echo ""
echo "3. Starting Qdrant vector database..."
docker compose up -d

# Wait for Qdrant to be ready
echo ""
echo "4. Waiting for Qdrant to be ready..."
sleep 5
until curl -s http://localhost:6333/health > /dev/null 2>&1; do
    echo "   Waiting for Qdrant..."
    sleep 2
done

echo ""
echo "✓ Installation complete!"
echo ""
echo "Next steps:"
echo "  - Run paraphrase mining: uv run python -m src.tiny_test"
echo "  - Generate synthetic pairs: uv run python -m src.synthetic_pair_creation"
echo ""
echo "To stop Qdrant: docker compose down"
