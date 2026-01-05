#!/bin/bash

# Adaptive RAG 실행 예시 스크립트

echo "=========================================="
echo "Adaptive RAG Application - Run Examples"
echo "=========================================="
echo ""

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# vLLM 서버 확인
echo -e "${YELLOW}[1] Checking vLLM server...${NC}"
if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo -e "${GREEN}✓ vLLM server is running${NC}"
else
    echo "✗ vLLM server is not running"
    echo ""
    echo "Please start vLLM server first:"
    echo "  pip install vllm"
    echo "  python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-chat-hf --port 8000"
    echo ""
    echo "Or use any other model supported by vLLM"
    exit 1
fi

echo ""
echo -e "${YELLOW}[2] Running component tests...${NC}"
python test_components.py

echo ""
echo -e "${YELLOW}[3] Running evaluation mode (10 samples)...${NC}"
python main.py \
    --mode eval \
    --num-samples 10 \
    --output results_sample.json \
    --vllm-url http://localhost:8000/v1 \
    --vllm-model meta-llama/Llama-2-7b-chat-hf

echo ""
echo -e "${GREEN}=========================================="
echo "Evaluation complete!"
echo "==========================================${NC}"
echo ""
echo "Results saved to: results_sample.json"
echo ""
echo "To run interactive mode:"
echo "  python main.py --mode interactive --num-samples 10"
echo ""
echo "To run full evaluation (100 samples):"
echo "  python main.py --mode eval --num-samples 100 --output results.json"
