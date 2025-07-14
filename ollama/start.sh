#!/bin/bash

# --- 1. Intel GPU 및 모델 경로 환경 변수 설정 (필수!) ---
export OLLAMA_MODELS=/models
export OLLAMA_NUM_GPU=999
export ZES_ENABLE_SYSMAN=1
export SYCL_CACHE_PERSISTENT=1
export OLLAMA_KEEP_ALIVE=10m
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1

# 사용할 Ollama 실행 파일 경로 변수화
OLLAMA_EXEC="/app/ollama-ipex/ollama"

# --- 2. 자동 모델 다운로드 로직 ---
# 필요한 모델 목록
REQUIRED_MODELS=("mistral:latest")

# Ollama 서버를 백그라운드에서 먼저 실행
echo "Starting Ollama server with Intel IPEX in background..."
echo "Using models from: $OLLAMA_MODELS"
$OLLAMA_EXEC serve &
OLLAMA_PID=$!

# 서버가 준비될 때까지 잠시 대기
sleep 5

echo "Checking for required models..."

# 필요한 모델들이 로컬에 있는지 확인하고, 없으면 다운로드
for model in "${REQUIRED_MODELS[@]}"; do
  if ! $OLLAMA_EXEC list | grep -q "$model"; then
    echo "Model $model not found. Pulling..."
    $OLLAMA_EXEC pull "$model"
  else
    echo "Model $model already exists."
  fi
done

echo "All required models are present."
echo "Bringing Ollama server to foreground..."

# 백그라운드에서 실행되던 Ollama 서버 프로세스를 포그라운드로 가져옴
wait $OLLAMA_PID