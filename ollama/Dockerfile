FROM ollama/ollama:latest

# Hash 충돌시 우회링크
# RUN sed -i 's|URIs: http://archive.ubuntu.com/ubuntu/|URIs: http://mirror.kakao.com/ubuntu/|g' /etc/apt/sources.list.d/ubuntu.sources && \
#     sed -i 's|URIs: http://security.ubuntu.com/ubuntu/|URIs: http://mirror.kakao.com/ubuntu/|g' /etc/apt/sources.list.d/ubuntu.sources

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository -y ppa:kobuk-team/intel-graphics

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libze-intel-gpu1 \
    libze1 \
    intel-metrics-discovery \
    intel-opencl-icd \
    clinfo \
    intel-gsc \
    && rm -rf /var/lib/apt/lists/*

COPY start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh


RUN cd /tmp && \
    wget https://github.com/ipex-llm/ipex-llm/releases/download/v2.2.0/ollama-ipex-llm-2.2.0-ubuntu.tgz && \
    tar -xzf ollama-ipex-llm-2.2.0-ubuntu.tgz && \
    mkdir -p /app && \
    mv ollama-ipex-llm-2.2.0-ubuntu /app/ollama-ipex && \
    rm ollama-ipex-llm-2.2.0-ubuntu.tgz

# Ollama API Port
EXPOSE 11434

ENTRYPOINT ["usr/local/bin/start.sh"]
CMD [] 