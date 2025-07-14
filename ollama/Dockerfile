FROM ollama/ollama:latest

COPY start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

# Ollama API Port
EXPOSE 11434

ENTRYPOINT ["usr/local/bin/start.sh"]
CMD [] 