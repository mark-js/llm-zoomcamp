# LLM Zoomcamp
[DataTalksClub LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp)

### Local LLM using llama.cpp
- google_gemma-4-E2B-it-Q4_K_M.gguf
- google_gemma-4-E4B-it-Q4_K_M.gguf
```
./build/bin/llama-server \
    -m /path/to/model.gguf \
    -c 32768 \
    -t 4 \
    -tb 8 \
    -b 2048 \
    -ub 2048
    --flash-attn on \
    --port 8080 \
    --host 0.0.0.0
```