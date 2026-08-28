# Orin

Lightweight symbolic AI + local RAG system designed for Raspberry Pi.

## Features

* Symbolic text encoding/decoding
* Fast pure-Python 384D semantic vectorization
* Cached binary float32 semantic index
* Exact-topic priority + semantic reranking
* Spelling correction
* Concept and alias recognition
* Intent detection
* Local RAG search
* Fact stitching
* Confidence checking
* Compact `knowledge.jsonl`
* Wikipedia learning
* Groq knowledge extraction
* No sentence-transformers
* No NumPy
* No PyTorch
* No Ollama

## Run

```bash
cd ~/Orin
python main_model-3.py --train
python main_model-3.py
```

Ask one question:

```bash
python main_model-3.py --ask "what is artificial intelligence"
```

For Wikipedia learning, set:

```bash
export GROQ_API_KEY="YOUR_GROQ_API_KEY"
```

Exit chat with:

```text
exit
```

MIT License

Copyright (c) 2026 Dhruv Mathummal Panambail

You are free to use, modify, and distribute this project.
Provided as-is without warranty.
