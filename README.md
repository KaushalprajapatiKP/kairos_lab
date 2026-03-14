# kairos_lab

Agentic AI framework that takes a Python ML module and returns verified, GPU-accelerated Triton/CUDA code.

## Status
Early development. Two agents working.

## Agents Built
- Profiler Agent — runs cProfile, identifies top bottlenecks via LLM
- AST Parser Agent — parses bottleneck functions, extracts loop structure, data types, memory patterns

## Run It
```bash
git clone https://github.com/KaushalprajapatiKP/kairos_lab.git
cd kairos_lab
uv venv && source .venv/bin/activate
uv add torch requests
ollama pull mistral
python kairos_lab/agents/profiler.py sample_script.py
python kairos_lab/agents/ast_parser.py sample_script.py
```

## Pipeline
Profiler → AST Parser → Architect → Generator → Verifier → HITL → Doc-Gen