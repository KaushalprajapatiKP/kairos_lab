# Kairos Labs

Agentic AI framework that takes a complete Python ML project and returns verified, production-grade GPU-accelerated Triton/CUDA code.

## Full Pipeline Architecture
```
User Python Project (single file or multi-file module)
                        │
                        ▼
          ┌─────────────────────────┐
          │   Dependency Resolver   │  Scans all project files recursively,
          │                         │  finds missing packages before pipeline runs
          └────────────┬────────────┘
                       │
                       ▼
          ┌─────────────────────────┐
          │  Project Graph Builder  │  Maps all functions, call graph,
          │                         │  entry points, leaf functions, call sites
          └────────────┬────────────┘
                       │
                       ▼
          ┌─────────────────────────┐
          │      Profiler Agent     │  Runs cProfile, identifies top 3
          │                         │  runtime bottlenecks via LLM analysis
          └────────────┬────────────┘
                       │
           ┌───────────┴───────────┐
           │     PARALLEL          │
           ▼                       ▼
┌──────────────────┐    ┌──────────────────────┐
│   Memory Agent   │    │    Dataflow Agent     │
│                  │    │                       │
│ tracemalloc on   │    │ Torch FX dataflow     │
│ original code    │    │ analysis              │
└────────┬─────────┘    └──────────┬────────────┘
         └───────────┬─────────────┘
                     │
                     ▼
          ┌─────────────────────────┐
          │      AST Parser         │  Parses bottleneck functions across
          │                         │  all project files, maps structure
          └────────────┬────────────┘
                       │
                       ▼
          ┌─────────────────────────┐
          │     Architect Agent     │  Decides strategy per function:
          │                         │  Triton / Numba / CUDA
          └────────────┬────────────┘
                       │
                       ▼
          ┌─────────────────────────┐
          │     Generator Agent     │  Writes optimized GPU code
          │                         │  via LLM (deepseek-r1:7b)
          └────────────┬────────────┘
                       │
           ┌───────────┴───────────┐
           │     PARALLEL          │
           ▼                       ▼
┌──────────────────┐    ┌──────────────────────┐
│ Performance      │    │  Correctness Critic   │
│ Critic           │    │                       │
│                  │    │ Validates output is   │
│ Checks speedup   │    │ mathematically correct│
│ claim is real    │    │                       │
└────────┬─────────┘    └──────────┬────────────┘
         └───────────┬─────────────┘
                     │
                     ▼
          ┌─────────────────────────┐
          │      Verifier Agent     │  Runs on real GPU, benchmarks
          │                         │  speed + correctness. Loops back
          │                         │  to Generator if not faster.
          └────────────┬────────────┘
                       │
                       ▼
          ┌─────────────────────────┐
          │      Diff Viewer        │  Shows exactly what changed
          │                         │  and why, line by line
          └────────────┬────────────┘
                       │
                       ▼
          ┌─────────────────────────┐
          │   Human-in-the-Loop     │  Engineer reviews edge cases
          │                         │  before final output
          └────────────┬────────────┘
                       │
                       ▼
          ┌─────────────────────────┐
          │   Documentation Agent   │  Auto-generates inline comments,
          │                         │  API docs, benchmark report,
          │                         │  migration notes
          └────────────┬────────────┘
                       │
                       ▼
          ┌─────────────────────────┐
          │      Learning Agent     │  Stores successful patterns in
          │                         │  FAISS vector DB for future jobs
          └────────────┬────────────┘
                       │
                       ▼
              Optimization Knowledge Base
```

## Example — What It Finds Automatically

Feed it this multi-file ML project:
```
sample_project/
    main.py
    trainer.py
    model.py
    data_loader.py
```

Pipeline output:
```
Dependency Resolver  → torch, numpy available across 4 files
Project Graph        → 15 functions found, slow_relu is leaf node called by forward
Profiler             → slow_relu 40% runtime, compute_accuracy 2%, compute_loss 1%
AST Parser           → nested loops, element-wise ops, depth 2, lines 44-58
Architect            → TRITON high confidence for slow_relu
Generator            → Triton kernel produced
```

## Setup

### 1. Clone
```bash
git clone https://github.com/KaushalprajapatiKP/kairos_lab.git
cd kairos_lab
```

### 2. Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Environment and dependencies
```bash
uv venv
source .venv/bin/activate
uv add torch networkx pydantic asttokens requests networkx numba
```

### 4. Install Ollama

**Mac:**
```bash
brew install ollama
```
Or download from https://ollama.com/download

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 5. Pull models
```bash
ollama pull deepseek-r1:7b
ollama pull mistral
```

### 6. Start Ollama
```bash
ollama serve
```

## Run

### Test with single file
```bash
python kairos_lab/agents/generator.py sample_script.py
```

### Test with multi-file project
```bash
python kairos_lab/agents/generator.py sample_project/main.py
```

### Run individual agents
```bash
python kairos_lab/agents/dependency_resolver.py sample_project/main.py
python kairos_lab/agents/project_graph_builder.py sample_project/main.py
python kairos_lab/agents/profiler.py sample_project/main.py
python kairos_lab/agents/ast_parser.py sample_project/main.py
python kairos_lab/agents/architect.py sample_project/main.py
python kairos_lab/agents/generator.py sample_project/main.py
```

## Tech Stack
| Layer | Tool |
|-------|------|
| Orchestration | LangGraph |
| Parsing | Python ast + asttokens |
| Graph Analysis | NetworkX |
| LLMs | Ollama — deepseek-r1:7b + mistral |
| Validation | Pydantic |
| API | FastAPI |
| Frontend | Chainlit |
| CLI | Click |
| Knowledge Base | FAISS + sentence-transformers |
| Optimization | OpenAI Triton + Numba |

## Agent Status
- [x] Dependency Resolver — recursive multi-file scanning
- [x] Project Graph Builder — call graph + call sites with line numbers
- [x] Profiler — cProfile + LLM bottleneck analysis
- [x] AST Parser — multi-file, asttokens line mapping
- [x] Architect — Triton / Numba / CUDA strategy decision
- [x] Generator — LLM code generation (deepseek-r1:7b)
- [ ] Memory Agent
- [ ] Dataflow Agent
- [ ] Verifier
- [ ] Performance Critic
- [ ] Correctness Critic
- [ ] Diff Viewer
- [ ] Human-in-the-Loop
- [ ] Documentation Agent
- [ ] Learning Agent

## Requirements
- Python 3.10+
- MacOS / Linux
- 8GB+ RAM
- Ollama running locally with deepseek-r1:7b and mistral pulled

## Status
Active development. Solo technical founder building in public from Vadodara, India.

## License
MIT