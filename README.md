# Kairos Labs

Agentic AI framework that takes a Python ML module and returns verified, production-grade GPU-accelerated Triton/CUDA code.

## Pipeline
```
Dependency Resolver → Project Graph Builder → Profiler → Memory Agent → Dataflow Agent
                                                                        ↓
                                              Learning Agent ← Doc-Gen ← HITL ← Diff Viewer
                                                                        ↑
                               AST Parser → Architect → Generator → Critic Agents → Verifier
```

## Example Output
Feed it this:
```python
def inefficient_loop(tensor):
    result = []
    for i in range(tensor.size(0)):
        row_sum = 0
        for j in range(tensor.size(1)):
            row_sum += tensor[i][j].item()
        result.append(row_sum)
    return result
```
Get this:
```
Dependency Resolver → torch available, safe to proceed
Project Graph Builder → 3 functions found, inefficient_loop is leaf node
Profiler             → inefficient_loop is 75% of runtime
AST Parser           → nested loops, element-wise tensor access, depth 2, line 13-20
Architect            → TRITON, high confidence
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
uv add torch networkx pydantic asttokens requests
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

## Run the Pipeline

### Individual agents
```bash
python kairos_lab/agents/dependency_resolver.py sample_script.py
python kairos_lab/agents/project_graph_builder.py sample_script.py
python kairos_lab/agents/profiler.py sample_script.py
python kairos_lab/agents/ast_parser.py sample_script.py
python kairos_lab/agents/architect.py sample_script.py
python kairos_lab/agents/generator.py sample_script.py
```

### Full pipeline (runs all 6 agents in order)
```bash
python kairos_lab/agents/generator.py sample_script.py
```

## Tech Stack
- **Orchestration** — LangGraph
- **Parsing** — Python ast + asttokens
- **Graph Analysis** — NetworkX
- **LLMs** — Ollama (deepseek-r1:7b + mistral)
- **Validation** — Pydantic
- **API** — FastAPI
- **Frontend** — Chainlit
- **CLI** — Click
- **Optimization targets** — OpenAI Triton, Numba

## Agent Status
- [x] Dependency Resolver
- [x] Project Graph Builder
- [x] Profiler
- [x] AST Parser
- [x] Architect
- [x] Generator
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
- Ollama running locally

## Status
Active development. Solo technical founder building in public from Vadodara, India.

## License
MIT