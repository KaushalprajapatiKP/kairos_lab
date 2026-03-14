# Kairos Labs

Agentic AI framework that takes a Python ML module and returns verified, GPU-accelerated Triton/CUDA code.

## What It Does
Runs a pipeline of AI agents that profile your Python ML code, parse its structure, and decide the best GPU acceleration strategy — automatically.

## Pipeline
```
Profiler → AST Parser → Architect → Generator → Verifier → HITL → Doc-Gen
```
First three agents are working. Generator, Verifier, and HITL are in active development.

## Example Output
Feed it this:
```python
def inefficient_loop(tensor):
    result = []
    for i in range(tensor.shape[0]):
        row_sum = 0
        for j in range(tensor.shape[1]):
            row_sum += tensor[i][j].item()
        result.append(row_sum)
    return result
```
Get this:
```
Profiler    → inefficient_loop is 75% of runtime
AST Parser  → nested loops, element-wise tensor access, depth 2
Architect   → TRITON, high confidence — parallelize across tensor dimensions
```

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/KaushalprajapatiKP/kairos_lab.git
cd kairos_lab
```

### 2. Install uv (fast Python package manager)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Create environment and install dependencies
```bash
uv venv
source .venv/bin/activate
uv add torch requests
```

### 4. Install Ollama (local LLM — free, no API key needed)

**Mac:**
```bash
brew install ollama
```
Or download directly from https://ollama.com/download

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download the installer from https://ollama.com/download

### 5. Pull the Mistral model
```bash
ollama pull mistral
```
This downloads ~4GB. One time only.

### 6. Start Ollama
```bash
ollama serve
```
Keep this running in a separate terminal.

## Run the Pipeline

### Profiler Agent
```bash
python kairos_lab/agents/profiler.py sample_script.py
```

### AST Parser Agent
```bash
python kairos_lab/agents/ast_parser.py sample_script.py
```

### Architect Agent (runs full pipeline — Profiler + AST + Strategy)
```bash
python kairos_lab/agents/architect.py sample_script.py
```

## Requirements
- Python 3.10+
- PyTorch
- Ollama running locally with Mistral pulled
- Mac M1/M2/M4, Linux, or Windows with 8GB+ RAM

## Status
Active development. 

- [x] Profiler Agent
- [x] AST Parser Agent  
- [x] Architect Agent
- [ ] Generator Agent
- [ ] Verifier Agent
- [ ] Human-in-the-Loop
- [ ] Documentation Agent

## Contributing
Early stage. If you want to follow along or contribute, open an issue or watch the repo.

## License
MIT