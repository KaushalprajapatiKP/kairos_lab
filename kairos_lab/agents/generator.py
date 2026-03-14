import requests
import json
import sys
sys.path.append(".")

from pathlib import Path
from kairos_lab.models import ArchitectDecision, GeneratorOutput

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "deepseek-r1:7b"


def get_function_source(script_path: str, function_name: str) -> str:
    import ast
    source = Path(script_path).read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            lines = source.split('\n')
            start = node.lineno - 1
            end = node.end_lineno
            return '\n'.join(lines[start:end])
    return ""


def generate_optimized_code(function_source: str, decision: ArchitectDecision) -> str:
    strategy = decision.strategy
    action = decision.action
    reason = decision.reason
    func_name = decision.function

    if strategy == "triton":
        strategy_instructions = """
Write a complete OpenAI Triton kernel that replaces this function.
Include:
1. The @triton.jit kernel function
2. A Python wrapper function with the same name and signature as the original
3. All necessary imports (triton, triton.language as tl, torch)
The wrapper must be a drop-in replacement — same inputs, same outputs.
"""
    elif strategy == "numba":
        strategy_instructions = """
Write an optimized version using Numba and NumPy that replaces this function.
Include:
1. @numba.jit(nopython=True) decorator
2. Replace Python lists with numpy arrays
3. All necessary imports (numba, numpy)
The function must be a drop-in replacement — same inputs, same outputs.
"""
    else:
        strategy_instructions = """
Write an optimized PyTorch version that replaces this function.
Use torch operations to fuse and vectorize — avoid Python loops entirely.
Include all necessary imports.
The function must be a drop-in replacement — same inputs, same outputs.
"""

    prompt = f"""You are an expert GPU performance engineer.

ORIGINAL FUNCTION:
```python
{function_source}
```

ANALYSIS:
- Strategy: {strategy.upper()}
- Reason: {reason}
- Action: {action}

TASK:
{strategy_instructions}

RULES:
- Output ONLY the optimized Python code
- No explanations before or after
- No markdown formatting or backticks
- Code must be complete and runnable
- Include all imports at the top
- Add brief inline comments on key optimizations

Generate optimized code now:"""

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }

    print(f"[Generator Agent] Calling LLM for {func_name} → {strategy.upper()}...")
    response = requests.post(OLLAMA_URL, json=payload)
    result = response.json()
    return result["response"]


def run_generator(script_path: str, architect_output: dict[str, ArchitectDecision]) -> dict[str, GeneratorOutput]:
    print("[Generator Agent] Starting code generation...")
    results = {}

    for func_name, decision in architect_output.items():
        print(f"\n[Generator Agent] Generating: {func_name}")
        original_source = get_function_source(script_path, func_name)
        if not original_source:
            print(f"[Generator Agent] Could not find source for {func_name}, skipping.")
            continue

        optimized_code = clean_code(generate_optimized_code(original_source, decision))

        results[func_name] = GeneratorOutput(
            function=func_name,
            strategy=decision.strategy,
            original_source=original_source,
            optimized_code=optimized_code
        )
        print(f"[Generator Agent] Done: {func_name}")

    return results

def clean_code(raw_output: str) -> str:
    """Strip markdown backticks from LLM output."""
    lines = raw_output.split('\n')
    cleaned = []
    inside_block = False
    
    for line in lines:
        if line.strip().startswith('```'):
            inside_block = not inside_block
            continue
        if inside_block or not line.strip().startswith('```'):
            cleaned.append(line)
    
    return '\n'.join(cleaned).strip()

if __name__ == "__main__":
    sys.path.append(".")

    from kairos_lab.agents.ast_parser import run_ast_parser
    from kairos_lab.agents.architect import run_architect

    script = sys.argv[1] if len(sys.argv) > 1 else "sample_script.py"
    bottlenecks = ["inefficient_loop", "matrix_ops"]

    print("[Pipeline] Step 1: AST Parser...")
    ast_output = run_ast_parser(script, bottlenecks)

    print("\n[Pipeline] Step 2: Architect...")
    architect_output = run_architect(ast_output)

    print("\n[Pipeline] Step 3: Generator...")
    generator_output = run_generator(script, architect_output)

    print("\n" + "=" * 50)
    print("GENERATOR AGENT OUTPUT")
    print("=" * 50)
    for func_name, result in generator_output.items():
        print(f"\n--- {func_name} → {result.strategy.upper()} ---")
        print("\n[ORIGINAL]")
        print(result.original_source)
        print("\n[OPTIMIZED]")
        print(result.optimized_code)