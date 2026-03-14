import requests
import json
import ast
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "deepseek-r1:7b"

def get_function_source(script_path: str, function_name: str) -> str:
    """
    Extract the source code for a specific function from a script.
    """
    
    source = Path(script_path).read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            lines = source.split('\n')
            start_line = node.lineno - 1
            end_line = node.end_lineno 
            return '\n'.join(lines[start_line:end_line])
        
    return ""


def generate_optimized_code(function_source: str, decision: dict) -> str:
    """
    Generate optimized code for a specific function based on the Architect Agent's decision.
    """
    
    strategy = decision["strategy"]
    action = decision["action"]
    reason = decision["reason"]
    func_name = decision["function"]

    if strategy == "triton":
        strategy_instructions = f"""
            Write a complete OpenAI Triton kernel that replaces this function.
        Include:
        1. The @triton.jit kernel function
        2. A Python wrapper function with the same name and signature as the original
        3. All necessary imports (triton, triton.language as tl, torch)
        The wrapper must be a drop-in replacement — same inputs, same outputs.
        """ 

    elif strategy == "numba":
        strategy_instructions = f"""
            Write an optimized version using Numba and NumPy that replaces this function.
        Include:
        1. @numba.jit(nopython=True) decorator
        2. Replace Python lists with numpy arrays
        3. All necessary imports (numba, numpy)
        The function must be a drop-in replacement — same inputs, same outputs.
        """
    
    elif strategy == "cuda":
        strategy_instructions = f"""
           Write an optimized PyTorch version that replaces this function.
        Use torch operations to fuse and vectorize — avoid Python loops entirely.
        Include all necessary imports.
        The function must be a drop-in replacement — same inputs, same outputs.
        """
    
    else:
        return f"No optimization strategy provided for {func_name}."

    prompt = f"""
    You are an expert GPU performance engineer specializing in Python to GPU code translation.
    Below is the source code for a function that is slow and needs optimization.
    You are given a decision on the best parallelization strategy to use.
    Generate the optimized code for the function.

    ORIGINAL FUNCTION:
    ```python
        {function_source}
    ```

    ANALYSIS:
- Strategy decided: {strategy.upper()}
- Reason: {reason}
- Action: {action}

TASK:
{strategy_instructions}

RULES:
- Output ONLY the optimized Python code
- No explanations before or after the code
- No markdown formatting or backticks
- The code must be complete and runnable
- Include all imports at the top
- Add a brief inline comment explaining the key optimization

Generate the optimized code now:"""

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }

    print(f"[Generator Agent] Calling LLM for {func_name} → {strategy.upper()} strategy...")
    response = requests.post(OLLAMA_URL, json=payload)
    result = response.json()
    return result["response"]

def run_generator(script_path: str, decisions: dict) -> dict:
    """Main entry point. Takes Architect output, generates optimized code per function."""

    print("[Generator Agent] Starting code generation...")

    results = {}

    for func_name, decision in architect_output.items():
        print(f"\n[Generator Agent] Generating optimized code for: {func_name}")

        # Get original function source
        original_source = get_function_source(script_path, func_name)
        if not original_source:
            print(f"[Generator Agent] Could not find source for {func_name}, skipping.")
            continue

        # Generate optimized version
        optimized_code = generate_optimized_code(original_source, decision)

        results[func_name] = {
            "function": func_name,
            "strategy": decision["strategy"],
            "original_source": original_source,
            "optimized_code": optimized_code
        }

        print(f"[Generator Agent] Done: {func_name}")

    return results

if __name__ == "__main__":
    import sys
    sys.path.append(".")

    from kairos_lab.agents.ast_parser import run_ast_parser
    from kairos_lab.agents.architect import run_architect
    
    import json

    script = sys.argv[1] if len(sys.argv) > 1 else "sample_script.py"
    bottlenecks = ["inefficient_loop", "matrix_ops"]

    # RUN Full Pipeline
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
        print(f"\n--- {func_name} → {result['strategy'].upper()} ---")
        print("\n[ORIGINAL CODE]")
        print(result["original_source"])
        print("\n[OPTIMIZED CODE]")
        print(result["optimized_code"])
