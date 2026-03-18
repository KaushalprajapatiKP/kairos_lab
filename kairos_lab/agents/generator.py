import requests
import sys
sys.path.append(".")

from pathlib import Path
from kairos_lab.models import ArchitectDecision, GeneratorOutput, ASTResult

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "deepseek-r1:7b"
MAX_RETRIES = 3


def get_function_source(script_path: str, function_name: str) -> str:
    import ast
    from pathlib import Path

    script = Path(script_path)
    project_folder = script.parent
    all_files = list(project_folder.rglob("*.py"))
    if script not in all_files:
        all_files.append(script)

    for f in all_files:
        if f.name == "__init__.py":
            continue
        try:
            source = f.read_text()
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    lines = source.split('\n')
                    start = node.lineno - 1
                    end = node.end_lineno
                    return '\n'.join(lines[start:end])
        except SyntaxError:
            continue

    return ""


def clean_code(raw_output: str) -> str:
    """Strip markdown backticks and prose from LLM output."""
    lines = raw_output.split('\n')
    cleaned = []
    inside_block = False

    PROSE_STARTS = [
        "The ", "This ", "Note ", "Key ", "Here ",
        "We ", "To ", "In ", "###", "**", "1.", "2.", "3.",
    ]

    for line in lines:
        stripped = line.strip()

        if stripped.startswith('```'):
            inside_block = not inside_block
            continue

        if not inside_block:
            is_prose = any(stripped.startswith(p) for p in PROSE_STARTS)
            if is_prose:
                continue

        cleaned.append(line)

    return '\n'.join(cleaned).strip()


def call_llm(prompt: str) -> str:
    """Direct LLM call with a prompt string."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    result = response.json()
    return result["response"]


def build_initial_prompt(function_source: str, decision: ArchitectDecision) -> str:
    """Build the initial generation prompt — surgical, strategy-specific."""

    if decision.strategy == "numba":
        return f"""Optimize this Python function using Numba.

ORIGINAL FUNCTION:
{function_source}

EXACT CHANGES REQUIRED:
1. Add at top: import numba as nb
2. Add at top: import numpy as np
3. Add @nb.jit(nopython=True) decorator directly above def {decision.function}
4. Replace all .item() calls with float()
5. Replace Python lists and .append() with numpy arrays
6. Keep all logic identical — do not rewrite the algorithm
7. Keep function named exactly: {decision.function}
8. Function must have a return statement

Output ONLY the optimized Python code. No explanations. No markdown.:"""

    if decision.strategy == "triton":
        return f"""Optimize this Python function using OpenAI Triton.

ORIGINAL FUNCTION:
{function_source}

EXACT REQUIREMENTS:
1. Write a @triton.jit kernel function
2. Write a Python wrapper named exactly: {decision.function}
3. Add all imports at top: import triton, import triton.language as tl, import torch
4. Wrapper must accept same inputs and return same outputs as original
5. Use tl.program_id, tl.load, tl.store for GPU parallelism

Output ONLY the optimized Python code. No explanations. No markdown.:"""

    # cuda fallback
    return f"""Optimize this Python function using PyTorch vectorization.

ORIGINAL FUNCTION:
{function_source}

EXACT REQUIREMENTS:
1. Replace all Python loops with torch operations
2. Use torch.matmul, torch.sum, torch.exp where applicable
3. Keep function named exactly: {decision.function}
4. Include all imports at top
5. Function must return same type as original

Output ONLY the optimized Python code. No explanations. No markdown.:"""


def build_retry_prompt(
    function_source: str,
    decision: ArchitectDecision,
    critic_issues: list[str]
) -> str:
    """Build retry prompt — surgical fixes, strategy-specific."""
    issues_text = "\n".join(f"- {issue}" for issue in critic_issues)

    if decision.strategy == "numba":
        return f"""ORIGINAL FUNCTION:
{function_source}

Previous attempt failed. Fix ONLY these issues:
{issues_text}

NUMBA REQUIREMENTS:
1. import numba as nb must be at top
2. import numpy as np must be at top
3. @nb.jit(nopython=True) must be directly above def {decision.function}
4. No .item() calls — replace with float()
5. No Python lists — replace with numpy arrays
6. Function named exactly: {decision.function}
7. Must have return statement

Output ONLY Python code:"""

    if decision.strategy == "triton":
        return f"""ORIGINAL FUNCTION:
{function_source}

Previous attempt failed. Fix ONLY these issues:
{issues_text}

TRITON REQUIREMENTS:
1. import triton at top
2. import triton.language as tl at top
3. @triton.jit kernel must be present
4. Wrapper function named exactly: {decision.function}
5. Must have return statement

Output ONLY Python code:"""

    return f"""ORIGINAL FUNCTION:
{function_source}

Fix these issues:
{issues_text}

Function must be named exactly: {decision.function}
Output ONLY Python code:"""


def run_generator(
    script_path: str,
    architect_output: dict[str, ArchitectDecision],
    ast_output: dict[str, ASTResult] = None,
    max_retries: int = MAX_RETRIES
) -> dict[str, GeneratorOutput]:
    """
    Main entry point. Generates optimized code per function.
    Retries up to max_retries times using both critic feedbacks.
    """
    from kairos_lab.agents.performance_critic import run_performance_critic
    from kairos_lab.agents.correctness_critic import run_correctness_critic

    print("[Generator Agent] Starting code generation...")
    results = {}

    for func_name, decision in architect_output.items():
        print(f"\n[Generator Agent] Generating: {func_name}")

        original_source = get_function_source(script_path, func_name)
        if not original_source:
            print(f"[Generator Agent] Could not find source for {func_name}, skipping.")
            continue

        optimized_code = ""
        critic_issues = []
        attempt = 0

        while attempt < max_retries:
            attempt += 1
            print(f"[Generator Agent] Attempt {attempt}/{max_retries} for {func_name}")

            if attempt == 1:
                prompt = build_initial_prompt(original_source, decision)
            else:
                prompt = build_retry_prompt(
                    original_source, decision, critic_issues
                )

            raw_code = call_llm(prompt)
            optimized_code = clean_code(raw_code)

            # Build temp output for critic evaluation
            temp_output = {
                func_name: GeneratorOutput(
                    function=func_name,
                    strategy=decision.strategy,
                    original_source=original_source,
                    optimized_code=optimized_code
                )
            }

            # Run Performance Critic
            perf_results = run_performance_critic(
                temp_output,
                {func_name: decision},
                knowledge_base=None
            )

            # Run Correctness Critic
            ast_for_func = {func_name: ast_output[func_name]} if ast_output and func_name in ast_output else {}
            corr_results = run_correctness_critic(
                temp_output,
                ast_for_func,
                knowledge_base=None
            )

            perf_result = perf_results.get(func_name)
            corr_result = corr_results.get(func_name)

            both_passed = (
                perf_result and perf_result.passed and
                corr_result and corr_result.passed
            )

            if both_passed:
                print(f"[Generator Agent] {func_name} passed both critics on attempt {attempt} "
                      f"(perf: {perf_result.score}, corr: {corr_result.score})")
                break
            else:
                # Collect all issues from both critics
                critic_issues = []
                if perf_result and not perf_result.passed:
                    critic_issues.extend(perf_result.issues)
                if corr_result and not corr_result.passed:
                    critic_issues.extend(corr_result.issues)

                perf_score = perf_result.score if perf_result else 0
                corr_score = corr_result.score if corr_result else 0

                print(f"[Generator Agent] {func_name} failed critics "
                      f"(perf: {perf_score}, corr: {corr_score}) — attempt {attempt}/{max_retries}")
                for issue in critic_issues:
                    print(f"  ⚠ {issue}")

                if attempt == max_retries:
                    print(f"[Generator Agent] {func_name} failed after {max_retries} attempts — keeping best attempt")

        results[func_name] = GeneratorOutput(
            function=func_name,
            strategy=decision.strategy,
            original_source=original_source,
            optimized_code=optimized_code
        )
        print(f"[Generator Agent] Done: {func_name}")

    return results


if __name__ == "__main__":
    sys.path.append(".")

    from kairos_lab.agents.profiler import run_profiler
    from kairos_lab.agents.ast_parser import run_ast_parser
    from kairos_lab.agents.architect import run_architect

    script = sys.argv[1] if len(sys.argv) > 1 else "sample_project/main.py"

    print("[Pipeline] Step 1: Profiler...")
    profiler_output = run_profiler(script)
    bottlenecks = profiler_output.top_functions

    print("\n[Pipeline] Step 2: AST Parser...")
    ast_output = run_ast_parser(script, bottlenecks)

    print("\n[Pipeline] Step 3: Architect...")
    architect_output = run_architect(ast_output)

    print("\n[Pipeline] Step 4: Generator with critic feedback loop...")
    generator_output = run_generator(script, architect_output, ast_output)

    print("\n" + "=" * 50)
    print("GENERATOR AGENT OUTPUT")
    print("=" * 50)
    for func_name, result in generator_output.items():
        print(f"\n--- {func_name} → {result.strategy.upper()} ---")
        print("\n[ORIGINAL]")
        print(result.original_source)
        print("\n[OPTIMIZED]")
        print(result.optimized_code)