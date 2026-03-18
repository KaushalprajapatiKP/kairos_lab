import requests
import sys
sys.path.append(".")

from pathlib import Path
from kairos_lab.models import ArchitectDecision, GeneratorOutput

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"
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

        # Skip prose lines outside code blocks
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
    """Build the initial generation prompt."""
    strategy = decision.strategy

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

    return f"""You are an expert GPU performance engineer.

ORIGINAL FUNCTION:
```python
{function_source}
```

ANALYSIS:
- Strategy: {strategy.upper()}
- Reason: {decision.reason}
- Action: {decision.action}

TASK:
{strategy_instructions}

STRICT RULES:
- Output ONLY valid Python code
- No explanations before or after the code
- No markdown formatting or backticks
- No prose or numbered lists
- Code must be complete and runnable
- The optimized function must be named exactly: {decision.function}
- Include all imports at the top

Generate optimized code now:"""


def build_retry_prompt(
    function_source: str,
    decision: ArchitectDecision,
    critic_issues: list[str]
) -> str:
    issues_text = "\n".join(f"- {issue}" for issue in critic_issues)

    return f"""Fix this Python function using {decision.strategy.upper()}.

ORIGINAL:
{function_source}

ISSUES TO FIX:
{issues_text}

RULES:
- Output ONLY Python code, nothing else
- Function must be named exactly: {decision.function}
- Include imports at top
- No explanations, no markdown

Code:"""

def run_generator(
    script_path: str,
    architect_output: dict[str, ArchitectDecision],
    max_retries: int = MAX_RETRIES
) -> dict[str, GeneratorOutput]:
    """
    Main entry point. Generates optimized code per function.
    Retries up to max_retries times using Performance Critic feedback.
    """
    from kairos_lab.agents.performance_critic import run_performance_critic

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
                    original_source, decision,
                    optimized_code, critic_issues
                )

            raw_code = call_llm(prompt)
            optimized_code = clean_code(raw_code)

            # Run Performance Critic check
            temp_output = {
                func_name: GeneratorOutput(
                    function=func_name,
                    strategy=decision.strategy,
                    original_source=original_source,
                    optimized_code=optimized_code
                )
            }

            critic_results = run_performance_critic(
                temp_output,
                {func_name: decision},
                knowledge_base=None
            )

            critic_result = critic_results.get(func_name)

            if critic_result and critic_result.passed:
                print(f"[Generator Agent] {func_name} passed critic on attempt {attempt} (score: {critic_result.score})")
                break
            else:
                critic_issues = critic_result.issues if critic_result else ["Unknown failure"]
                print(f"[Generator Agent] {func_name} failed critic (score: {critic_result.score if critic_result else 0}) — attempt {attempt}/{max_retries}")
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