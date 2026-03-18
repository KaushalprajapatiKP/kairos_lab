import ast
import sys
sys.path.append(".")

from kairos_lab.models import CriticOutput, GeneratorOutput, ArchitectDecision


def load_patterns(knowledge_base: dict = None) -> tuple[dict, list, list]:
    """
    Load optimization patterns.
    First checks knowledge base (Learning Agent output).
    Falls back to hardcoded defaults if not found.
    """

    DEFAULT_POSITIVE = {
        "triton": [
            "tl.program_id", "tl.load", "tl.store",
            "tl.arange", "tl.sum", "@triton.jit", "triton.jit",
        ],
        "numba": [
            "@nb.jit", "@numba.jit", "nopython=True",
            "np.dot", "np.matmul", "np.sum", "np.exp", "@nb.njit",
        ],
        "cuda": [
            "torch.cuda", ".cuda()", "torch.matmul", "torch.einsum",
        ]
    }

    DEFAULT_NEGATIVE = [
        "for i in range", "for j in range",
        ".item()", "result.append", ".tolist()",
    ]

    DEFAULT_PROSE = [
        "The ", "This ", "Note ", "Key ", "Here ",
        "We ", "To ", "In ", "###", "**", "1.", "2.", "3.",
    ]

    if not knowledge_base:
        return DEFAULT_POSITIVE, DEFAULT_NEGATIVE, DEFAULT_PROSE

    positive = knowledge_base.get("performance_positive", DEFAULT_POSITIVE)
    negative = knowledge_base.get("performance_negative", DEFAULT_NEGATIVE)
    prose = knowledge_base.get("prose_patterns", DEFAULT_PROSE)

    return positive, negative, prose


def check_syntax(code: str) -> tuple[bool, str]:
    """Check if generated code is syntactically valid Python."""
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, str(e)


def check_has_function(code: str, function_name: str) -> bool:
    """Check if generated code contains the expected function."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                return True
    except SyntaxError:
        pass
    return False


def check_positive_patterns(code: str, strategy: str, positive_patterns: dict) -> list[str]:
    """Check for patterns that indicate good optimization."""
    found = []
    patterns = positive_patterns.get(strategy, [])
    for pattern in patterns:
        if pattern in code:
            found.append(pattern)
    return found


def check_negative_patterns(code: str, negative_patterns: list) -> list[str]:
    """Check for patterns that indicate optimization is missing or broken."""
    found = []
    for pattern in negative_patterns:
        if pattern in code:
            found.append(pattern)
    return found


def check_prose_leaked(code: str, prose_patterns: list) -> list[str]:
    """Check if LLM explanation text leaked into the code."""
    leaked = []
    for line in code.split('\n'):
        stripped = line.strip()
        for pattern in prose_patterns:
            if stripped.startswith(pattern):
                leaked.append(stripped[:50])
                break
    return leaked


def check_imports_present(code: str, strategy: str) -> tuple[bool, list[str]]:
    """Check if required imports for the strategy are present."""
    required = {
        "triton": ["triton"],
        "numba": ["numba"],
        "cuda": ["torch"],
    }
    missing = []
    for imp in required.get(strategy, []):
        if imp not in code:
            missing.append(imp)
    return len(missing) == 0, missing


def score_performance(
    syntax_ok: bool,
    has_function: bool,
    positive_hits: list,
    negative_hits: list,
    prose_leaked: list,
    imports_ok: bool,
    strategy: str,
    positive_patterns: dict
) -> float:
    """Calculate a performance score from 0.0 to 1.0."""

    if not syntax_ok:
        return 0.0

    if not has_function:
        return 0.1

    score = 0.3  # base for valid syntax + function present

    if imports_ok:
        score += 0.1

    max_positive = len(positive_patterns.get(strategy, []))
    if max_positive > 0:
        score += 0.4 * (len(positive_hits) / max_positive)

    score -= 0.1 * len(negative_hits)
    score -= 0.15 * len(prose_leaked)

    return round(max(0.0, min(1.0, score)), 2)


def analyze_function(
    func_name: str,
    gen_output: GeneratorOutput,
    architect_decision: ArchitectDecision,
    positive_patterns: dict,
    negative_patterns: list,
    prose_patterns: list
) -> CriticOutput:
    """Run performance critic analysis on a single generated function."""

    code = gen_output.optimized_code
    strategy = gen_output.strategy
    issues = []
    suggestions = []

    # Check 1 — syntax
    syntax_ok, syntax_error = check_syntax(code)
    if not syntax_ok:
        issues.append(f"Syntax error: {syntax_error}")
        suggestions.append("Regenerate — code has syntax errors")

    # Check 2 — function present
    has_function = check_has_function(code, func_name)
    if not has_function:
        issues.append(f"Function {func_name} not found in generated code")
        suggestions.append(f"Regenerate — wrapper function named {func_name} is missing")

    # Check 3 — positive optimization patterns
    positive_hits = check_positive_patterns(code, strategy, positive_patterns)
    if not positive_hits:
        issues.append(f"No {strategy.upper()} optimization patterns detected")
        suggestions.append(f"Generated code may not actually use {strategy.upper()} — verify kernel is present")

    # Check 4 — negative patterns
    negative_hits = check_negative_patterns(code, negative_patterns)
    if negative_hits:
        issues.append(f"Python loop patterns still present: {negative_hits}")
        suggestions.append("Optimization may be incomplete — Python loops found in generated code")

    # Check 5 — prose leaked
    prose_leaked = check_prose_leaked(code, prose_patterns)
    if prose_leaked:
        issues.append(f"LLM explanation text leaked into code: {len(prose_leaked)} lines")
        suggestions.append("Strip prose from generated code before execution")

    # Check 6 — imports
    imports_ok, missing_imports = check_imports_present(code, strategy)
    if not imports_ok:
        issues.append(f"Missing imports for {strategy}: {missing_imports}")
        suggestions.append(f"Add missing imports: {missing_imports}")

    score = score_performance(
        syntax_ok, has_function, positive_hits,
        negative_hits, prose_leaked, imports_ok,
        strategy, positive_patterns
    )

    passed = score >= 0.5 and syntax_ok and has_function

    return CriticOutput(
        function=func_name,
        critic_type="performance",
        passed=passed,
        score=score,
        issues=issues,
        suggestions=suggestions
    )


def run_performance_critic(
    generator_output: dict[str, GeneratorOutput],
    architect_output: dict[str, ArchitectDecision],
    knowledge_base: dict = None
) -> dict[str, CriticOutput]:
    """
    Main entry point. Runs performance critic on all generated functions.
    Accepts optional knowledge_base from Learning Agent.
    Falls back to hardcoded defaults if not provided.
    """

    print("[Performance Critic] Loading patterns...")
    positive_patterns, negative_patterns, prose_patterns = load_patterns(knowledge_base)

    print("[Performance Critic] Running static performance analysis...")
    results = {}

    for func_name, gen_output in generator_output.items():
        print(f"\n[Performance Critic] Analyzing: {func_name}")

        architect_decision = architect_output.get(func_name)
        if not architect_decision:
            print(f"[Performance Critic] No architect decision for {func_name} — skipping")
            continue

        result = analyze_function(
            func_name, gen_output, architect_decision,
            positive_patterns, negative_patterns, prose_patterns
        )
        results[func_name] = result

        status = "PASSED" if result.passed else "FAILED"
        print(f"[Performance Critic] {func_name} → {status} (score: {result.score})")
        if result.issues:
            for issue in result.issues:
                print(f"  ⚠ {issue}")

    return results


if __name__ == "__main__":
    import json
    from kairos_lab.agents.profiler import run_profiler
    from kairos_lab.agents.ast_parser import run_ast_parser
    from kairos_lab.agents.architect import run_architect
    from kairos_lab.agents.generator import run_generator

    script = sys.argv[1] if len(sys.argv) > 1 else "sample_project/main.py"

    profiler_output = run_profiler(script)
    bottlenecks = profiler_output.top_functions

    ast_output = run_ast_parser(script, bottlenecks)
    architect_output = run_architect(ast_output)
    generator_output = run_generator(script, architect_output)

    # knowledge_base=None until Learning Agent is built
    results = run_performance_critic(generator_output, architect_output, knowledge_base=None)

    print("\n" + "=" * 50)
    print("PERFORMANCE CRITIC OUTPUT")
    print("=" * 50)
    print(json.dumps({k: v.model_dump() for k, v in results.items()}, indent=2))