import ast
import sys
sys.path.append(".")

from kairos_lab.models import CriticOutput, GeneratorOutput, ASTResult


def load_correctness_patterns(knowledge_base: dict = None) -> tuple[list, list]:
    """
    Load correctness patterns.
    Checks knowledge base first, falls back to defaults.
    """

    DEFAULT_REQUIRED_ELEMENTS = [
        "return",
    ]

    DEFAULT_DANGEROUS_PATTERNS = [
        "os.system",
        "subprocess",
        "eval(",
        "exec(",
        "open(",
        "__import__",
        "shutil",
        "rmdir",
        "unlink",
    ]

    if not knowledge_base:
        return DEFAULT_REQUIRED_ELEMENTS, DEFAULT_DANGEROUS_PATTERNS

    required = knowledge_base.get("correctness_required", DEFAULT_REQUIRED_ELEMENTS)
    dangerous = knowledge_base.get("dangerous_patterns", DEFAULT_DANGEROUS_PATTERNS)

    return required, dangerous


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


def check_has_return(code: str, function_name: str) -> bool:
    """Check if the function has a return statement."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                for child in ast.walk(node):
                    if isinstance(child, ast.Return):
                        return True
    except SyntaxError:
        pass
    return False


def check_signature_preserved(
    code: str,
    function_name: str,
    original_ast: ASTResult
) -> tuple[bool, str]:
    """
    Check if the optimized function preserves the original signature.
    Argument count and names should match.
    """
    if not original_ast or not original_ast.found:
        return True, ""

    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                optimized_args = [
                    arg.arg for arg in node.args.args
                    if arg.arg != "self"
                ]
                original_args = [
                    a for a in original_ast.args
                    if a != "self"
                ]

                if len(optimized_args) != len(original_args):
                    return False, (
                        f"Argument count mismatch — "
                        f"original: {original_args}, "
                        f"optimized: {optimized_args}"
                    )
                return True, ""
    except SyntaxError:
        pass

    return True, ""


def check_return_type_consistent(
    code: str,
    function_name: str,
    original_ast: ASTResult
) -> tuple[bool, str]:
    """
    Check if return structure is consistent with original.
    Both should return something or both should return nothing.
    """
    if not original_ast or not original_ast.found:
        return True, ""

    original_has_return = len(original_ast.returns) > 0
    optimized_has_return = check_has_return(code, function_name)

    if original_has_return and not optimized_has_return:
        return False, "Original function returns a value but optimized version does not"

    return True, ""


def check_dangerous_patterns(code: str, dangerous_patterns: list) -> list[str]:
    """Check for dangerous operations in generated code."""
    found = []
    for pattern in dangerous_patterns:
        if pattern in code:
            found.append(pattern)
    return found


def check_loop_elimination(
    code: str,
    original_ast: ASTResult,
    strategy: str
) -> tuple[bool, str]:
    """
    For Numba/Triton strategies — check if Python loops were eliminated
    or properly handled.
    """
    if not original_ast or original_ast.loop_count == 0:
        return True, ""

    if strategy == "numba":
        # Numba can keep loops — they get JIT compiled
        # Just check the @jit decorator is present
        if "@nb.jit" in code or "@numba.jit" in code or "@nb.njit" in code:
            return True, ""
        return False, "Numba strategy requires @jit decorator — not found"

    if strategy == "triton":
        # Triton should eliminate Python loops entirely
        loop_patterns = ["for i in range", "for j in range", "while "]
        found = [p for p in loop_patterns if p in code]
        if found:
            return False, f"Triton kernel should not have Python loops: {found}"

    return True, ""


def score_correctness(
    syntax_ok: bool,
    has_function: bool,
    has_return: bool,
    signature_ok: bool,
    return_type_ok: bool,
    no_dangerous: bool,
    loop_ok: bool
) -> float:
    """Calculate correctness score from 0.0 to 1.0."""

    if not syntax_ok:
        return 0.0

    if not has_function:
        return 0.1

    score = 0.2  # base for syntax + function

    if has_return:
        score += 0.2

    if signature_ok:
        score += 0.2

    if return_type_ok:
        score += 0.15

    if no_dangerous:
        score += 0.15

    if loop_ok:
        score += 0.1

    return round(min(1.0, score), 2)


def analyze_function(
    func_name: str,
    gen_output: GeneratorOutput,
    original_ast: ASTResult,
    required_elements: list,
    dangerous_patterns: list
) -> CriticOutput:
    """Run correctness critic analysis on a single generated function."""

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
        suggestions.append(f"Regenerate — function named {func_name} is missing")

    # Check 3 — return statement
    has_return = check_has_return(code, func_name)
    if not has_return and has_function:
        issues.append("Function has no return statement")
        suggestions.append("Add return statement — original function returns a value")

    # Check 4 — signature preserved
    signature_ok, sig_error = check_signature_preserved(code, func_name, original_ast)
    if not signature_ok:
        issues.append(f"Signature mismatch: {sig_error}")
        suggestions.append("Fix function signature to match original")

    # Check 5 — return type consistent
    return_type_ok, ret_error = check_return_type_consistent(code, func_name, original_ast)
    if not return_type_ok:
        issues.append(f"Return type issue: {ret_error}")
        suggestions.append("Ensure optimized function returns same type as original")

    # Check 6 — dangerous patterns
    dangerous_found = check_dangerous_patterns(code, dangerous_patterns)
    no_dangerous = len(dangerous_found) == 0
    if dangerous_found:
        issues.append(f"Dangerous operations found: {dangerous_found}")
        suggestions.append("Remove dangerous operations from generated code")

    # Check 7 — loop elimination
    loop_ok, loop_error = check_loop_elimination(code, original_ast, strategy)
    if not loop_ok:
        issues.append(f"Loop issue: {loop_error}")
        suggestions.append(loop_error)

    score = score_correctness(
        syntax_ok, has_function, has_return,
        signature_ok, return_type_ok, no_dangerous, loop_ok
    )

    passed = score >= 0.5 and syntax_ok and has_function

    return CriticOutput(
        function=func_name,
        critic_type="correctness",
        passed=passed,
        score=score,
        issues=issues,
        suggestions=suggestions
    )


def run_correctness_critic(
    generator_output: dict[str, GeneratorOutput],
    ast_output: dict[str, ASTResult],
    knowledge_base: dict = None
) -> dict[str, CriticOutput]:
    """
    Main entry point. Runs correctness critic on all generated functions.
    Accepts optional knowledge_base from Learning Agent.
    Falls back to defaults if not provided.
    """

    print("[Correctness Critic] Loading patterns...")
    required_elements, dangerous_patterns = load_correctness_patterns(knowledge_base)

    print("[Correctness Critic] Running static correctness analysis...")
    results = {}

    for func_name, gen_output in generator_output.items():
        print(f"\n[Correctness Critic] Analyzing: {func_name}")

        original_ast = ast_output.get(func_name)

        result = analyze_function(
            func_name, gen_output, original_ast,
            required_elements, dangerous_patterns
        )
        results[func_name] = result

        status = "PASSED" if result.passed else "FAILED"
        print(f"[Correctness Critic] {func_name} → {status} (score: {result.score})")
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

    results = run_correctness_critic(
        generator_output, ast_output, knowledge_base=None
    )

    print("\n" + "=" * 50)
    print("CORRECTNESS CRITIC OUTPUT")
    print("=" * 50)
    print(json.dumps({k: v.model_dump() for k, v in results.items()}, indent=2))