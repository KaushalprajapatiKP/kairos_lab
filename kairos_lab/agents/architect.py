import json
import sys
sys.path.append(".")

from kairos_lab.models import ASTResult, ArchitectDecision


def decide_strategy(ast_result: ASTResult) -> ArchitectDecision:
    pattern = ast_result.memory_access_pattern
    data_types = ast_result.data_types
    loop_depth = ast_result.max_nesting_depth
    func_name = ast_result.function_name

    if (pattern == "element_wise" and
            "torch_tensor" in data_types and
            loop_depth >= 2):
        return ArchitectDecision(
            function=func_name,
            strategy="triton",
            confidence="high",
            reason="Nested loops with element-wise tensor access. Triton kernel will parallelize across rows and columns.",
            action=f"Generate Triton kernel for {func_name} that parallelizes across tensor dimensions."
        )

    elif (pattern == "element_wise" and
          "python_list" in data_types and
          loop_depth >= 1):
        return ArchitectDecision(
            function=func_name,
            strategy="numba",
            confidence="high",
            reason="Simple loops with Python lists or scalar ops. Numba JIT will give significant speedup.",
            action=f"Apply @numba.jit decorator to {func_name} and replace list with numpy array."
        )

    elif pattern == "vectorized" and "torch_tensor" in data_types:
        return ArchitectDecision(
            function=func_name,
            strategy="cuda",
            confidence="medium",
            reason="Already vectorized torch ops. Custom CUDA kernel can fuse operations.",
            action=f"Write fused CUDA kernel for {func_name} to reduce memory bandwidth."
        )

    elif pattern == "vectorized":
        return ArchitectDecision(
            function=func_name,
            strategy="numba",
            confidence="medium",
            reason="Vectorized pattern without explicit tensor ops. Numba is safest bet.",
            action=f"Apply @numba.jit to {func_name}."
        )

    else:
        return ArchitectDecision(
            function=func_name,
            strategy="numba",
            confidence="low",
            reason="Pattern unclear. Defaulting to Numba as lowest-risk option.",
            action=f"Apply @numba.jit to {func_name} as first optimization attempt."
        )


def run_architect(ast_output: dict[str, ASTResult]) -> dict[str, ArchitectDecision]:
    print("[Architect Agent] Deciding parallelization strategies...")
    decisions = {}
    for func_name, ast_result in ast_output.items():
        if not ast_result.found:
            print(f"[Architect Agent] Skipping {func_name} — not found in AST")
            continue
        decision = decide_strategy(ast_result)
        decisions[func_name] = decision
        print(f"[Architect Agent] {func_name} → {decision.strategy.upper()} ({decision.confidence} confidence)")
    return decisions


if __name__ == "__main__":
    import sys
    sys.path.append(".")

    from kairos_lab.agents.ast_parser import run_ast_parser

    script = sys.argv[1] if len(sys.argv) > 1 else "sample_script.py"
    from kairos_lab.agents.profiler import run_profiler
    profiler_output = run_profiler(script)
    bottlenecks = profiler_output.top_functions


    ast_output = run_ast_parser(script, bottlenecks)
    decisions = run_architect(ast_output)

    print("\n" + "=" * 50)
    print("ARCHITECT AGENT OUTPUT")
    print("=" * 50)
    print(json.dumps({k: v.model_dump() for k, v in decisions.items()}, indent=2))