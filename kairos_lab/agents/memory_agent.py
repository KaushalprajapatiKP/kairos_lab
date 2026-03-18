import tracemalloc
import ast
import sys
from pathlib import Path
sys.path.append(".")

from kairos_lab.models import MemoryAgentOutput, MemoryComparisonOutput, GeneratorOutput

ORIGINAL_IMPORTS = "import torch\nimport numpy as np\nimport math\n"
OPTIMIZED_IMPORTS = "import torch\nimport numpy as np\nimport math\nimport numba as nb\n"

def is_method(source_code: str, function_name: str) -> bool:
    """Check if a function is a class method."""
    tree = ast.parse(source_code)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in ast.walk(node):
                if isinstance(item, ast.FunctionDef) and item.name == function_name:
                    return True
    return False


def build_test_input(function_name: str) -> list:
    """Build minimal test inputs. Replaced by Dataflow Agent later."""
    import torch
    inputs = {
        "slow_relu": [torch.randn(32, 256)],
        "compute_accuracy": [torch.randn(32, 10), torch.randint(0, 10, (32,))],
        "compute_loss_manual": [torch.randn(32, 10), torch.randint(0, 10, (32,))],
        "inefficient_loop": [torch.randn(100, 100)],
        "_generate_data": [],
        "normalize": [[[float(i*j)/1000 for j in range(10)] for i in range(10)]],
        "matrix_ops": [],
    }
    return inputs.get(function_name, [])


def measure_memory(code: str, function_name: str, test_inputs: list) -> float:
    """Execute code, measure and return peak memory in MB."""

    # Syntax check first
    try:
        compile(code, "<string>", "exec")
    except SyntaxError as e:
        print(f"[Memory Agent] Syntax error in code for {function_name}: {e}")
        return -1.0

    namespace = {}
    try:
        exec(compile(code, "<string>", "exec"), namespace)
    except Exception as e:
        print(f"[Memory Agent] Execution error for {function_name}: {e}")
        return -1.0

    if function_name not in namespace:
        print(f"[Memory Agent] Function {function_name} not found in namespace")
        return -1.0

    func = namespace[function_name]

    tracemalloc.start()
    try:
        func(*test_inputs)
    except Exception as e:
        print(f"[Memory Agent] Runtime error for {function_name}: {e}")
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    return round(peak / (1024 * 1024), 4)


def find_function_source(script_path: str, function_name: str) -> tuple[str, bool]:
    """
    Find function source across all project files.
    Returns (full_source, is_method).
    """
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
                    method = is_method(source, function_name)
                    return source, method
        except SyntaxError:
            continue

    return "", False

def find_class_for_method(source_code: str, function_name: str) -> str:
    """Find the class name that contains this method."""
    tree = ast.parse(source_code)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in ast.walk(node):
                if isinstance(item, ast.FunctionDef) and item.name == function_name:
                    return node.name
    return ""


def measure_method_memory(source: str, class_name: str, function_name: str, test_inputs: list) -> float:
    """Instantiate class and measure memory of calling a method."""
    
    # Build init args for known classes
    init_args = {
        "SlowMLP": "input_dim=128, hidden_dim=256, output_dim=10",
        "SlowAttention": "dim=128",
        "SlowDataset": "size=100, feature_dim=32",
    }
    
    args = init_args.get(class_name, "")
    
    wrapper = f"""
{ORIGINAL_IMPORTS}
import torch.nn as nn
{source}

def __test_method__({', '.join(['self_input_' + str(i) for i in range(len(test_inputs))])}):
    instance = {class_name}({args})
    return instance.{function_name}({', '.join(['self_input_' + str(i) for i in range(len(test_inputs))])})
"""
    
    # Syntax check
    try:
        compile(wrapper, "<string>", "exec")
    except SyntaxError as e:
        print(f"[Memory Agent] Syntax error building method wrapper: {e}")
        return -1.0

    namespace = {}
    try:
        exec(compile(wrapper, "<string>", "exec"), namespace)
    except Exception as e:
        print(f"[Memory Agent] Error instantiating class {class_name}: {e}")
        return -1.0

    func = namespace.get("__test_method__")
    if not func:
        return -1.0

    tracemalloc.start()
    try:
        func(*test_inputs)
    except Exception as e:
        print(f"[Memory Agent] Runtime error for {class_name}.{function_name}: {e}")
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    return round(peak / (1024 * 1024), 4)

def run_memory_agent_pass1(script_path: str, bottleneck_functions: list) -> dict[str, MemoryAgentOutput]:
    """
    PASS 1 — Measure baseline memory of original functions.
    Runs before Generator. Only measures standalone functions, skips methods.
    """
    print("[Memory Agent] PASS 1 — Measuring baseline memory on original code...")
    results = {}

    for func_name in bottleneck_functions:
        print(f"\n[Memory Agent] Measuring original: {func_name}")

        source, method = find_function_source(script_path, func_name)

        if not source:
            print(f"[Memory Agent] Could not find source for {func_name}")
            results[func_name] = MemoryAgentOutput(
                function=func_name,
                peak_mb=-1.0,
                phase="original",
                warning="Function source not found"
            )
            continue

        if method:
            print(f"[Memory Agent] {func_name} is a class method — finding class...")
            class_name = find_class_for_method(source, func_name)
            if not class_name:
                print(f"[Memory Agent] Could not find class for {func_name}")
                results[func_name] = MemoryAgentOutput(
                    function=func_name,
                    peak_mb=-1.0,
                    phase="original",
                    warning="Class not found for method"
                )
                continue
            print(f"[Memory Agent] Found class: {class_name}")
            test_inputs = build_test_input(func_name)
            peak_mb = measure_method_memory(source, class_name, func_name, test_inputs)
            results[func_name] = MemoryAgentOutput(
                function=func_name,
                peak_mb=peak_mb,
                phase="original",
                warning=None if peak_mb >= 0 else "Could not measure class method memory"
            )
            if peak_mb >= 0:
                print(f"[Memory Agent] {func_name} baseline: {peak_mb}MB")
            continue

        full_code = ORIGINAL_IMPORTS + source
        test_inputs = build_test_input(func_name)
        peak_mb = measure_memory(full_code, func_name, test_inputs)

        results[func_name] = MemoryAgentOutput(
            function=func_name,
            peak_mb=peak_mb,
            phase="original",
            warning=None if peak_mb >= 0 else "Could not measure memory"
        )

        if peak_mb >= 0:
            print(f"[Memory Agent] {func_name} baseline: {peak_mb}MB")

    return results


def run_memory_agent_pass2(generator_output: dict[str, GeneratorOutput]) -> dict[str, MemoryAgentOutput]:
    """
    PASS 2 — Measure memory of optimized functions after Verifier confirms they work.
    """
    print("[Memory Agent] PASS 2 — Measuring memory on optimized code...")
    results = {}

    for func_name, gen_output in generator_output.items():
        print(f"\n[Memory Agent] Measuring optimized: {func_name}")

        prefix = OPTIMIZED_IMPORTS
        if gen_output.strategy == "numba":
            prefix += "import numba as nb\n"

        full_code = prefix + "\n" + gen_output.optimized_code
        test_inputs = build_test_input(func_name)
        peak_mb = measure_memory(full_code, func_name, test_inputs)

        results[func_name] = MemoryAgentOutput(
            function=func_name,
            peak_mb=peak_mb,
            phase="optimized",
            warning=None if peak_mb >= 0 else "Could not measure optimized memory"
        )

        if peak_mb >= 0:
            print(f"[Memory Agent] {func_name} optimized: {peak_mb}MB")

    return results


def compare_memory(
    pass1: dict[str, MemoryAgentOutput],
    pass2: dict[str, MemoryAgentOutput]
) -> dict[str, MemoryComparisonOutput]:
    """Compare Pass 1 vs Pass 2 memory. Called after both passes complete."""
    results = {}

    for func_name in pass1:
        if func_name not in pass2:
            continue

        original_mb = pass1[func_name].peak_mb
        optimized_mb = pass2[func_name].peak_mb

        if original_mb < 0 or optimized_mb < 0:
            results[func_name] = MemoryComparisonOutput(
                function=func_name,
                original_peak_mb=original_mb,
                optimized_peak_mb=optimized_mb,
                memory_delta_mb=0.0,
                memory_increased=False,
                warning="Could not compare — one or both measurements failed"
            )
            continue

        delta = optimized_mb - original_mb
        memory_increased = delta > 0.1

        results[func_name] = MemoryComparisonOutput(
            function=func_name,
            original_peak_mb=original_mb,
            optimized_peak_mb=optimized_mb,
            memory_delta_mb=round(delta, 4),
            memory_increased=memory_increased,
            warning=f"Optimized uses {delta:.2f}MB more memory. Review before deploying." if memory_increased else None
        )

    return results


if __name__ == "__main__":
    import json
    from kairos_lab.agents.profiler import run_profiler

    script = sys.argv[1] if len(sys.argv) > 1 else "sample_project/main.py"

    profiler_output = run_profiler(script)
    bottlenecks = profiler_output.top_functions

    print(f"\n[Memory Agent] Bottlenecks to measure: {bottlenecks}")

    pass1_results = run_memory_agent_pass1(script, bottlenecks)

    print("\n" + "=" * 50)
    print("MEMORY AGENT PASS 1 OUTPUT")
    print("=" * 50)
    print(json.dumps({k: v.model_dump() for k, v in pass1_results.items()}, indent=2))