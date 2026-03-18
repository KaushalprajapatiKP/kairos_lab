import tracemalloc
import ast
import sys
from pathlib import Path
sys.path.append(".")

from kairos_lab.models import (
    MemoryAgentOutput,
    MemoryComparisonOutput,
    GeneratorOutput,
    DataflowOutput
)

# Map package names to their common import aliases
IMPORT_ALIASES = {
    "torch": "import torch\nimport torch.nn as nn",
    "numpy": "import numpy as np",
    "pandas": "import pandas as pd",
    "matplotlib": "import matplotlib.pyplot as plt",
    "scipy": "import scipy",
    "sklearn": "import sklearn",
    "tensorflow": "import tensorflow as tf",
    "numba": "import numba as nb",
    "triton": "import triton\nimport triton.language as tl",
}


def build_import_string(available_packages: list[str], include_numba: bool = False) -> str:
    """Build import string from Dependency Resolver output — no hardcoding."""
    lines = []

    for pkg in available_packages:
        if pkg in IMPORT_ALIASES:
            lines.append(IMPORT_ALIASES[pkg])
        else:
            lines.append(f"import {pkg}")

    # Always include math — stdlib, always safe
    lines.append("import math")

    # Add numba for optimized pass if not already included
    if include_numba and "import numba as nb" not in "\n".join(lines):
        lines.append("import numba as nb")

    return "\n".join(lines) + "\n"


def is_method(source_code: str, function_name: str) -> bool:
    """Check if a function is a class method."""
    tree = ast.parse(source_code)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in ast.walk(node):
                if isinstance(item, ast.FunctionDef) and item.name == function_name:
                    return True
    return False


def find_class_for_method(source_code: str, function_name: str) -> str:
    """Find the class name that contains this method."""
    tree = ast.parse(source_code)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in ast.walk(node):
                if isinstance(item, ast.FunctionDef) and item.name == function_name:
                    return node.name
    return ""


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


def build_test_input_from_dataflow(dataflow: DataflowOutput) -> list:
    """Build test inputs from Dataflow Agent output — no hardcoding."""
    import torch

    inputs = []
    for arg_name, shape in dataflow.input_shapes.items():
        dtype = dataflow.input_types.get(arg_name, "torch.float32")
        if not shape:
            continue
        if dtype == "torch.int64":
            inputs.append(torch.randint(0, 10, shape))
        else:
            inputs.append(torch.randn(shape))

    return inputs


def measure_memory(code: str, function_name: str, test_inputs: list) -> float:
    """Execute code, measure and return peak memory in MB."""

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


def measure_method_memory(
    source: str,
    class_name: str,
    function_name: str,
    test_inputs: list,
    class_init_args: dict = None,
    import_string: str = ""
) -> float:
    """Instantiate class and measure memory of calling a method."""

    if class_init_args:
        args_str = ", ".join(f"{k}={v}" for k, v in class_init_args.items())
    else:
        args_str = ""

    input_params = ", ".join([f"input_{i}" for i in range(len(test_inputs))])
    input_args = ", ".join([f"input_{i}" for i in range(len(test_inputs))])

    wrapper = f"""
{import_string}
{source}

def __test_method__({input_params}):
    instance = {class_name}({args_str})
    return instance.{function_name}({input_args})
"""

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


def run_memory_agent_pass1(
    script_path: str,
    bottleneck_functions: list,
    dataflow_output: dict[str, DataflowOutput] = None,
    available_packages: list[str] = None
) -> dict[str, MemoryAgentOutput]:
    """
    PASS 1 — Measure baseline memory of original functions.
    Runs before Generator. Uses Dataflow Agent output for test inputs.
    Uses Dependency Resolver output for imports.
    """
    print("[Memory Agent] PASS 1 — Measuring baseline memory on original code...")

    import_string = build_import_string(available_packages or [], include_numba=False)
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

        dataflow = dataflow_output.get(func_name) if dataflow_output else None
        test_inputs = build_test_input_from_dataflow(dataflow) if dataflow else []

        if method:
            print(f"[Memory Agent] {func_name} is a class method — finding class...")
            class_name = find_class_for_method(source, func_name)
            if not class_name:
                results[func_name] = MemoryAgentOutput(
                    function=func_name,
                    peak_mb=-1.0,
                    phase="original",
                    warning="Class not found for method"
                )
                continue

            print(f"[Memory Agent] Found class: {class_name}")
            class_init_args = dataflow.class_init_args if dataflow else None
            peak_mb = measure_method_memory(
                source, class_name, func_name,
                test_inputs, class_init_args, import_string
            )

        else:
            full_code = import_string + source
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


def run_memory_agent_pass2(
    generator_output: dict[str, GeneratorOutput],
    dataflow_output: dict[str, DataflowOutput] = None,
    available_packages: list[str] = None
) -> dict[str, MemoryAgentOutput]:
    """
    PASS 2 — Measure memory of optimized functions after Verifier confirms they work.
    Uses Dependency Resolver output for imports.
    """
    print("[Memory Agent] PASS 2 — Measuring memory on optimized code...")

    import_string = build_import_string(available_packages or [], include_numba=True)
    results = {}

    for func_name, gen_output in generator_output.items():
        print(f"\n[Memory Agent] Measuring optimized: {func_name}")

        full_code = import_string + "\n" + gen_output.optimized_code

        dataflow = dataflow_output.get(func_name) if dataflow_output else None
        test_inputs = build_test_input_from_dataflow(dataflow) if dataflow else []

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
    from kairos_lab.agents.dataflow_agent import run_dataflow_agent
    from kairos_lab.agents.dependency_resolver import run_dependency_resolver

    script = sys.argv[1] if len(sys.argv) > 1 else "sample_project/main.py"

    profiler_output = run_profiler(script)
    bottlenecks = profiler_output.top_functions
    print(f"\n[Memory Agent] Bottlenecks: {bottlenecks}")

    dependency_output = run_dependency_resolver(script)
    available_packages = dependency_output.available
    print(f"[Memory Agent] Available packages: {available_packages}")

    dataflow_output = run_dataflow_agent(script, bottlenecks)

    pass1_results = run_memory_agent_pass1(
        script, bottlenecks, dataflow_output, available_packages
    )

    print("\n" + "=" * 50)
    print("MEMORY AGENT PASS 1 OUTPUT")
    print("=" * 50)
    print(json.dumps({k: v.model_dump() for k, v in pass1_results.items()}, indent=2))