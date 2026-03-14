import ast
from pathlib import Path
import sys
sys.path.append(".")

from kairos_lab.models import ASTResult


def parse_function(source_code: str, function_name: str) -> ASTResult:
    tree = ast.parse(source_code)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            loop_count, max_nesting_depth = analyze_loops(node)
            return ASTResult(
                function_name=function_name,
                found=True,
                args=[arg.arg for arg in node.args.args],
                loop_count=loop_count,
                max_nesting_depth=max_nesting_depth,
                data_types=detect_data_types(node),
                memory_access_pattern=detect_memory_access_pattern(node),
                operations=extract_operations(node),
                returns=extract_returns(node)
            )

    return ASTResult(
        function_name=function_name,
        found=False,
        args=[],
        loop_count=0,
        max_nesting_depth=0,
        data_types=[],
        memory_access_pattern="unknown",
        operations=[],
        returns=[]
    )


def analyze_loops(func_node: ast.FunctionDef) -> tuple:
    loop_count = 0
    max_nesting_depth = 0

    def walk_depth(node, depth=0):
        nonlocal loop_count, max_nesting_depth
        if isinstance(node, ast.For) or isinstance(node, ast.While):
            loop_count += 1
            max_nesting_depth = max(max_nesting_depth, depth + 1)
            for child in ast.iter_child_nodes(node):
                walk_depth(child, depth + 1)
        else:
            for child in ast.iter_child_nodes(node):
                walk_depth(child, depth)

    walk_depth(func_node)
    return loop_count, max_nesting_depth


def detect_data_types(func_node: ast.FunctionDef) -> list:
    types_found = set()
    for node in ast.walk(func_node):
        if isinstance(node, ast.Attribute):
            if node.attr in ("append", "extend"):
                types_found.add("python_list")
            if node.attr in ("item", "shape", "dtype", "cuda", "cpu"):
                types_found.add("torch_tensor")
            if node.attr in ("numpy", "array", "zeros", "ones"):
                types_found.add("numpy_array")
        if isinstance(node, ast.Subscript):
            types_found.add("indexed_access")
        if isinstance(node, ast.List):
            types_found.add("python_list")
    return list(types_found)


def detect_memory_access_pattern(func_node: ast.FunctionDef) -> str:
    has_nested_loops = False
    has_subscript_in_loop = False
    has_vectorized_ops = False

    def check_node(node, in_loop=False):
        nonlocal has_nested_loops, has_subscript_in_loop, has_vectorized_ops
        if isinstance(node, (ast.For, ast.While)):
            if in_loop:
                has_nested_loops = True
            for child in ast.iter_child_nodes(node):
                check_node(child, in_loop=True)
        elif isinstance(node, ast.Subscript) and in_loop:
            has_subscript_in_loop = True
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in ("mm", "matmul", "dot", "sum", "mean", "sigmoid"):
                    has_vectorized_ops = True
            for child in ast.iter_child_nodes(node):
                check_node(child, in_loop)
        else:
            for child in ast.iter_child_nodes(node):
                check_node(child, in_loop)

    check_node(func_node)

    if has_nested_loops and has_subscript_in_loop:
        return "element_wise"
    elif has_vectorized_ops:
        return "vectorized"
    else:
        return "sequential"


def extract_operations(func_node: ast.FunctionDef) -> list:
    ops = set()
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                ops.add(node.func.attr)
            elif isinstance(node.func, ast.Name):
                ops.add(node.func.id)
        if isinstance(node, ast.BinOp):
            ops.add(f"binary_{type(node.op).__name__}")
        if isinstance(node, ast.AugAssign):
            ops.add(f"augassign_{type(node.op).__name__}")
    return list(ops)


def extract_returns(func_node: ast.FunctionDef) -> list:
    returns = []
    for node in ast.walk(func_node):
        if isinstance(node, ast.Return) and node.value:
            returns.append(ast.unparse(node.value))
    return returns


def run_ast_parser(script_path: str, bottleneck_functions: list) -> dict[str, ASTResult]:
    source = Path(script_path).read_text()
    results = {}
    for func_name in bottleneck_functions:
        print(f"[AST Parser Agent] Parsing function: {func_name}")
        results[func_name] = parse_function(source, func_name)
    return results


if __name__ == "__main__":
    import json
    script = sys.argv[1] if len(sys.argv) > 1 else "sample_script.py"
    bottlenecks = ["inefficient_loop", "matrix_ops"]
    output = run_ast_parser(script, bottlenecks)

    print("\n" + "=" * 50)
    print("AST PARSER AGENT OUTPUT")
    print("=" * 50)
    print("\n[AST PARSER AGENT OUTPUT]")
    print(json.dumps({k: v.model_dump() for k, v in output.items()}, indent=2))