import ast
import asttokens
from pathlib import Path
import sys
sys.path.append(".")

from kairos_lab.models import ASTResult


def parse_function(source_code: str, function_name: str, line_hint: int = None) -> ASTResult:
    atok = asttokens.ASTTokens(source_code, parse=True)
    tree = atok.tree

    best_match = None

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            if best_match is None:
                best_match = node  
            if line_hint is not None and abs(node.lineno - line_hint) < 5:
                best_match = node 
                break

    if best_match is None:
        return ASTResult(
            function_name=function_name,
            found=False,
            args=[], loop_count=0, max_nesting_depth=0,
            data_types=[], memory_access_pattern="unknown",
            operations=[], returns=[], start_line=0, end_line=0
        )

    loop_count, max_nesting_depth = analyze_loops(best_match)

    return ASTResult(
        function_name=function_name,
        found=True,
        args=[arg.arg for arg in best_match.args.args],
        loop_count=loop_count,
        max_nesting_depth=max_nesting_depth,
        data_types=detect_data_types(best_match),
        memory_access_pattern=detect_memory_access_pattern(best_match),
        operations=extract_operations(best_match),
        returns=extract_returns(best_match),
        start_line=best_match.lineno,
        end_line=best_match.end_lineno
    )


def analyze_loops(func_node):
    loop_count = 0
    max_nesting_depth = 0

    def walk_depth(node, depth=0):
        nonlocal loop_count, max_nesting_depth
        if isinstance(node, (ast.For, ast.While)):
            loop_count += 1
            max_nesting_depth = max(max_nesting_depth, depth + 1)
            for child in ast.iter_child_nodes(node):
                walk_depth(child, depth + 1)
        else:
            for child in ast.iter_child_nodes(node):
                walk_depth(child, depth)

    walk_depth(func_node)
    return loop_count, max_nesting_depth


def detect_data_types(func_node):
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


def detect_memory_access_pattern(func_node):
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


def extract_operations(func_node):
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


def extract_returns(func_node):
    returns = []
    for node in ast.walk(func_node):
        if isinstance(node, ast.Return) and node.value:
            returns.append(ast.unparse(node.value))
    return returns


def run_ast_parser(script_path: str, bottleneck_functions: list, line_hints: dict = None) -> dict[str, ASTResult]:
    
    # Find all Python files in the project
    script = Path(script_path)
    project_folder = script.parent
    all_files = list(project_folder.rglob("*.py"))
    if script not in all_files:
        all_files.append(script)

    # Combine all source into one searchable map
    function_sources = {}
    for f in all_files:
        if f.name == "__init__.py":
            continue
        source = f.read_text()
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    function_sources[node.name] = source
        except SyntaxError:
            continue

    line_hints = line_hints or {}
    results = {}

    for func_name in bottleneck_functions:
        print(f"[AST Parser Agent] Parsing function: {func_name}")
        source = function_sources.get(func_name)
        if source:
            results[func_name] = parse_function(source, func_name, line_hints.get(func_name))
        else:
            results[func_name] = ASTResult(
                function_name=func_name, found=False,
                args=[], loop_count=0, max_nesting_depth=0,
                data_types=[], memory_access_pattern="unknown",
                operations=[], returns=[], start_line=0, end_line=0
            )

    return results

if __name__ == "__main__":
    import json
    import sys
    sys.path.append(".")
    from kairos_lab.agents.profiler import run_profiler

    script = sys.argv[1] if len(sys.argv) > 1 else "sample_script.py"
    
    # Get real bottlenecks from Profiler
    profiler_output = run_profiler(script)
    bottlenecks = profiler_output.top_functions

    print(f"\n[AST Parser] Using profiler bottlenecks: {bottlenecks}")
    output = run_ast_parser(script, bottlenecks)

    print("\n" + "=" * 50)
    print("AST PARSER AGENT OUTPUT")
    print("=" * 50)
    print(json.dumps({k: v.model_dump() for k, v in output.items()}, indent=2))