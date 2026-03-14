import ast
from pathlib import Path

def parse_function(source_code: str, function_name: str) -> dict:
    """ Parse a function from source code and extract the structure of the function.
    
    Args:
        source_code: The source code of the file.
        function_name: The name of the function to parse.
        
    Returns:
        A dictionary containing the structure of the function.
    """
    
    tree = ast.parse(source_code)

    result = {
        "function_name": function_name,
        "found" : False,
        "args" : [],
        "loop_count" : 0,
        "max_nesting_depth" : 0,
        "data_types" : [],
        "memory_access_pattern" : "unknown",
        "operations" : [],
        "returns" : [],
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            result["found"] = True
            
            # Extract function arguments
            result["args"] = [arg.arg for arg in node.args.args]

            # Count Loops and Nesting Depth
            result["loop_count"], result["max_nesting_depth"] = analyze_loops(node)

            # Detect Data Types
            result["data_types"] = detect_data_types(node)

            # Detect Memory Access Pattern
            result["memory_access_pattern"] = detect_memory_access_pattern(node)

            # Extract Operations
            result["operations"] = extract_operations(node)

            # Extract Return Values
            result["returns"] = extract_returns(node)

            break

    return result


def analyze_loops(func_node: ast.FunctionDef) -> tuple:
    """
        Analyze the loops in a function and return the number of loops and the maximum nesting depth.
    """
    loop_count = 0
    max_nesting_depth = 0

    def walk_depth(node, depth = 0):
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
    """
    Detect the data types used in a function. Returns a list of data types.
    """

    types_found = set()

    for node in ast.walk(func_node):

        # Detect attribute access like tensor[i][j], list.append, etc.
        if isinstance(node, ast.Attribute):
            if node.attr in ("append", "extend"):
                types_found.add("python_list")
            if node.attr in ("item", "shape", "dtype", "cuda", "cpu"):
                types_found.add("torch_tensor")
            if node.attr in ("numpy", "array", "zeros", "ones"):
                types_found.add("numpy_array")
        
        # Detact subscript access like tensor[i][j]
        if isinstance(node, ast.Subscript):
            types_found.add("indexed_access")
        
        # Detect list literals
        if isinstance(node, ast.List):
            types_found.add("python_list")
        
    return list(types_found)


def detect_memory_access_pattern(func_node: ast.FunctionDef) -> str:
    """
    Detect whether memory access is element-wise or vectorized.
    """

    has_nested_loops = False
    has_subscript_in_loop = False
    has_vectorized_ops = False

    def check_node(node, in_loop = False):
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
    """
    Extract the operations performed in a function. Returns a list of operations.
    """

    ops = set()
    
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                ops.add(node.func.attr)
            elif isinstance(node.func, ast.Name):
                ops.add(node.func.id)
        
        if isinstance(node, ast.BinOp):
            op_name = type(node.op).__name__
            ops.add(f"binary_{op_name}")
        
        if isinstance(node, ast.AugAssign):
            op_name = type(node.op).__name__
            ops.add(f"augassign_{op_name}")
    
    return list(ops)
 
def extract_returns(func_node: ast.FunctionDef) -> list:
    """
    Extract the return values from a function. Returns a list of return values.
    """

    returns = []

    for node in ast.walk(func_node):
        if isinstance(node, ast.Return) and node.value:
            returns.append(ast.unparse(node.value))
    
    return returns


def run_ast_parser(script_path: str, bottleneck_functions: list) -> dict:
    """
    Main Entry Point for the AST Parser Agent.
    Parses the script and extracts the structure of the bottleneck functions.
    """

    source = Path(script_path).read_text()

    results = {}

    for func_name in bottleneck_functions:
        print(f"[AST Parser Agent] Parsing function: {func_name}")
        results[func_name] = parse_function(source, func_name)
    
    return results

if __name__ == "__main__":
    import sys
    import json

    script = sys.argv[1] if len(sys.argv) > 1 else "sample_script.py"

    # These comes from Profiler Agent output = hardcoded for now

    bottlenecks = ["inefficient_loop", "matrix_ops"]

    output = run_ast_parser(script, bottlenecks)

    print("\n" + "="*50)
    print("AST PARSER AGENT OUTPUT")
    print("="*50)
    print("\n[AST PARSER AGENT OUTPUT]")
    print(json.dumps(output, indent=2))