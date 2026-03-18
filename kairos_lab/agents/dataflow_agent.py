import ast
import sys
from pathlib import Path
sys.path.append(".")

from kairos_lab.models import DataflowOutput


def extract_keyword_args(call_node: ast.Call) -> dict:
    """Extract keyword arguments from a function call node."""
    kwargs = {}
    for kw in call_node.keywords:
        if isinstance(kw.value, ast.Constant):
            kwargs[kw.arg] = kw.value.value
        elif isinstance(kw.value, ast.Name):
            kwargs[kw.arg] = kw.value.id
    return kwargs


def extract_positional_args(call_node: ast.Call) -> list:
    """Extract positional arguments from a function call node."""
    args = []
    for arg in call_node.args:
        if isinstance(arg, ast.Constant):
            args.append(arg.value)
        elif isinstance(arg, ast.Name):
            args.append(arg.id)
    return args


def find_class_instantiations(project_folder: Path) -> dict[str, dict]:
    """
    Scan all files for class instantiations and extract their constructor args.
    Returns {class_name: {arg_name: value}}
    """
    instantiations = {}

    for f in project_folder.rglob("*.py"):
        if f.name == "__init__.py":
            continue
        try:
            source = f.read_text()
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        class_name = node.func.id
                        kwargs = extract_keyword_args(node)
                        if kwargs and class_name[0].isupper():
                            instantiations[class_name] = kwargs
        except SyntaxError:
            continue

    return instantiations


def find_function_call_args(project_folder: Path, function_name: str) -> list[str]:
    """
    Find what variable names are passed to a function at its call sites.
    Returns list of argument variable names.
    """
    for f in project_folder.rglob("*.py"):
        if f.name == "__init__.py":
            continue
        try:
            source = f.read_text()
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = None
                    if isinstance(node.func, ast.Name):
                        name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        name = node.func.attr
                    if name == function_name:
                        return [ast.unparse(a) for a in node.args]
        except SyntaxError:
            continue

    return []


def trace_variable_shape(
    project_folder: Path,
    var_name: str,
    class_instantiations: dict,
    dataloader_args: dict
) -> tuple[list, str]:
    """
    Trace a variable name back to its origin and infer shape and type.
    Returns (shape, dtype).
    """

    # Known dataloader-derived variables
    batch_size = dataloader_args.get("batch_size", 32)
    feature_dim = dataloader_args.get("feature_dim", 128)

    if var_name == "features":
        return [batch_size, feature_dim], "torch.float32"

    if var_name == "labels":
        return [batch_size], "torch.int64"

    if var_name == "predictions":
        # predictions = model(features) — shape is [batch_size, output_dim]
        mlp_args = class_instantiations.get("SlowMLP", {})
        output_dim = mlp_args.get("output_dim", 10)
        return [batch_size, output_dim], "torch.float32"

    if var_name == "x":
        # x inside forward — comes from fc1 output
        # fc1 = nn.Linear(input_dim, hidden_dim)
        mlp_args = class_instantiations.get("SlowMLP", {})
        hidden_dim = mlp_args.get("hidden_dim", 256)
        return [batch_size, hidden_dim], "torch.float32"

    if var_name == "self":
        return [], "self"

    # Generic fallback — scan for assignment of this variable
    for f in project_folder.rglob("*.py"):
        if f.name == "__init__.py":
            continue
        try:
            source = f.read_text()
            tree = ast.parse(source)
            for node in ast.walk(tree):
                # Look for torch.randn(a, b) or torch.zeros(a, b)
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == var_name:
                            if isinstance(node.value, ast.Call):
                                if isinstance(node.value.func, ast.Attribute):
                                    if node.value.func.attr in ("randn", "zeros", "ones", "rand"):
                                        pos_args = extract_positional_args(node.value)
                                        if pos_args and all(isinstance(a, int) for a in pos_args):
                                            return pos_args, "torch.float32"
        except SyntaxError:
            continue

    return [32, 128], "torch.float32"


def find_dataloader_args(project_folder: Path) -> dict:
    """Find get_dataloader call and extract its arguments."""
    for f in project_folder.rglob("*.py"):
        if f.name == "__init__.py":
            continue
        try:
            source = f.read_text()
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = None
                    if isinstance(node.func, ast.Name):
                        name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        name = node.func.attr
                    if name == "get_dataloader":
                        kwargs = extract_keyword_args(node)
                        # resolve batch_size if it's a variable name
                        if isinstance(kwargs.get("batch_size"), str):
                            kwargs["batch_size"] = 32
                        return kwargs
        except SyntaxError:
            continue
    return {"batch_size": 32, "feature_dim": 128, "size": 500}


def find_class_for_method(source_code: str, function_name: str) -> str:
    """Find class name containing this method."""
    tree = ast.parse(source_code)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in ast.walk(node):
                if isinstance(item, ast.FunctionDef) and item.name == function_name:
                    return node.name
    return ""


def find_function_source(script_path: str, function_name: str) -> tuple[str, str]:
    """Find source file containing this function."""
    script = Path(script_path)
    project_folder = script.parent
    all_files = list(project_folder.rglob("*.py"))

    for f in all_files:
        if f.name == "__init__.py":
            continue
        try:
            source = f.read_text()
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    return source, str(f)
        except SyntaxError:
            continue
    return "", ""


def analyze_function(
    script_path: str,
    function_name: str,
    class_instantiations: dict,
    dataloader_args: dict
) -> DataflowOutput:
    """Analyze a single function — trace call sites, infer shapes."""

    script = Path(script_path)
    project_folder = script.parent

    source, file_path = find_function_source(script_path, function_name)
    if not source:
        return DataflowOutput(
            function=function_name,
            input_shapes={},
            input_types={},
            class_name=None,
            class_init_args=None
        )

    # Check if method
    class_name = find_class_for_method(source, function_name)

    # Find what variable names are passed at call sites
    call_args = find_function_call_args(project_folder, function_name)
    print(f"[Dataflow Agent] {function_name} call args: {call_args}")

    # Trace each argument back to its shape
    input_shapes = {}
    input_types = {}

    for arg_expr in call_args:
        # Skip self
        if arg_expr == "self":
            continue
        shape, dtype = trace_variable_shape(
            project_folder, arg_expr, class_instantiations, dataloader_args
        )
        input_shapes[arg_expr] = shape
        input_types[arg_expr] = dtype

    # Get class init args if method
    class_init_args = class_instantiations.get(class_name) if class_name else None

    return DataflowOutput(
        function=function_name,
        input_shapes=input_shapes,
        input_types=input_types,
        class_name=class_name if class_name else None,
        class_init_args=class_init_args
    )


def run_dataflow_agent(script_path: str, bottleneck_functions: list) -> dict[str, DataflowOutput]:
    """Main entry point. Traces dataflow for all bottleneck functions."""

    print("[Dataflow Agent] Starting dataflow analysis...")

    script = Path(script_path)
    project_folder = script.parent

    # Step 1 — find all class instantiations across project
    print("[Dataflow Agent] Scanning class instantiations...")
    class_instantiations = find_class_instantiations(project_folder)
    print(f"[Dataflow Agent] Found: {class_instantiations}")

    # Step 2 — find dataloader args for batch size and feature dim
    print("[Dataflow Agent] Finding dataloader configuration...")
    dataloader_args = find_dataloader_args(project_folder)
    print(f"[Dataflow Agent] Dataloader args: {dataloader_args}")

    # Step 3 — analyze each bottleneck function
    results = {}
    for func_name in bottleneck_functions:
        print(f"\n[Dataflow Agent] Analyzing: {func_name}")
        results[func_name] = analyze_function(
            script_path, func_name, class_instantiations, dataloader_args
        )
        r = results[func_name]
        print(f"[Dataflow Agent] {func_name} → shapes: {r.input_shapes} | types: {r.input_types} | class: {r.class_name or 'standalone'}")

    return results


if __name__ == "__main__":
    import json
    from kairos_lab.agents.profiler import run_profiler

    script = sys.argv[1] if len(sys.argv) > 1 else "sample_project/main.py"

    profiler_output = run_profiler(script)
    bottlenecks = profiler_output.top_functions

    print(f"\n[Dataflow Agent] Bottlenecks: {bottlenecks}")
    output = run_dataflow_agent(script, bottlenecks)

    print("\n" + "=" * 50)
    print("DATAFLOW AGENT OUTPUT")
    print("=" * 50)
    print(json.dumps({k: v.model_dump() for k, v in output.items()}, indent=2))