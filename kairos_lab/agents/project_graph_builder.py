import ast
import sys
from pathlib import Path
import networkx as nx
sys.path.append(".")

from kairos_lab.models import ProjectGraphOutput


def extract_all_functions(tree: ast.AST) -> list[str]:
    """Extract all function names defined in the script."""
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions.append(node.name)
    return functions


def extract_function_calls_with_lines(func_node: ast.FunctionDef) -> list[dict]:
    """Extract all function calls with line numbers from inside a function."""
    calls = []
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.append({
                    "called": node.func.id,
                    "line": node.lineno
                })
            elif isinstance(node.func, ast.Attribute):
                calls.append({
                    "called": node.func.attr,
                    "line": node.lineno
                })
    return calls


def build_call_graph(source_code: str) -> tuple[dict[str, list[str]], dict[str, list[dict]]]:
    """
    Build two structures:
    - call_graph: function -> list of internal functions it calls
    - call_sites: function -> list of {called, line} dicts
    """
    tree = ast.parse(source_code)
    all_functions = set(extract_all_functions(tree))
    call_graph = {}
    call_sites = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            all_calls = extract_function_calls_with_lines(node)

            # call_graph: only internal function calls, deduplicated
            internal_calls = list(set([
                c["called"] for c in all_calls
                if c["called"] in all_functions and c["called"] != node.name
            ]))

            # call_sites: all internal calls with line numbers, including duplicates
            internal_sites = [
                c for c in all_calls
                if c["called"] in all_functions and c["called"] != node.name
            ]

            call_graph[node.name] = internal_calls
            call_sites[node.name] = internal_sites

    return call_graph, call_sites


def find_entry_points(call_graph: dict[str, list[str]]) -> list[str]:
    """Entry points are functions nobody else calls."""
    all_functions = set(call_graph.keys())
    called_by_others = set()
    for func, calls in call_graph.items():
        for called in calls:
            called_by_others.add(called)
    return sorted(list(all_functions - called_by_others))


def find_leaf_functions(call_graph: dict[str, list[str]]) -> list[str]:
    """Leaf functions are functions that call no other internal functions."""
    return sorted([f for f, calls in call_graph.items() if len(calls) == 0])


def run_project_graph_builder(script_path: str) -> ProjectGraphOutput:
    """Main entry point. Returns ProjectGraphOutput Pydantic model."""

    print(f"[Project Graph Builder] Scanning: {script_path}")

    source = Path(script_path).read_text()
    tree = ast.parse(source)

    functions_found = extract_all_functions(tree)
    print(f"[Project Graph Builder] Functions found: {functions_found}")

    call_graph, call_sites = build_call_graph(source)

    # Build NetworkX graph
    G = nx.DiGraph()
    for func, calls in call_graph.items():
        G.add_node(func)
        for called in calls:
            G.add_edge(func, called)

    entry_points = find_entry_points(call_graph)
    leaf_functions = find_leaf_functions(call_graph)

    print(f"[Project Graph Builder] Entry points: {entry_points}")
    print(f"[Project Graph Builder] Leaf functions: {leaf_functions}")
    print(f"[Project Graph Builder] Call sites mapped: {sum(len(v) for v in call_sites.values())} total")

    return ProjectGraphOutput(
        script_path=script_path,
        functions_found=functions_found,
        call_graph=call_graph,
        entry_points=entry_points,
        leaf_functions=leaf_functions,
        call_sites=call_sites
    )


if __name__ == "__main__":
    import json
    script = sys.argv[1] if len(sys.argv) > 1 else "sample_script.py"
    output = run_project_graph_builder(script)

    print("\n" + "=" * 50)
    print("PROJECT GRAPH BUILDER OUTPUT")
    print("=" * 50)
    print(json.dumps(output.model_dump(), indent=2))