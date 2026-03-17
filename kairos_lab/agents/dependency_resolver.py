import ast
import importlib
import sys
from pathlib import Path
sys.path.append(".")

from kairos_lab.models import DependencyResolverOutput

KNOWN_PACKAGE_MAPS = {
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "PIL": "pillow",
    "torch": "torch",
    "torchvision": "torchvision",
    "torchtext": "torchtext",
    "torchaudio": "torchaudio",
    "torchmetrics": "torchmetrics",
    "torchvision": "torchvision",
    "torchtext": "torchtext",
    "torchaudio": "torchaudio",
    "torchmetrics": "torchmetrics",
    "tl": "triton",
    "numba": "numba",
    "np" : "numpy",
    "pd" : "pandas",
    "plt" : "matplotlib",
    "seaborn" : "seaborn",
    "scipy" : "scipy",
    "statsmodels" : "statsmodels",
    "xgboost" : "xgboost",
    "lightgbm" : "lightgbm",
    "catboost" : "catboost",
    "mlflow" : "mlflow",
}

# Imports we don't need to check - stdlib or always available
STDLIB_SKIP = {
    "os", "sys", "re", "io", "json", "math", "time", "random",
    "pathlib", "typing", "collections", "itertools", "functools",
    "subprocess", "threading", "multiprocessing", "logging",
    "unittest", "abc", "copy", "enum", "dataclasses", "warnings",
    "traceback", "inspect", "ast", "cProfile", "pstats"
}

def extract_imports(source_code: str) -> list[str]:
    """Extract all top-level import statements from source code."""
    tree = ast.parse(source_code)
    imports = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # Get root package name: torch.nn -> torch
                root = alias.name.split('.')[0]
                imports.add(root)
        
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split('.')[0]
                imports.add(root)
    
    imports = imports - STDLIB_SKIP
    return list(imports)

def check_availability(imports: list[str]) -> tuple[list[str], list[str]]:
    """Check which imports are available in current environment."""
    available = []
    missing = []

    for imp in imports:
        try:
            importlib.import_module(imp)
            available.append(imp)
        except ImportError:
            # Try known aliases
            pip_name = KNOWN_PACKAGE_MAPS.get(imp)
            missing.append(f"{imp} (install: pip install {pip_name})")
    return available, missing

def resolve_local_imports(script_path: str, project_root: str = None) -> list[str]:
    """Recursively find all Python files in the project."""
    script = Path(script_path)
    root = Path(project_root) if project_root else script.parent
    
    all_files = [script_path]
    visited = {str(script.resolve())}
    queue = [script]

    while queue:
        current = queue.pop(0)
        source = current.read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.ImportFrom) and node.module:
                module = node.module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name

            if module:
                # Convert module path to file path
                module_path = root / Path(module.replace(".", "/") + ".py")
                if module_path.exists():
                    resolved = str(module_path.resolve())
                    if resolved not in visited:
                        visited.add(resolved)
                        all_files.append(str(module_path))
                        queue.append(module_path)

    return all_files


def run_dependency_resolver(script_path: str) -> DependencyResolverOutput:
    """Main entry point. Returns DependencyResolverOutput Pydantic model."""

    print(f"[Dependency Resolver] Scanning: {script_path}")

    # Find all project files recursively
    project_root = str(Path(script_path).parent.parent)
    all_files = resolve_local_imports(script_path, project_root)
    print(f"[Dependency Resolver] Files scanned: {all_files}")

    # Collect all imports across all files
    all_imports = set()
    for f in all_files:
        source = Path(f).read_text()
        imports = extract_imports(source)
        all_imports.update(imports)

    # Remove local package names
    project_packages = {Path(script_path).parent.name}
    all_imports = all_imports - project_packages

    imports_found = sorted(list(all_imports))
    print(f"[Dependency Resolver] Imports found: {imports_found}")

    available, missing = check_availability(imports_found)

    can_proceed = len(missing) == 0
    warning = None

    if missing:
        warning = f"Missing dependencies: {missing}. Pipeline may fail."
        print(f"[Dependency Resolver] WARNING — {warning}")
    else:
        print(f"[Dependency Resolver] All dependencies available. Safe to proceed.")

    return DependencyResolverOutput(
        script_path=script_path,
        imports_found=imports_found,
        available=available,
        missing=missing,
        can_proceed=can_proceed,
        warning=warning
    )

if __name__ == "__main__":
    import json
    script = sys.argv[1] if len(sys.argv) > 1 else "sample_script.py"
    output = run_dependency_resolver(script)

    print("\n" + "=" * 50)
    print("DEPENDENCY RESOLVER OUTPUT")
    print("=" * 50)
    print(json.dumps(output.model_dump(), indent=2))