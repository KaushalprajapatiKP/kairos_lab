import cProfile
import pstats
import io
import requests
from pathlib import Path
import sys
sys.path.append(".")

from kairos_lab.models import ProfilerOutput

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "deepseek-r1:7b"


def profile_script(script_path: str) -> str:
    profiler = cProfile.Profile()
    script = Path(script_path).read_text()
    namespace = {"__name__": "__main__"}

    profiler.enable()
    exec(compile(script, script_path, 'exec'), namespace)
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('tottime')
    stats.print_stats(50)

    raw = stream.getvalue()

    filtered = []
    for line in raw.split('\n'):
        if 'site-packages' in line:
            continue
        if 'frozen importlib' in line:
            continue
        filtered.append(line)

    return '\n'.join(filtered)


def ask_llm(profile_output: str) -> str:
    prompt = f"""You are an expert Python performance engineer.

Below is cProfile output from a Python ML script.
Identify the TOP 3 bottlenecks in the USER'S code only.

For each bottleneck return:
1. Function name and line number
2. WHY it is slow (be specific)
3. ONE concrete optimization suggestion using NumPy, PyTorch vectorization, or Triton

PROFILE OUTPUT:
{profile_output}
"""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    return response.json()["response"]


def extract_top_functions(profile_raw: str) -> list[str]:
    """Extract top user-defined function names from cProfile output."""
    
    STDLIB_EXCLUDE = {
        'get_annotations', 'create_dynamic', 'exec_dynamic',
        '_process_class', '_create_fn', 'update_wrapper',
        '_signature_from_callable', '_signature_from_function',
        '_joinrealpath', 'loads', 'read', 'readlines', 'open_code',
        'stat', 'lstat', 'listdir', 'join', 'startswith',
        '__build_class__', '__new__', 'isinstance', 'getattr',
        'hasattr', 'setattr', 'len', 'any', 'append', 'item'
    }

    functions = []
    for line in profile_raw.split('\n'):
        # Only look at lines from user project files
        if ('sample_script' in line or 
            'sample_project' in line or
            ('kairos_lab' not in line and '.py:' in line and 
             'site-packages' not in line and
             'uv/python' not in line)):
            parts = line.strip().split()
            if parts:
                last = parts[-1]
                if '(' in last and ')' in last:
                    func_name = last.split('(')[-1].rstrip(')')
                    if (func_name not in ('', '<module>') and 
                        func_name not in STDLIB_EXCLUDE and
                        func_name not in functions):
                        functions.append(func_name)

    return functions[:3]


def run_profiler(script_path: str) -> ProfilerOutput:
    """Main entry point. Returns ProfilerOutput Pydantic model."""

    print(f"[Profiler Agent] Profiling: {script_path}")
    print("[Profiler Agent] Running cProfile...")
    profile_raw = profile_script(script_path)

    print("[Profiler Agent] Asking LLM for bottleneck analysis...")
    bottlenecks_raw = ask_llm(profile_raw)

    top_functions = extract_top_functions(profile_raw)
    print(f"[Profiler Agent] Top functions identified: {top_functions}")

    return ProfilerOutput(
        script_path=script_path,
        profile_raw=profile_raw,
        bottlenecks_raw=bottlenecks_raw,
        top_functions=top_functions
    )


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "sample_script.py"
    output = run_profiler(path)

    print("\n" + "=" * 50)
    print("PROFILER AGENT OUTPUT")
    print("=" * 50)
    print("\n[PROFILE RAW]")
    print(output.profile_raw)
    print("\n[LLM ANALYSIS]")
    print(output.bottlenecks_raw)
    print("\n[TOP FUNCTIONS]")
    print(output.top_functions)