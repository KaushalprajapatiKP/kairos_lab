from argparse import Namespace
import cProfile
import pstats
import io
import requests
import json
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"

# def profile_script(script_path: str) -> pstats.Stats:
#     """ Run cProfile on a Python script and returns the profile stats."""
#     profiler = cProfile.Profile()

#     script = Path(script_path).read_text()
#     namespace = {}

#     profiler.enable()
#     exec(compile(script, script_path, "exec"), namespace)
#     profiler.disable()

#     stream = io.StringIO()
#     stats = pstats.Stats(profiler, stream=stream)
#     stats.sort_stats("cumulative")
#     stats.print_stats(20)
    
#     return stream.getvalue()

def profile_script(script_path: str) -> str:
    profiler = cProfile.Profile()
    
    script = Path(script_path).read_text()
    namespace = {}

    exec("import torch", namespace)
    
    profiler.enable()
    exec(compile(script, script_path, 'exec'), namespace)
    profiler.disable()
    
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('tottime')  
    
    # Filter to only show user code, not library internals
    stats.print_stats(20)
    
    raw = stream.getvalue()
    
    filtered_lines = []
    for line in raw.split('\n'):
        if (script_path in line or 
            'sample_script' in line or
            line.strip().startswith('ncalls') or
            line.strip().startswith('Ordered') or
            line.strip() == '' or
            'function calls' in line):
            filtered_lines.append(line)
    
    filtered = '\n'.join(filtered_lines)
    
    # Fall back to raw if filter removed everything
    return filtered if len(filtered_lines) > 5 else raw


def ask_llm(profile_output: str) -> str:
    """ Ask the LLM to analyze the profile output and get bottleneck analysis."""
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
        "stream": False,
    }

    response = requests.post(OLLAMA_URL, json=payload)
    return response.json()["response"]


def run_profiler(script_path: str) -> str:
    """ Main Entry Point for the Profiler Agent."""
    print(f"[Profiler Agent] Profiling script: {script_path}")

    # step 1: profile the script
    print(f"[Profiler Agent] Profiling script via cProfile...")
    profile_output = profile_script(script_path)
    print(f"[Profiler Agent] cProfile output: {profile_output}")

    # step 2: ask the LLM for bottleneck analysis
    print(f"[Profiler Agent] Asking LLM for bottleneck analysis...")
    analysis = ask_llm(profile_output)
    
    # step 3: return the structured analysis
    result = {
        "script" : script_path,
        "profile_raw" : profile_output,
        "bottlenecks" : analysis,
    }

    return result


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "sample_script.py"
    output = run_profiler(path)

    print("\n" + "="*50)
    print("PROFILER AGENT OUTPUT")
    print("="*50)
    print("\n[RAW PROFILE - TOP 20 FUNCTIONS]")
    print(output["profile_raw"])
    print("\n[LLM BOTTLENECK ANALYSIS]")
    print(output["bottlenecks"])