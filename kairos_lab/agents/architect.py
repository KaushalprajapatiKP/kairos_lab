from pathlib import Path
import json


# Decision rules for Parallelization strategy

RULES = {
    "triton" : {
        "memory_patterns" : ["element_wise"],
        "data_types" : ["torch_tensor", "indexed_access"],
        "min_loop_depth" : 2,
        "reason" : "Nested loops with element-wise tensor access. Triton kernel will parallelise across rows and columns."
        },
    "cuda" : {
        "memory_patterns" : ["vectorized"],
        "data_types" : ["torch_tensor"],
        "min_loop_depth" : 0,
        "reason" : "Vectorized operations on tensor with indexed access. CUDA kernel will parallelise across rows and columns."
        },
    "numba" : {
        "memory_patterns" : ["sequential","element_wise"],
        "data_types" : ["python_list", "indexed_access"],
        "min_loop_depth" : 1,
        "reason" : "Simple loops with Python lists or scalar ops. Numba JIT will give significant speedup with minimal rewrite."
    }
}

def decide_strategy(ast_result: dict) -> dict:
    """
    Given AST Parser output for one function, 
    decide the best parallelization strategy.
    """

    pattern = ast_result.get("memory_access_pattern", "unknown")
    data_types = ast_result.get("data_types", [])
    loop_depth = ast_result.get("max_nesting_depth", 0)
    func_name = ast_result.get("function_name", "unknown")

    decision = {
        "function" : func_name,
        "strategy" : None,
        "confidence" : "low",
        "reason" : "",
        "action" : ""
    }

    # Rule 1 : nested loops + tensor indexed access = Triton
    if (pattern == "element_wise" and "torch_tensor" in data_types and loop_depth >= RULES["triton"]["min_loop_depth"]):
        decision["strategy"] = "triton"
        decision["confidence"] = "high"
        decision["reason"] = RULES["triton"]["reason"]
        decision["action"] = f"Generate Triton kernel for {func_name} that parallelizes across tensor dimensions."

    # Rule 2 : element wise with python lists = Numba
    elif (pattern == "element_wise" and "python_list" in data_types and loop_depth >= RULES["numba"]["min_loop_depth"]):
        decision["strategy"] = "numba"
        decision["confidence"] = "high"
        decision["reason"] = RULES["numba"]["reason"]
        decision["action"] = f"Apply @numba.njit decorator to {func_name} for JIT compilation and replace list with numpy array."

    # Rule 3 : vectorized torch ops = CUDA fusion
    elif (pattern == "vectorized" and "torch_tensor" in data_types):
        decision["strategy"] = "cuda"
        decision["confidence"] = "medium"
        decision["reason"] = RULES["cuda"]["reason"]
        decision["action"] = f"Write fused CUDA kernel for {func_name} to reduce memory bandwidth."
    
    # Rule 4 : vectorized but no tensors detected = Numba
    elif (pattern == "vectorized"):
        decision["strategy"] = "numba"
        decision["confidence"] = "medium"
        decision["reason"] = "Vectorized pattern without explicit tensor ops. Numba is safest bet."
        decision["action"] = f"Apply @numba.jit to {func_name}."
    
    else :
        decision["strategy"] = "numba"
        decision["confidence"] = "low"
        decision["reason"] = "Pattern unclear. Defaulting to Numba as lowest-risk option."
        decision["action"] = f"Apply @numba.jit to {func_name} as first optimization attempt."

    return decision

def run_architect(ast_output: dict) -> dict:
    """
    Main Entry Point for the Architect Agent.
    Takes AST Parser output and returns a strategy per function.
    """

    decisions = {}

    for func_name, ast_result in ast_output.items():
        if not ast_result.get("found"):
            print(f"[Architect Agent] Skipping {func_name} — not found in AST")
            continue
        
        decision = decide_strategy(ast_result)
        decisions[func_name] = decision
        print(f"[Architect Agent] {func_name} → {decision['strategy'].upper()} ({decision['confidence']} confidence)")

    return decisions

if __name__ == "__main__":
    import sys
    sys.path.append(".")

    from kairos_lab.agents.ast_parser import run_ast_parser

    script = sys.argv[1] if len(sys.argv) > 1 else "sample_script.py"
    bottlenecks = ["inefficient_loop", "matrix_ops"]

    # Run AST Parser first
    print(f"[Architect Agent] Running AST Parser for bottleneck functions...")
    ast_output = run_ast_parser(script, bottlenecks)

    # Run Architect Agent
    print(f"[Architect Agent] Running Architect Agent...")
    decisions = run_architect(ast_output)

    # Print results
    print("\n" + "="*50)
    print("ARCHITECT AGENT OUTPUT")
    print("="*50)
    print(json.dumps(decisions, indent=2))
