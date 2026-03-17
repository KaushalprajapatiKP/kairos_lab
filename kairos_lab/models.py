from pydantic import BaseModel
from typing import Optional


class BottleneckInfo(BaseModel):
    function: str
    total_time: float
    percentage: float


class ProfilerOutput(BaseModel):
    script_path: str
    profile_raw: str
    bottlenecks_raw: str
    top_functions: list[str]


class ASTResult(BaseModel):
    function_name: str
    found: bool
    args: list[str]
    loop_count: int
    max_nesting_depth: int
    data_types: list[str]
    memory_access_pattern: str
    operations: list[str]
    returns: list[str]
    start_line: int = 0
    end_line: int = 0

class DependencyResolverOutput(BaseModel):
    script_path: str
    imports_found: list[str]
    available: list[str]
    missing: list[str]
    can_proceed: bool
    warning: Optional[str] = None

class ArchitectDecision(BaseModel):
    function: str
    strategy: str  # triton | numba | cuda
    confidence: str  # high | medium | low
    reason: str
    action: str


class GeneratorOutput(BaseModel):
    function: str
    strategy: str
    original_source: str
    optimized_code: str


class VerifierOutput(BaseModel):
    function: str
    strategy: str
    passed: bool
    correct: bool
    faster: bool
    original_time: Optional[float] = None
    optimized_time: Optional[float] = None
    speedup: Optional[float] = None
    error: Optional[str] = None

class ProjectGraphOutput(BaseModel):
    script_path: str
    functions_found: list[str]
    call_graph: dict[str, list[str]]
    entry_points: list[str]
    leaf_functions: list[str]
    call_sites: dict[str, list[dict]]  


class PipelineState(BaseModel):
    script_path: str
    profiler_output: Optional[ProfilerOutput] = None
    ast_output: Optional[dict[str, ASTResult]] = None
    architect_output: Optional[dict[str, ArchitectDecision]] = None
    generator_output: Optional[dict[str, GeneratorOutput]] = None
    verifier_output: Optional[dict[str, VerifierOutput]] = None
    approved: bool = False
    final_output: Optional[dict] = None
    dependency_output: Optional[DependencyResolverOutput] = None
    project_graph_output: Optional[ProjectGraphOutput] = None