from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class StepResult(str, Enum):
    SUCCESS = "SUCCESS"
    NEXT_STEP = "NEXT_STEP"
    INSUFFICIENT = "INSUFFICIENT"
    RETRY = "RETRY"
    FALLBACK = "FALLBACK"
    WAITING_FOR_APPROVAL = "WAITING_FOR_APPROVAL"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"

class StepInfo(BaseModel):
    step_id: str
    description: str
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    result: Optional[str] = None
    status: Optional[StepResult] = None

class WorkflowState(BaseModel):
    request_id: str
    original_request: str
    current_step_index: int = 0
    plan: List[StepInfo] = Field(default_factory=list)
    completed_steps: List[StepInfo] = Field(default_factory=list)
    failures: List[str] = Field(default_factory=list)
    decisions: List[str] = Field(default_factory=list)
    context_data: Dict[str, Any] = Field(default_factory=dict)
