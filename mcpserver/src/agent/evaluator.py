from .state import StepResult, StepInfo, WorkflowState

class ResultEvaluator:
    def evaluate(self, step: StepInfo, result: str) -> StepResult:
        # Evaluate the result string to determine the next state
        result_upper = result.upper()
        if "ERROR" in result_upper or "FAIL" in result_upper or "EXCEPTION" in result_upper:
            return StepResult.FAILED
        if "INSUFFICIENT" in result_upper:
            return StepResult.INSUFFICIENT
        return StepResult.SUCCESS
