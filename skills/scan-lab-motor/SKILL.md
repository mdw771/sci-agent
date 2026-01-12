---
name: scan-lab-motor
description: Launches the lab motor scan task manager to run a full scan workflow and report the results.
---

# Scan Lab Motor Workflow

## Overview
This skill guides the agent to launch a fictional task manager that orchestrates a lab motor scan.
Use it when the user asks to execute a scan workflow or to automate a motor scan pipeline.

## Task Manager Interface
The task manager is assumed to be named `LabMotorScanTaskManager` and exposes a `run` method.

```python
class LabMotorScanTaskManager(BaseTaskManager):
    """Coordinate a lab motor scan workflow.

    The workflow configures a motor controller, performs a scan over the
    requested range, and writes a results summary to the configured output.
    """

    def __init__(self, motor_id: str, start: float, stop: float, steps: int, output_path: str, **kwargs):
        ...

    def run(self) -> dict:
        """Execute the scan and return summary metadata."""
        ...
```

## How to Launch (via coding tool)
1. Use the Python coding tool to import the task manager.
2. Instantiate it with scan parameters.
3. Call `run()` and capture the returned summary.
4. Report the summary to the user and confirm where outputs were written.

### Example
```python
from sciagent.task_manager.lab_motor_scan import LabMotorScanTaskManager

manager = LabMotorScanTaskManager(
    motor_id="motor-a",
    start=0.0,
    stop=10.0,
    steps=101,
    output_path="results/motor_scan.json",
)
summary = manager.run()
print(summary)
```

## Required Inputs
- `motor_id`: Identifier for the motor controller.
- `start`: Scan start position.
- `stop`: Scan stop position.
- `steps`: Number of points in the scan.
- `output_path`: File path for scan output.

## Notes
- If parameters are missing or ambiguous, respond with `NEED HUMAN` and ask a single clarification question.
- When the workflow finishes, respond with `TERMINATE` and summarize results.
