"""Code execution tools for LLM agents."""

from __future__ import annotations

import os
import selectors
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Optional

from sciagent.tool.base import BaseTool, ToolReturnType, check, tool


class PythonCodingTool(BaseTool):
    """Expose a tool that executes Python code in an isolated subprocess."""

    name: str = "python_coding"

    @check
    def __init__(
        self,
        *,
        default_timeout: Optional[float] = None,
        working_directory: Optional[str] = None,
        environment: Optional[Dict[str, str]] = None,
        require_approval: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the coding tool.

        Parameters
        ----------
        default_timeout : float, optional
            Timeout (in seconds) applied when executing code unless a call
            overrides it.
        working_directory : str, optional
            Working directory for executed code. Defaults to the current
            working directory when the tool is instantiated.
        environment : Dict[str, str], optional
            Environment variables to overlay on top of the current process
            environment when running code.
        """
        self._default_timeout = default_timeout
        self._working_directory = working_directory or os.getcwd()
        self._environment = environment or {}

        super().__init__(require_approval=require_approval, **kwargs)

    @tool(name="execute_python_code", return_type=ToolReturnType.DICT)
    def execute_code(
        self,
        code: str,
        *,
        timeout: Optional[float] = None,
        cwd: Optional[str] = None,
        input_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute Python code in a subprocess and capture the result.

        Parameters
        ----------
        code : str
            Python source code to execute.
        timeout : float, optional
            Timeout (in seconds) for this execution. Falls back to the tool
            default when omitted.
        cwd : str, optional
            Working directory for this execution. Defaults to the tool's
            configured working directory.
        input_text : str, optional
            Text supplied to the subprocess via standard input.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing stdout, stderr, returncode, and a
            timeout flag. If execution fails before the subprocess starts,
            an error message is included.
        """
        if not isinstance(code, str):
            raise TypeError("code must be a string containing Python source")

        exec_timeout = timeout if timeout is not None else self._default_timeout
        exec_cwd = cwd or self._working_directory
        env = os.environ.copy()
        env.update(self._environment)

        tmp_file = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
        stderr_file = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False)
        try:
            tmp_file.write(code)
            tmp_file.flush()
            tmp_file.close()

            process = subprocess.Popen(
                [sys.executable, tmp_file.name],
                stdout=subprocess.PIPE,
                stderr=stderr_file,
                stdin=subprocess.PIPE if input_text is not None else None,
                text=True,
                cwd=exec_cwd,
                env=env,
            )

            if input_text is not None and process.stdin is not None:
                process.stdin.write(input_text)
                process.stdin.close()

            stdout_chunks: list[str] = []
            selector = selectors.DefaultSelector()
            if process.stdout is not None:
                selector.register(process.stdout, selectors.EVENT_READ)

            start_time = time.monotonic()
            timed_out = False

            while True:
                if exec_timeout is not None:
                    remaining = exec_timeout - (time.monotonic() - start_time)
                    if remaining <= 0:
                        timed_out = True
                        process.kill()
                        break
                else:
                    remaining = None

                events = selector.select(timeout=remaining)
                for key, _ in events:
                    line = key.fileobj.readline()
                    if line == "":
                        selector.unregister(key.fileobj)
                        continue
                    print(line, end="")
                    stdout_chunks.append(line)

                if process.poll() is not None:
                    break

            if process.stdout is not None:
                remaining_output = process.stdout.read()
                if remaining_output:
                    print(remaining_output, end="")
                    stdout_chunks.append(remaining_output)

            process.wait()

            stderr_file.seek(0)
            stderr_text = stderr_file.read()

            if timed_out:
                return {
                    "stdout": "".join(stdout_chunks),
                    "stderr": stderr_text,
                    "returncode": None,
                    "timeout": True,
                    "error": f"Execution timed out after {exec_timeout} seconds",
                }

            return {
                "stdout": "".join(stdout_chunks),
                "stderr": stderr_text,
                "returncode": process.returncode,
                "timeout": False,
            }
        except Exception as exc:  # pragma: no cover - best effort reporting
            return {
                "stdout": "",
                "stderr": "",
                "returncode": None,
                "timeout": False,
                "error": str(exc),
            }
        finally:
            try:
                os.unlink(tmp_file.name)
            except OSError:
                pass
            try:
                stderr_file.close()
                os.unlink(stderr_file.name)
            except OSError:
                pass


class BashCodingTool(BaseTool):
    """Expose a tool that executes Bash code in an isolated subprocess."""

    name: str = "bash_coding"

    @check
    def __init__(
        self,
        *,
        default_timeout: Optional[float] = None,
        working_directory: Optional[str] = None,
        environment: Optional[Dict[str, str]] = None,
        shell_path: str = "/bin/bash",
        require_approval: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the Bash coding tool."""
        self._default_timeout = default_timeout
        self._working_directory = working_directory or os.getcwd()
        self._environment = environment or {}
        self._shell_path = shell_path

        super().__init__(require_approval=require_approval, **kwargs)

    @tool(name="execute_bash_code", return_type=ToolReturnType.DICT)
    def execute_code(
        self,
        code: str,
        *,
        timeout: Optional[float] = None,
        cwd: Optional[str] = None,
        input_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute Bash code in a subprocess and capture the result."""
        if not isinstance(code, str):
            raise TypeError("code must be a string containing Bash source")

        exec_timeout = timeout if timeout is not None else self._default_timeout
        exec_cwd = cwd or self._working_directory
        env = os.environ.copy()
        env.update(self._environment)

        tmp_file = tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False)
        stderr_file = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False)
        try:
            tmp_file.write(code)
            tmp_file.flush()
            tmp_file.close()

            process = subprocess.Popen(
                [self._shell_path, tmp_file.name],
                stdout=subprocess.PIPE,
                stderr=stderr_file,
                stdin=subprocess.PIPE if input_text is not None else None,
                text=True,
                cwd=exec_cwd,
                env=env,
            )

            if input_text is not None and process.stdin is not None:
                process.stdin.write(input_text)
                process.stdin.close()

            stdout_chunks: list[str] = []
            selector = selectors.DefaultSelector()
            if process.stdout is not None:
                selector.register(process.stdout, selectors.EVENT_READ)

            start_time = time.monotonic()
            timed_out = False

            while True:
                if exec_timeout is not None:
                    remaining = exec_timeout - (time.monotonic() - start_time)
                    if remaining <= 0:
                        timed_out = True
                        process.kill()
                        break
                else:
                    remaining = None

                events = selector.select(timeout=remaining)
                for key, _ in events:
                    line = key.fileobj.readline()
                    if line == "":
                        selector.unregister(key.fileobj)
                        continue
                    print(line, end="")
                    stdout_chunks.append(line)

                if process.poll() is not None:
                    break

            if process.stdout is not None:
                remaining_output = process.stdout.read()
                if remaining_output:
                    print(remaining_output, end="")
                    stdout_chunks.append(remaining_output)

            process.wait()

            stderr_file.seek(0)
            stderr_text = stderr_file.read()

            if timed_out:
                return {
                    "stdout": "".join(stdout_chunks),
                    "stderr": stderr_text,
                    "returncode": None,
                    "timeout": True,
                    "error": f"Execution timed out after {exec_timeout} seconds",
                }

            return {
                "stdout": "".join(stdout_chunks),
                "stderr": stderr_text,
                "returncode": process.returncode,
                "timeout": False,
            }
        except Exception as exc:  # pragma: no cover - best effort reporting
            return {
                "stdout": "",
                "stderr": "",
                "returncode": None,
                "timeout": False,
                "error": str(exc),
            }
        finally:
            try:
                os.unlink(tmp_file.name)
            except OSError:
                pass
            try:
                stderr_file.close()
                os.unlink(stderr_file.name)
            except OSError:
                pass
