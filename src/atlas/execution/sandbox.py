"""Sandboxed code execution for backtesting."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Whitelist of allowed imports for backtesting code
ALLOWED_IMPORTS = {
    # Core
    "json",
    "math",
    "datetime",
    "collections",
    "itertools",
    "functools",
    "typing",
    # Data/numeric
    "numpy",
    "np",
    "pandas",
    "pd",
    # Finance
    "yfinance",
    "yf",
    "backtesting",
    "vectorbt",
    "vbt",
    "pandas_ta",
    "ta",
    # Stats
    "scipy",
    "scipy.stats",
    "scipy.optimize",
    # Warnings (commonly used)
    "warnings",
}

# Dangerous modules that should never be imported
BLOCKED_IMPORTS = {
    "os",
    "sys",
    "subprocess",
    "shutil",
    "socket",
    "requests",
    "urllib",
    "http",
    "ftplib",
    "smtplib",
    "pickle",
    "shelve",
    "marshal",
    "importlib",
    "__builtins__",
    "builtins",
    "eval",
    "exec",
    "compile",
    "open",  # Will be handled specially
    "input",
    "breakpoint",
}


@dataclass
class ExecutionConfig:
    """Configuration for sandboxed execution."""
    timeout: int = 120  # seconds
    max_memory_mb: int = 1024  # MB
    max_output_size: int = 10 * 1024 * 1024  # 10MB
    working_dir: Path | None = None


@dataclass
class ExecutionResult:
    """Result from code execution."""
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    execution_time: float
    output_data: dict[str, Any] | None = None
    error: str | None = None


def _create_wrapper_code(user_code: str) -> str:
    """Wrap user code with safety checks and output capture."""
    return f'''
import sys
import json
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Redirect to capture output
_output_buffer = []
_original_print = print

def _safe_print(*args, **kwargs):
    """Capture print output."""
    import io
    buffer = io.StringIO()
    kwargs['file'] = buffer
    _original_print(*args, **kwargs)
    _output_buffer.append(buffer.getvalue())

# Replace print
print = _safe_print

# Execute user code
try:
{_indent_code(user_code, 4)}
except Exception as _e:
    import traceback
    _error_msg = traceback.format_exc()
    _original_print(json.dumps({{"error": str(_e), "traceback": _error_msg}}))
    sys.exit(1)

# Output captured prints
for _line in _output_buffer:
    _original_print(_line, end='')
'''


def _indent_code(code: str, spaces: int) -> str:
    """Indent code by specified number of spaces."""
    indent = " " * spaces
    return "\n".join(indent + line for line in code.splitlines())


def _validate_code(code: str) -> tuple[bool, str | None]:
    """Validate code for dangerous operations.

    Returns (is_valid, error_message).
    """
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    # Check for dangerous operations
    for node in ast.walk(tree):
        # Check imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name.split('.')[0]
                if module in BLOCKED_IMPORTS:
                    return False, f"Import of '{module}' is not allowed"

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module = node.module.split('.')[0]
                if module in BLOCKED_IMPORTS:
                    return False, f"Import from '{module}' is not allowed"

        # Check for eval/exec calls
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ('eval', 'exec', 'compile', 'open', '__import__'):
                    return False, f"Use of '{node.func.id}' is not allowed"

        # Check for attribute access to dangerous modules
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                if node.value.id in ('os', 'sys', 'subprocess'):
                    return False, f"Access to '{node.value.id}' module is not allowed"

    return True, None


class SandboxedExecutor:
    """Execute Python code in a sandboxed subprocess."""

    def __init__(self, config: ExecutionConfig | None = None) -> None:
        self.config = config or ExecutionConfig()

    async def execute(self, code: str, validate: bool = True) -> ExecutionResult:
        """Execute code in a sandboxed subprocess.

        Args:
            code: Python code to execute
            validate: Whether to validate code before execution

        Returns:
            ExecutionResult with stdout, stderr, and parsed output
        """
        import time
        start_time = time.time()

        # Validate code
        if validate:
            is_valid, error = _validate_code(code)
            if not is_valid:
                return ExecutionResult(
                    success=False,
                    stdout="",
                    stderr=error or "Code validation failed",
                    exit_code=-1,
                    execution_time=0,
                    error=error,
                )

        # Create temp file with wrapped code
        wrapped_code = _create_wrapper_code(code)

        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False,
        ) as f:
            f.write(wrapped_code)
            temp_path = f.name

        try:
            # Run in subprocess
            result = await self._run_subprocess(temp_path)
            execution_time = time.time() - start_time

            # Parse output for JSON data
            output_data = self._parse_output(result.stdout)

            return ExecutionResult(
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
                execution_time=execution_time,
                output_data=output_data,
                error=result.stderr if result.returncode != 0 else None,
            )

        finally:
            # Cleanup temp file
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    async def _run_subprocess(self, script_path: str) -> subprocess.CompletedProcess:
        """Run the script in a subprocess with resource limits."""
        env = os.environ.copy()
        # Restrict some environment variables
        env.pop('OPENAI_API_KEY', None)
        env.pop('ZAI_API_KEY', None)
        env.pop('ANTHROPIC_API_KEY', None)

        # Build command
        cmd = [sys.executable, script_path]

        # Run with timeout
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=self.config.working_dir,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.timeout,
            )

            return subprocess.CompletedProcess(
                args=cmd,
                returncode=process.returncode or 0,
                stdout=stdout.decode('utf-8', errors='replace')[:self.config.max_output_size],
                stderr=stderr.decode('utf-8', errors='replace')[:self.config.max_output_size],
            )

        except asyncio.TimeoutError:
            process.kill()
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=-1,
                stdout="",
                stderr=f"Execution timed out after {self.config.timeout} seconds",
            )
        except Exception as e:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=-1,
                stdout="",
                stderr=str(e),
            )

    def _parse_output(self, stdout: str) -> dict[str, Any] | None:
        """Try to parse JSON from stdout."""
        import re

        # Try to find JSON object in output
        # Look for the last JSON object (the result)
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, stdout, re.DOTALL)

        if matches:
            # Try each match from last to first
            for match in reversed(matches):
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue

        # Try to parse entire output as JSON
        try:
            return json.loads(stdout.strip())
        except json.JSONDecodeError:
            pass

        return None


async def execute_backtest_code(code: str, timeout: int = 120) -> ExecutionResult:
    """Convenience function to execute backtest code.

    Args:
        code: Python backtest code to execute
        timeout: Maximum execution time in seconds

    Returns:
        ExecutionResult with backtest output
    """
    config = ExecutionConfig(timeout=timeout)
    executor = SandboxedExecutor(config)
    return await executor.execute(code)
