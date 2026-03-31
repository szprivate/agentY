"""Cross-platform shell execution tool for running skill scripts."""

import json
import subprocess
import sys

from strands import tool


@tool
def run_script(command: str, timeout: int = 120) -> str:
    """Run a shell command and return its stdout/stderr.

    Use this to execute skill scripts (e.g. Python scripts under skills/).
    Works on both Windows and Unix.

    Args:
        command: The full command to run (e.g. 'python ./skills/image-downsize/scripts/downsize.py ...').
        timeout: Maximum seconds to wait for the command to finish (default 120).
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            executable=None,
        )
        output = {
            "exit_code": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
        return json.dumps(output)
    except subprocess.TimeoutExpired:
        return json.dumps({"error": f"Command timed out after {timeout}s", "command": command})
    except Exception as e:
        return json.dumps({"error": str(e)})
