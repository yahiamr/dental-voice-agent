"""Tests that guards and booking stay free of frameworks. Stage: 1.

What this file is:
    Proves the import rule holds.

What it contains:
    - Importing dental_agent.guards and dental_agent.booking loads neither LangChain nor
      LangGraph.

Used by:
    pytest
"""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("package", ["dental_agent.guards", "dental_agent.booking"])
def test_package_loads_no_framework(package):
    # A new process is needed, because this test process has already imported the frameworks.
    code = (
        "import importlib, sys\n"
        f"importlib.import_module('{package}')\n"
        "importlib.import_module('dental_agent.guards.yes_rule')\n"
        "importlib.import_module('dental_agent.guards.urgent')\n"
        "importlib.import_module('dental_agent.guards.validate')\n"
        "importlib.import_module('dental_agent.guards.reply_check')\n"
        "importlib.import_module('dental_agent.booking.calendar')\n"
        "importlib.import_module('dental_agent.booking.seed')\n"
        "bad = [name for name in sys.modules if name.startswith(('langchain', 'langgraph'))]\n"
        "print(bad)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
