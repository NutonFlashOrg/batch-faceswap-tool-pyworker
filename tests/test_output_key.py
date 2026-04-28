"""Unit tests for _output_key_for in workflow_transform.py.

The comfyui-json directory name uses a hyphen and cannot be imported as a
normal Python package, so we load the module directly by file path.
"""
import importlib.util
import os
import sys

import pytest


def _load_workflow_transform():
    """Load workers/comfyui-json/workflow_transform.py by file path."""
    here = os.path.dirname(__file__)
    module_path = os.path.join(here, "..", "workers", "comfyui-json", "workflow_transform.py")
    spec = importlib.util.spec_from_file_location("workflow_transform", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_wt = _load_workflow_transform()
_output_key_for = _wt._output_key_for


def test_output_key_derived_from_source_key():
    src = "jobs/abc123/videos/v0/source.mp4"
    assert _output_key_for(src) == "jobs/abc123/videos/v0/output.mp4"


def test_output_key_rejects_unexpected_shape():
    with pytest.raises(ValueError):
        _output_key_for("legacy/users/u/results/r/source.mp4_x")
