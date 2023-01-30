"""
Unit and regression test for the mfGPR package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import mfGPR


def test_mfGPR_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "mfGPR" in sys.modules
