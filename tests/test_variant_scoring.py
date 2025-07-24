import pytest
import subprocess
import tempfile
import os
import sys
import pandas as pd
from pathlib import Path

class TestVariantScoringCLI:

    @pytest.fixture(scope="class")
    def script_path(self, src):
        """Fixture to provide the path to the variant_scoring.py script"""
        return os.path.join(src, "variant_scoring.py")

    def test_variant_scoring_help(self, script_path):
        """Test that variant_scoring.py shows help without errors"""
        if not os.path.exists(script_path):
            pytest.skip(f"Script {script_path} not found")
        
        cmd = [sys.executable, script_path, '--help']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Should exit successfully and show help
        assert result.returncode == 0
        assert 'usage:' in result.stdout.lower() or 'help' in result.stdout.lower()
        # Check that required arguments are mentioned
        assert '--list' in result.stdout
        assert '--genome' in result.stdout
        assert '--model' in result.stdout
        assert '--out_prefix' in result.stdout
        assert '--chrom_sizes' in result.stdout
    
    def test_variant_scoring_missing_required_args(self, script_path):
        """Test that variant_scoring.py fails gracefully with missing required arguments"""
        if not os.path.exists(script_path):
            pytest.skip(f"Script {script_path} not found")
        
        cmd = [sys.executable, script_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Should fail with non-zero exit code
        assert result.returncode != 0

        # Should mention missing required arguments
        error_text = result.stderr.lower()
        assert 'required' in error_text or 'argument' in error_text or 'missing' in error_text