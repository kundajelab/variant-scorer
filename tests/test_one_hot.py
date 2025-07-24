import pytest
import numpy as np
import os
import sys

# Add src to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils.one_hot import dna_to_one_hot, one_hot_to_dna


class TestOneHotConversion:
    
    def test_dna_to_one_hot_simple(self):
        """Test basic DNA to one-hot conversion"""
        seqs = ["ACGT"]
        result = dna_to_one_hot(seqs)
        
        # Expected one-hot encoding for "ACGT" (alphabetical order: A=0, C=1, G=2, T=3)
        expected = np.array([[[1, 0, 0, 0],  # A
                              [0, 1, 0, 0],  # C
                              [0, 0, 1, 0],  # G
                              [0, 0, 0, 1]]], dtype=np.int8)  # T
        
        assert result.shape == (1, 4, 4)
        np.testing.assert_array_equal(result, expected)
    
    def test_dna_to_one_hot_multiple_sequences(self):
        """Test conversion of multiple sequences"""
        seqs = ["ATCG", "GCTA", "AAAA"]
        result = dna_to_one_hot(seqs)
        
        assert result.shape == (3, 4, 4)
        
        # Check first sequence "ATCG"
        expected_first = [[1, 0, 0, 0],  # A
                         [0, 0, 0, 1],  # T
                         [0, 1, 0, 0],  # C
                         [0, 0, 1, 0]]  # G
        np.testing.assert_array_equal(result[0], expected_first)

		# Check second sequence "GCTA"
        expected_second = [[0, 0, 1, 0],  # G
						  [0, 1, 0, 0],  # C
						  [0, 0, 0, 1],  # T
						  [1, 0, 0, 0]]
        np.testing.assert_array_equal(result[1], expected_second)

        # Check third sequence "AAAA"
        expected_third = [[1, 0, 0, 0],  # A
                         [1, 0, 0, 0],  # A
                         [1, 0, 0, 0],  # A
                         [1, 0, 0, 0]]  # A
        np.testing.assert_array_equal(result[2], expected_third)
    
    def test_dna_to_one_hot_lowercase(self):
        """Test that lowercase sequences are converted to uppercase"""
        seqs = ["atcg"]
        result = dna_to_one_hot(seqs)
        
        expected = np.array([[[1, 0, 0, 0],  # A
                              [0, 0, 0, 1],  # T
                              [0, 1, 0, 0],  # C
                              [0, 0, 1, 0]]], dtype=np.int8)  # G
        
        np.testing.assert_array_equal(result, expected)
    
    def test_dna_to_one_hot_invalid_bases(self):
        """Test that invalid bases get all-zero encoding"""
        seqs = ["ANCG"]  # N is not a valid base
        result = dna_to_one_hot(seqs)
        
        expected = np.array([[[1, 0, 0, 0],  # A
                              [0, 0, 0, 0],  # N -> all zeros
                              [0, 1, 0, 0],  # C
                              [0, 0, 1, 0]]], dtype=np.int8)  # G
        
        np.testing.assert_array_equal(result, expected)
    
    def test_one_hot_to_dna_simple(self):
        """Test basic one-hot to DNA conversion"""
        one_hot = np.array([[[1, 0, 0, 0],  # A
                             [0, 1, 0, 0],  # C
                             [0, 0, 1, 0],  # G
                             [0, 0, 0, 1]]], dtype=np.int8)  # T
        
        result = one_hot_to_dna(one_hot)
        expected = ["ACGT"]
        
        assert result == expected
    
    def test_one_hot_to_dna_multiple_sequences(self):
        """Test conversion of multiple one-hot sequences"""
        one_hot = np.array([[[1, 0, 0, 0],  # A
                             [0, 0, 0, 1],  # T
                             [0, 1, 0, 0],  # C
                             [0, 0, 1, 0]], # G
                            [[0, 0, 1, 0],  # G
                             [0, 1, 0, 0],  # C
                             [0, 0, 0, 1],  # T
                             [1, 0, 0, 0]]], dtype=np.int8)  # A
        
        result = one_hot_to_dna(one_hot)
        expected = ["ATCG", "GCTA"]
        
        assert result == expected
    
    def test_one_hot_to_dna_all_zeros(self):
        """Test that all-zero encodings convert to N"""
        one_hot = np.array([[[1, 0, 0, 0],  # A
                             [0, 0, 0, 0],  # all zeros -> N
                             [0, 1, 0, 0],  # C
                             [0, 0, 1, 0]]], dtype=np.int8)  # G
        
        result = one_hot_to_dna(one_hot)
        expected = ["ANCG"]
        
        assert result == expected
    
    def test_roundtrip_conversion(self):
        """Test that DNA -> one-hot -> DNA is consistent"""
        original_seqs = ["ATCG", "GCTA", "AAAA", "TTTT"]
        
        # Convert to one-hot and back
        one_hot = dna_to_one_hot(original_seqs)
        recovered_seqs = one_hot_to_dna(one_hot)
        
        assert recovered_seqs == original_seqs
    
    def test_roundtrip_with_invalid_bases(self):
        """Test roundtrip with invalid bases (should become N)"""
        original_seqs = ["ANCG", "GCTX"]
        expected_seqs = ["ANCG", "GCTN"]  # X becomes N
        
        # Convert to one-hot and back
        one_hot = dna_to_one_hot(original_seqs)
        recovered_seqs = one_hot_to_dna(one_hot)
        
        assert recovered_seqs == expected_seqs
    
    def test_equal_length_requirement(self):
        """Test that sequences must be equal length"""
        seqs = ["ATCG", "GC"]  # Different lengths
        
        with pytest.raises(AssertionError):
            dna_to_one_hot(seqs)
    
    def test_empty_sequence(self):
        """Test handling of empty sequences"""
        seqs = [""]
        result = dna_to_one_hot(seqs)
        
        assert result.shape == (1, 0, 4)
        
        # Test roundtrip
        recovered = one_hot_to_dna(result)
        assert recovered == [""]
    
    def test_data_types(self):
        """Test that output data types are correct"""
        seqs = ["ATCG"]
        one_hot = dna_to_one_hot(seqs)
        
        # Should be int8
        assert one_hot.dtype == np.int8
        
        # Should contain only 0s and 1s
        assert np.all(np.isin(one_hot, [0, 1]))
        
        # Test one_hot_to_dna returns strings
        dna_seqs = one_hot_to_dna(one_hot)
        assert isinstance(dna_seqs, list)
        assert all(isinstance(seq, str) for seq in dna_seqs)
