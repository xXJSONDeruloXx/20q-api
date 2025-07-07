"""
Tests for weight quantization and I/O operations.
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path

from twentyq_ann.io import WeightIO


class TestWeightIO:
    """Test cases for WeightIO class."""
    
    def test_json_save_load(self):
        """Test saving and loading weights in JSON format."""
        # Create test weights
        weights = np.random.uniform(-1, 1, (5, 3))
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save weights
            WeightIO.save_weights(weights, temp_path)
            
            # Load weights
            loaded_weights = WeightIO.load_weights(temp_path)
            
            # Check they match
            np.testing.assert_array_almost_equal(weights, loaded_weights)
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_binary_save_load(self):
        """Test saving and loading weights in binary format."""
        # Create test weights
        weights = np.random.uniform(-1, 1, (5, 3))
        
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save weights
            WeightIO.save_weights(weights, temp_path)
            
            # Load weights
            loaded_weights = WeightIO.load_weights(temp_path)
            
            # Check they're approximately equal (due to quantization)
            np.testing.assert_array_almost_equal(weights, loaded_weights, decimal=2)
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_quantization_symmetry(self):
        """Test that quantization and dequantization are symmetric."""
        # Create test weights
        weights = np.random.uniform(-1, 1, (10, 8))
        
        # Quantize and dequantize
        quantized, scale_factor = WeightIO._quantize_weights(weights)
        dequantized = WeightIO._dequantize_weights(quantized, scale_factor)
        
        # Check they're approximately equal
        np.testing.assert_array_almost_equal(weights, dequantized, decimal=2)
    
    def test_quantization_range(self):
        """Test that quantized weights are in valid range."""
        # Create test weights with extreme values
        weights = np.array([[-100, 0, 100], [0.1, -0.1, 0]])
        
        # Quantize
        quantized, scale_factor = WeightIO._quantize_weights(weights)
        
        # Check range
        assert np.all(quantized >= -127)
        assert np.all(quantized <= 127)
        assert quantized.dtype == np.int8
        assert scale_factor > 0
    
    def test_zero_weights_quantization(self):
        """Test quantization of zero weights."""
        weights = np.zeros((3, 3))
        
        quantized, scale_factor = WeightIO._quantize_weights(weights)
        
        assert np.all(quantized == 0)
        assert scale_factor == 1.0
    
    def test_format_conversion(self):
        """Test conversion between formats."""
        # Create test weights
        weights = np.random.uniform(-1, 1, (4, 4))
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f1:
            json_path = f1.name
        
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f2:
            bin_path = f2.name
        
        try:
            # Save as JSON
            WeightIO.save_weights(weights, json_path)
            
            # Convert to binary
            WeightIO.convert_format(json_path, bin_path)
            
            # Load binary
            loaded_weights = WeightIO.load_weights(bin_path)
            
            # Check they're approximately equal
            np.testing.assert_array_almost_equal(weights, loaded_weights, decimal=2)
            
        finally:
            Path(json_path).unlink(missing_ok=True)
            Path(bin_path).unlink(missing_ok=True)
    
    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        # Create test weights
        weights = np.random.uniform(-1, 1, (10, 10))
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f1:
            json_path = f1.name
        
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f2:
            bin_path = f2.name
        
        try:
            # Save in both formats
            WeightIO.save_weights(weights, json_path)
            WeightIO.save_weights(weights, bin_path)
            
            # Calculate compression ratio
            ratio = WeightIO.get_compression_ratio(json_path, bin_path)
            
            # Binary should be smaller
            assert ratio > 1.0
            
        finally:
            Path(json_path).unlink(missing_ok=True)
            Path(bin_path).unlink(missing_ok=True)
    
    def test_legacy_json_format(self):
        """Test loading legacy JSON format (just array)."""
        weights = np.random.uniform(-1, 1, (3, 3))
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save in legacy format (just the array)
            with open(temp_path, 'w') as file:
                json.dump(weights.tolist(), file)
            
            # Load using WeightIO
            loaded_weights = WeightIO.load_weights(temp_path)
            
            # Check they match
            np.testing.assert_array_almost_equal(weights, loaded_weights)
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_unsupported_format(self):
        """Test handling of unsupported file formats."""
        weights = np.random.uniform(-1, 1, (2, 2))
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            WeightIO.save_weights(weights, "test.txt")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            WeightIO.load_weights("test.txt")
    
    def test_file_not_found(self):
        """Test handling of missing files."""
        with pytest.raises(FileNotFoundError):
            WeightIO.load_weights("nonexistent.json")
    
    def test_binary_header_format(self):
        """Test binary file header format."""
        weights = np.random.uniform(-1, 1, (7, 5))
        
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save weights
            WeightIO.save_weights(weights, temp_path)
            
            # Read header manually
            with open(temp_path, 'rb') as file:
                n_objects = int.from_bytes(file.read(4), 'little')
                n_questions = int.from_bytes(file.read(4), 'little')
                
            assert n_objects == 7
            assert n_questions == 5
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
