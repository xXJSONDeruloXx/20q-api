"""
I/O utilities for weight matrices with support for JSON and binary formats.
"""

import numpy as np
import json
import struct
from pathlib import Path
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)


class WeightIO:
    """Utilities for loading and saving weight matrices."""
    
    @staticmethod
    def save_weights(weights: np.ndarray, filename: Union[str, Path]) -> None:
        """
        Save weights to file (JSON or binary format based on extension).
        
        Args:
            weights: Weight matrix to save
            filename: Output filename (.json or .bin)
        """
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if filepath.suffix.lower() == '.json':
            WeightIO._save_json(weights, filepath)
        elif filepath.suffix.lower() == '.bin':
            WeightIO._save_binary(weights, filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    @staticmethod
    def load_weights(filename: Union[str, Path]) -> np.ndarray:
        """
        Load weights from file.
        
        Args:
            filename: Input filename
            
        Returns:
            Loaded weight matrix
        """
        filepath = Path(filename)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Weight file not found: {filepath}")
            
        if filepath.suffix.lower() == '.json':
            return WeightIO._load_json(filepath)
        elif filepath.suffix.lower() == '.bin':
            return WeightIO._load_binary(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    @staticmethod
    def _save_json(weights: np.ndarray, filepath: Path) -> None:
        """Save weights as JSON (float32 format)."""
        data = {
            'weights': weights.astype(np.float32).tolist(),
            'shape': weights.shape,
            'dtype': 'float32'
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved weights to {filepath} (JSON format)")
    
    @staticmethod
    def _load_json(filepath: Path) -> np.ndarray:
        """Load weights from JSON format."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            # Legacy format - just the weights array
            return np.array(data, dtype=np.float32)
        else:
            # New format with metadata
            weights = np.array(data['weights'], dtype=np.float32)
            expected_shape = tuple(data.get('shape', weights.shape))
            
            if weights.shape != expected_shape:
                weights = weights.reshape(expected_shape)
            
            return weights
    
    @staticmethod
    def _save_binary(weights: np.ndarray, filepath: Path) -> None:
        """
        Save weights as binary format (signed 8-bit quantized).
        
        Format:
        - 4 bytes: n_objects (uint32)
        - 4 bytes: n_questions (uint32)
        - 4 bytes: scale_factor (float32)
        - n_objects * n_questions bytes: quantized weights (int8)
        """
        n_objects, n_questions = weights.shape
        
        # Quantize to 8-bit signed integers
        quantized, scale_factor = WeightIO._quantize_weights(weights)
        
        with open(filepath, 'wb') as f:
            # Write header
            f.write(struct.pack('<I', n_objects))  # Little-endian uint32
            f.write(struct.pack('<I', n_questions))
            f.write(struct.pack('<f', scale_factor))  # Little-endian float32
            
            # Write quantized weights
            f.write(quantized.tobytes())
        
        logger.info(f"Saved weights to {filepath} (binary format, scale={scale_factor:.6f})")
    
    @staticmethod
    def _load_binary(filepath: Path) -> np.ndarray:
        """Load weights from binary format."""
        with open(filepath, 'rb') as f:
            # Read header
            n_objects = struct.unpack('<I', f.read(4))[0]
            n_questions = struct.unpack('<I', f.read(4))[0]
            scale_factor = struct.unpack('<f', f.read(4))[0]
            
            # Read quantized weights
            quantized_data = f.read(n_objects * n_questions)
            quantized = np.frombuffer(quantized_data, dtype=np.int8)
            quantized = quantized.reshape((n_objects, n_questions))
            
            # Dequantize
            weights = WeightIO._dequantize_weights(quantized, scale_factor)
            
        logger.info(f"Loaded weights from {filepath} (binary format)")
        return weights
    
    @staticmethod
    def _quantize_weights(weights: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Quantize float32 weights to signed 8-bit integers.
        
        Args:
            weights: Float32 weight matrix
            
        Returns:
            Tuple of (quantized weights, scale factor)
        """
        # Find the maximum absolute value
        max_abs = np.max(np.abs(weights))
        
        if max_abs == 0:
            return np.zeros_like(weights, dtype=np.int8), 1.0
        
        # Calculate scale factor to map to [-127, 127] range
        scale_factor = max_abs / 127.0
        
        # Quantize and clip to valid range
        quantized = np.round(weights / scale_factor).astype(np.int8)
        quantized = np.clip(quantized, -127, 127)
        
        return quantized, scale_factor
    
    @staticmethod
    def _dequantize_weights(quantized: np.ndarray, scale_factor: float) -> np.ndarray:
        """
        Dequantize signed 8-bit integers back to float32.
        
        Args:
            quantized: Quantized weight matrix (int8)
            scale_factor: Scale factor used during quantization
            
        Returns:
            Dequantized float32 weight matrix
        """
        return quantized.astype(np.float32) * scale_factor
    
    @staticmethod
    def convert_format(
        input_file: Union[str, Path], 
        output_file: Union[str, Path]
    ) -> None:
        """
        Convert weights between JSON and binary formats.
        
        Args:
            input_file: Input filename
            output_file: Output filename
        """
        weights = WeightIO.load_weights(input_file)
        WeightIO.save_weights(weights, output_file)
        
        logger.info(f"Converted {input_file} to {output_file}")
    
    @staticmethod
    def get_compression_ratio(json_file: Union[str, Path], bin_file: Union[str, Path]) -> float:
        """
        Calculate compression ratio between JSON and binary formats.
        
        Args:
            json_file: Path to JSON file
            bin_file: Path to binary file
            
        Returns:
            Compression ratio (json_size / bin_size)
        """
        json_size = Path(json_file).stat().st_size
        bin_size = Path(bin_file).stat().st_size
        
        return json_size / bin_size if bin_size > 0 else 0.0
