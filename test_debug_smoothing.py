#!/usr/bin/env python3
"""Test debug panel smoothing functionality"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mudra_recognizer import MudraRecognizer

def test_debug_smoothing():
    """Test that debug scores are smoothed properly"""
    recognizer = MudraRecognizer()
    
    print("Testing debug panel smoothing...")
    
    # Test that display buffers exist
    assert hasattr(recognizer, 'display_scores_right'), "Missing display_scores_right buffer"
    assert hasattr(recognizer, 'display_scores_left'), "Missing display_scores_left buffer"
    assert hasattr(recognizer, 'DISPLAY_ALPHA'), "Missing DISPLAY_ALPHA constant"
    assert hasattr(recognizer, 'samyuktha_enabled'), "Missing samyuktha_enabled flag"
    
    # Test samyuktha is disabled by default
    assert not recognizer.samyuktha_enabled, "Samyuktha should be disabled by default"
    
    # Test display alpha value
    assert recognizer.DISPLAY_ALPHA == 0.15, f"Expected DISPLAY_ALPHA=0.15, got {recognizer.DISPLAY_ALPHA}"
    
    # Test buffers start empty
    assert len(recognizer.display_scores_right) == 0, "Right display buffer should start empty"
    assert len(recognizer.display_scores_left) == 0, "Left display buffer should start empty"
    
    print("✓ Debug smoothing system initialized correctly!")
    print(f"✓ Samyuktha Hastas: {'ON' if recognizer.samyuktha_enabled else 'OFF'}")
    print(f"✓ Display Alpha: {recognizer.DISPLAY_ALPHA}")

if __name__ == "__main__":
    test_debug_smoothing()
