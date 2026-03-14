#!/usr/bin/env python3
"""Test two-hand mudra recognition"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mudra_recognizer import MudraRecognizer
import numpy as np

def create_mock_landmarks(hand_type='open'):
    """Create mock hand landmarks for testing"""
    if hand_type == 'open':
        # Open hand (Pataka-like)
        lm = [(0.5, 0.5, 0.0)] * 21  # Simple mock
        lm[5] = (0.4, 0.3, 0.0)  # Index MCP
        lm[8] = (0.4, 0.2, 0.0)  # Index tip
        lm[9] = (0.5, 0.3, 0.0)  # Middle MCP
        lm[12] = (0.5, 0.2, 0.0)  # Middle tip
        lm[13] = (0.6, 0.3, 0.0)  # Ring MCP
        lm[16] = (0.6, 0.2, 0.0)  # Ring tip
        lm[17] = (0.7, 0.3, 0.0)  # Pinky MCP
        lm[20] = (0.7, 0.2, 0.0)  # Pinky tip
    elif hand_type == 'fist':
        # Closed fist (Mushti-like)
        lm = [(0.5, 0.5, 0.0)] * 21
        lm[5] = (0.4, 0.4, 0.0)  # Index MCP
        lm[8] = (0.4, 0.45, 0.0)  # Index tip (bent)
        lm[9] = (0.5, 0.4, 0.0)  # Middle MCP
        lm[12] = (0.5, 0.45, 0.0)  # Middle tip (bent)
        lm[13] = (0.6, 0.4, 0.0)  # Ring MCP
        lm[16] = (0.6, 0.45, 0.0)  # Ring tip (bent)
        lm[17] = (0.7, 0.4, 0.0)  # Pinky MCP
        lm[20] = (0.7, 0.45, 0.0)  # Pinky tip (bent)
    else:
        lm = [(0.5, 0.5, 0.0)] * 21
    
    return lm

def test_two_hand_mudras():
    """Test two-hand mudra recognition"""
    recognizer = MudraRecognizer()
    
    print("Testing two-hand mudras...")
    
    # Test Anjali (both open hands together)
    hand1 = create_mock_landmarks('open')
    hand2 = create_mock_landmarks('open')
    
    # Position hands close together for Anjali
    for i in range(21):
        hand2[i] = (hand2[i][0] + 0.1, hand2[i][1], hand2[i][2])
    
    result = recognizer.recognize_two_hand((hand1, 'Right'), (hand2, 'Left'))
    print(f"Anjali test result: {result}")
    
    # Test Dola (both fists)
    hand1 = create_mock_landmarks('fist')
    hand2 = create_mock_landmarks('fist')
    
    result = recognizer.recognize_two_hand((hand1, 'Right'), (hand2, 'Left'))
    print(f"Dola test result: {result}")
    
    print("✓ Two-hand mudra system working!")

if __name__ == "__main__":
    test_two_hand_mudras()
