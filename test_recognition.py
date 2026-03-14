# Test the core mudra recognition functions
from mudra_recognizer import MudraRecognizer
import numpy as np

# Create test landmarks for a flat open hand (Alapadma)
test_landmarks = [
    (0.5, 0.5, 0.0),  # wrist
    (0.4, 0.3, 0.0),  # thumb CMC
    (0.35, 0.25, 0.0),  # thumb MCP  
    (0.3, 0.2, 0.0),  # thumb IP
    (0.25, 0.15, 0.0),  # thumb tip
    (0.6, 0.4, 0.0),  # index MCP
    (0.65, 0.35, 0.0),  # index PIP
    (0.7, 0.3, 0.0),  # index tip
    (0.5, 0.5, 0.0),  # middle MCP
    (0.55, 0.45, 0.0),  # middle PIP
    (0.6, 0.4, 0.0),  # middle tip
    (0.4, 0.5, 0.0),  # ring MCP
    (0.35, 0.55, 0.0),  # ring PIP
    (0.3, 0.6, 0.0),  # ring tip
    (0.3, 0.6, 0.0),  # pinky MCP
    (0.2, 0.65, 0.0),  # pinky PIP
    (0.1, 0.7, 0.0),  # pinky tip
]

recognizer = MudraRecognizer()
mudra, score = recognizer.recognize_single(test_landmarks, 'Right', debug=True)
print(f'Test result: {mudra} with score {score:.2f}')
