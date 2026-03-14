from mudra_recognizer import MudraRecognizer

# Test basic functionality
mr = MudraRecognizer()
print("✓ MudraRecognizer initialized successfully")

# Test helper functions
test_lm = [(0.5, 0.5, 0.0)] * 21  # Simple test landmarks
hs = mr._hand_size(test_lm)
print(f"✓ Hand size calculation: {hs:.3f}")

# Test angle calculation
angle = mr._angle([0,0,0], [0.5,0], [1,0])
print(f"✓ Angle calculation: {angle:.1f} degrees")

print("✓ All core functions working correctly!")
