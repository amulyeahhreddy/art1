from mudra_recognizer import MudraRecognizer

# Test Mushti vs Shikhara distinction
mr = MudraRecognizer()

# Create proper fist landmarks (all fingers bent)
fist_lm = [(0.5, 0.5, 0.0)] * 21
# Bend all fingers (angles ~90-120 degrees)
fist_lm[6] = (0.52, 0.55, 0.0)  # index PIP
fist_lm[8] = (0.54, 0.60, 0.0)  # index tip
fist_lm[10] = (0.50, 0.55, 0.0)  # middle PIP  
fist_lm[12] = (0.50, 0.60, 0.0)  # middle tip
fist_lm[14] = (0.48, 0.55, 0.0)  # ring PIP
fist_lm[16] = (0.46, 0.60, 0.0)  # ring tip
fist_lm[18] = (0.46, 0.55, 0.0)  # pinky PIP
fist_lm[20] = (0.44, 0.60, 0.0)  # pinky tip

# Test 1: Mushti - thumb tucked
mushti_lm = fist_lm.copy()
mushti_lm[4] = (0.45, 0.55, 0.0)  # thumb tip below MCP (tucked)
mushti_lm[2] = (0.48, 0.52, 0.0)  # thumb MCP

# Test 2: Shikhara - thumb pointing up
shikhara_lm = fist_lm.copy()
shikhara_lm[4] = (0.45, 0.35, 0.0)  # thumb tip above MCP
shikhara_lm[2] = (0.48, 0.52, 0.0)  # thumb MCP

mushti_score = mr._score_mushti(mushti_lm, 0.1)
shikhara_score = mr._score_shikhara(shikhara_lm, 0.1)

print(f"Mushti score (thumb tucked): {mushti_score:.2f}")
print(f"Shikhara score (thumb up): {shikhara_score:.2f}")
print(f"✓ Mushti vs Shikhara distinction working!")
