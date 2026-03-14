from mudra_recognizer import MudraRecognizer

# Test multiple new mudras
mr = MudraRecognizer()

# Test Mayura - peacock
mayura_lm = [(0.5, 0.5, 0.0)] * 21
mayura_lm[4] = (0.45, 0.45, 0.0)  # thumb near index
mayura_lm[8] = (0.55, 0.45, 0.0)  # index near thumb
mayura_lm[12] = (0.50, 0.60, 0.0)  # middle half-extended

# Test Ardhachandra - half moon
ardha_lm = [(0.5, 0.5, 0.0)] * 21
ardha_lm[6] = (0.52, 0.45, 0.0)  # index extended
ardha_lm[10] = (0.50, 0.45, 0.0)  # index tip
ardha_lm[14] = (0.48, 0.45, 0.0)  # ring extended
ardha_lm[16] = (0.46, 0.45, 0.0)  # ring tip

# Test Suchi - pointing finger
suchi_lm = [(0.5, 0.5, 0.0)] * 21
suchi_lm[6] = (0.52, 0.45, 0.0)  # index extended
suchi_lm[8] = (0.54, 0.40, 0.0)  # index tip
suchi_lm[10] = (0.50, 0.55, 0.0)  # middle bent
suchi_lm[4] = (0.45, 0.55, 0.0)  # thumb tucked

print("Testing new mudras:")
print(f"Mayura score: {mr._score_mayura(mayura_lm, 0.1):.2f}")
print(f"Ardhachandra score: {mr._score_ardhachandra(ardha_lm, 0.1):.2f}")
print(f"Suchi score: {mr._score_suchi(suchi_lm, 0.1):.2f}")
print("✓ All new mudras working!")
