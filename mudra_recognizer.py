import math
import numpy as np
from collections import deque, Counter


class MudraRecognizer:
    def __init__(self):
        # Temporal smoothing buffers for each hand (7-frame history)
        self.left_hand_history = deque(maxlen=7)
        self.right_hand_history = deque(maxlen=7)
        self.debug_mode = False
        
        # Hysteresis state for preventing flickering
        self.left_confirmed = None
        self.right_confirmed = None
        
        # Store last features for debugging
        self.last_features = {}
        self.last_confidence = 0.0
        
        # Initialize mudra groups mapping
        self._init_mudra_groups()
    
    # ==================== HELPER FUNCTIONS ====================
    
    def _dist(self, p1, p2):
        """Euclidean distance using x,y only (ignore z unless specified)"""
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    
    def _hand_size(self, lm):
        """Normalize factor = wrist(0) to middle MCP(9) distance"""
        # If result < 0.01, return 0.01 to avoid division by zero
        return max(self._dist(lm[0], lm[9]), 0.01)
    
    def _angle(self, a, b, c):
        """Angle in degrees at point b, formed by vectors (b->a) and (b->c)"""
        # Vector from b to a
        v1 = [a[0] - b[0], a[1] - b[1]]
        # Vector from b to c
        v2 = [c[0] - b[0], c[1] - b[1]]
        
        # Calculate magnitudes
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag1 == 0 or mag2 == 0:
            return 180.0  # Default to extended
        
        # Calculate dot product
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        
        # Calculate angle in degrees
        cos_angle = dot_product / (mag1 * mag2)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp to [-1, 1]
        angle = math.degrees(math.acos(cos_angle))
        
        return angle
    
    def _finger_angle(self, lm, mcp, pip, tip):
        """Returns angle at pip joint"""
        return self._angle(lm[mcp], lm[pip], lm[tip])
    
    def _is_extended(self, lm, mcp, pip, tip, threshold=150):
        """Check if finger is extended above threshold"""
        return self._finger_angle(lm, mcp, pip, tip) > threshold
    
    def _is_bent(self, lm, mcp, pip, tip, threshold=120):
        """Check if finger is bent below threshold"""
        return self._finger_angle(lm, mcp, pip, tip) < threshold
    
    def _is_half_bent(self, lm, mcp, pip, tip):
        """Check if finger is half-bent (120-150 degrees)"""
        angle = self._finger_angle(lm, mcp, pip, tip)
        return 120 <= angle <= 150
    
    def _thumb_tucked(self, lm, hs):
        """Thumb tip(4) is close to index MCP(5) — thumb folded INTO palm"""
        return self._dist(lm[4], lm[5]) / hs < 0.35
    
    def _thumb_extended(self, lm, hs):
        """Thumb tip(4) is far from index MCP(5)"""
        return self._dist(lm[4], lm[5]) / hs > 0.45
    
    def _tips_touching(self, lm, tip1, tip2, hs, threshold=0.12):
        """Check if two fingertips are touching"""
        return self._dist(lm[tip1], lm[tip2]) / hs < threshold
    
    def _spread(self, lm, mcp1, mcp2, hs):
        """Normalized spread between two finger MCPs"""
        return self._dist(lm[mcp1], lm[mcp2]) / hs
    
    def _all_four_extended(self, lm, threshold=148):
        """Check if all four fingers are extended"""
        return all([
            self._is_extended(lm, 5, 6, 8, threshold),   # Index
            self._is_extended(lm, 9, 10, 12, threshold), # Middle
            self._is_extended(lm, 13, 14, 16, threshold), # Ring
            self._is_extended(lm, 17, 18, 20, threshold)  # Pinky
        ])
    
    def _all_four_bent(self, lm, threshold=120):
        """Check if all four fingers are bent"""
        return all([
            self._is_bent(lm, 5, 6, 8, threshold),   # Index
            self._is_bent(lm, 9, 10, 12, threshold), # Middle
            self._is_bent(lm, 13, 14, 16, threshold), # Ring
            self._is_bent(lm, 17, 18, 20, threshold)  # Pinky
        ])
    
    def _normalize_hand(self, lm, handedness):
        """Mirror left hand: replace each x with (1.0 - x)"""
        if handedness == 'Left':
            lm = [(1.0 - p[0], p[1], p[2]) for p in lm]
        return lm
    
    def _get_finger_angles(self, lm):
        """Return dict: {thumb, index, middle, ring, pinky} angles"""
        return {
            'thumb': self._finger_angle(lm, 1, 2, 4),
            'index': self._finger_angle(lm, 5, 6, 8),
            'middle': self._finger_angle(lm, 9, 10, 12),
            'ring': self._finger_angle(lm, 13, 14, 16),
            'pinky': self._finger_angle(lm, 17, 18, 20)
        }
    
    # ==================== STAGE 1: PRE-FILTER GROUPS ====================
    
    def _get_stage1_groups(self, lm, hs):
        """Compute boolean flags first (cheap operations)"""
        four_ext = self._all_four_extended(lm, 148)
        four_bent = self._all_four_bent(lm, 120)
        thumb_ext = self._thumb_extended(lm, hs)
        thumb_tuck = self._thumb_tucked(lm, hs)
        pinch_14 = self._tips_touching(lm, 4, 8, hs, 0.15)  # thumb-index
        idx_ext = self._is_extended(lm, 5, 6, 8)
        mid_ext = self._is_extended(lm, 9, 10, 12)
        ring_ext = self._is_extended(lm, 13, 14, 16)
        pink_ext = self._is_extended(lm, 17, 18, 20)
        ext_count = sum([idx_ext, mid_ext, ring_ext, pink_ext])
        
        groups = []
        
        if four_ext and thumb_ext:
            groups.append('all_open')
        if four_ext and thumb_tuck:
            groups.append('four_open')
        if four_bent:
            groups.append('fist')
        if pinch_14:
            groups.append('pinch')
        if ext_count == 1:
            groups.append('one_up')
        if ext_count == 2:
            groups.append('two_up')
        if ext_count == 3:
            groups.append('three_up')
        if not groups:
            groups.append('mixed')
            
        return groups, {
            'four_ext': four_ext, 'four_bent': four_bent,
            'thumb_ext': thumb_ext, 'thumb_tuck': thumb_tuck,
            'pinch_14': pinch_14, 'ext_count': ext_count
        }
    
    # ==================== STAGE 2: MUDRA SCORING FUNCTIONS ====================
    
    def _score_pataka(self, lm, hs):
        """
        Pataka: ALL 4 fingers fully extended, held TOGETHER (low spread),
        thumb TUCKED across palm (NOT open/extended)
        THE OLD WRONG DEFINITION "all 5 fingers open" IS ABOLISHED.
        """
        score = 0.0
        if self._all_four_extended(lm, 152):
            score += 0.35
        if self._thumb_tucked(lm, hs):
            score += 0.35
        if self._spread(lm, 5, 17, hs) < 0.55:  # fingers together
            score += 0.15
        if self._spread(lm, 5, 17, hs) < 0.42:  # bonus: very tight
            score += 0.15
        return min(score, 1.0)
    
    def _score_alapadma(self, lm, hs):
        """
        Alapadma: ALL 5 fingers fully extended AND maximally spread
        apart like a blooming lotus / open fan
        """
        score = 0.0
        if self._all_four_extended(lm, 148) and self._is_extended(lm, 1, 2, 4, 140):
            score += 0.40
        
        # Calculate average spread of adjacent MCP pairs
        spreads = [
            self._spread(lm, 1, 5, hs),   # thumb-index
            self._spread(lm, 5, 9, hs),   # index-middle
            self._spread(lm, 9, 13, hs),  # middle-ring
            self._spread(lm, 13, 17, hs)  # ring-pinky
        ]
        avg_spread = sum(spreads) / len(spreads)
        
        if avg_spread > 0.18:
            score += 0.25
        if avg_spread > 0.22:  # bonus
            score += 0.15
        if self._spread(lm, 5, 17, hs) > 0.60:  # total spread
            score += 0.20
        return min(score, 1.0)
    
    def _score_mushti(self, lm, hs):
        """Full tight fist. ALL 4 fingers bent tight AND thumb 
        bent/tucked OVER fingers. No part of thumb extended upward."""
        s = 0.0
        # Four fingers must be tightly bent
        if self._all_four_bent(lm, 105):          s += 0.50
        # Thumb must be tucked/bent over fingers (not pointing up)
        if self._thumb_tucked(lm, hs):        s += 0.35
        # HARD PENALTY: if thumb tip is above thumb MCP, it is 
        # pointing up = Shikhara not Mushti. Subtract heavily.
        if lm[4][1] < lm[2][1]:              s -= 0.40
        # Extra confirmation: thumb tip should be near index MCP area
        if self._dist(lm[4], lm[5]) / hs < 0.30:  s += 0.15
        return max(min(s, 1.0), 0.0)
    
    def _score_shikhara(self, lm, hs):
        """Fist with thumb pointing straight UP away from fingers.
        Thumb tip must be clearly ABOVE thumb MCP in image coords."""
        s = 0.0
        # Four fingers must be in fist
        if self._all_four_bent(lm, 110):          s += 0.45
        # Thumb must be extended outward (not tucked)
        if self._thumb_extended(lm, hs):           s += 0.25
        # CRITICAL: thumb tip must be ABOVE thumb MCP 
        # In image coords: lower Y value = higher on screen
        if lm[4][1] < lm[2][1]:              s += 0.30
        # Bonus: thumb tip well above wrist level
        if lm[4][1] < lm[0][1]:              s += 0.10
        # HARD PENALTY: if thumb is tucked, it cannot be Shikhara
        if self._thumb_tucked(lm, hs):        s -= 0.50
        return max(min(s, 1.0), 0.0)
    
    def _score_mayura(self, lm, hs):
        """Peacock: Thumb+Index tips touching, middle half-extended,
        ring+pinky bent back."""
        s = 0.0
        if self._tips_touching(lm, 4, 8, hs, 0.13):     s += 0.45
        ma = self._finger_angle(lm, 9, 10, 12)
        if 125 <= ma <= 158:                         s += 0.25
        if self._is_bent(lm, 13, 14, 16):              s += 0.15
        if self._is_bent(lm, 17, 18, 20):              s += 0.15
        return min(s, 1.0)
    
    def _score_ardhachandra(self, lm, hs):
        """Half Moon: All 5 extended, MODERATE spread.
        More spread than Pataka, less than Alapadma."""
        s = 0.0
        if self._all_four_extended(lm, 145):                  s += 0.30
        if self._is_extended(lm, 1, 2, 4, 135):         s += 0.15
        sp = self._spread(lm, 5, 17, hs)
        if 0.40 <= sp <= 0.65:                        s += 0.35
        import numpy as np
        avg = np.mean([
            self._spread(lm, 5, 9, hs),
            self._spread(lm, 9, 13, hs),
            self._spread(lm, 13, 17, hs),
        ])
        if 0.13 <= avg <= 0.22:                       s += 0.20
        return min(s, 1.0)
    
    def _score_arala(self, lm, hs):
        """Bent Index: Index half-bent/curved (120-155°), 
        middle+ring+pinky straight, thumb extended out."""
        s = 0.0
        ia = self._finger_angle(lm, 5, 6, 8)
        if 120 <= ia <= 155:                          s += 0.35
        if self._is_extended(lm, 9, 10, 12, 155):       s += 0.20
        if self._is_extended(lm, 13, 14, 16, 155):      s += 0.20
        if self._is_extended(lm, 17, 18, 20, 155):      s += 0.10
        if self._thumb_extended(lm, hs):                   s += 0.15
        return min(s, 1.0)
    
    def _score_shukatunda(self, lm, hs):
        """Parrot Beak: Index extended, middle half-bent, 
        ring+pinky tightly bent, thumb tip near middle DIP."""
        s = 0.0
        if self._is_extended(lm, 5, 6, 8, 160):        s += 0.30
        ma = self._finger_angle(lm, 9, 10, 12)
        if 125 <= ma <= 162:                          s += 0.20
        if self._is_bent(lm, 13, 14, 16, 115):          s += 0.20
        if self._is_bent(lm, 17, 18, 20, 115):          s += 0.15
        if self._dist(lm[4], lm[11]) / hs < 0.25:   s += 0.15
        return min(s, 1.0)
    
    def _score_kapittha(self, lm, hs):
        """Apple Hold: Index+Middle curled 85-135°, 
        thumb near fingertips, ring+pinky bent."""
        s = 0.0
        ia = self._finger_angle(lm, 5, 6, 8)
        ma = self._finger_angle(lm, 9, 10, 12)
        if 85 <= ia <= 135:                           s += 0.25
        if 85 <= ma <= 135:                           s += 0.25
        if self._dist(lm[4], lm[8]) / hs < 0.20:    s += 0.25
        if self._is_bent(lm, 13, 14, 16):               s += 0.15
        if self._is_bent(lm, 17, 18, 20):               s += 0.10
        return min(s, 1.0)
    
    def _score_katakamukha(self, lm, hs):
        """Bracelet: Thumb+Index+Middle tips triangle, ring+pinky extended."""
        s = 0.0
        if self._dist(lm[4], lm[8]) / hs  < 0.15:  s += 0.20
        if self._dist(lm[8], lm[12]) / hs < 0.15:  s += 0.15
        if self._dist(lm[4], lm[12]) / hs < 0.15:  s += 0.15
        if self._is_extended(lm, 13, 14, 16, 145):      s += 0.25
        if self._is_extended(lm, 17, 18, 20, 145):      s += 0.25
        return min(s, 1.0)
    
    def _score_suchi(self, lm, hs):
        """Needle: Index pointing, ALL others tightly bent, thumb tucked."""
        s = 0.0
        if self._is_extended(lm, 5, 6, 8, 162):        s += 0.35
        if self._is_bent(lm, 9, 10, 12, 110):          s += 0.20
        if self._is_bent(lm, 13, 14, 16, 110):         s += 0.20
        if self._is_bent(lm, 17, 18, 20, 110):         s += 0.15
        if self._thumb_tucked(lm, hs):               s += 0.10
        return min(s, 1.0)
    
    def _score_chandrakala(self, lm, hs):
        """Crescent: Thumb+Index C-shape (tips 0.13-0.30 apart), 
        all other fingers fully bent."""
        s = 0.0
        d = self._dist(lm[4], lm[8]) / hs
        if 0.13 <= d <= 0.30:                        s += 0.50
        if self._is_bent(lm, 9, 10, 12, 115):          s += 0.20
        if self._is_bent(lm, 13, 14, 16, 115):         s += 0.20
        if self._is_bent(lm, 17, 18, 20, 115):         s += 0.10
        return min(s, 1.0)
    
    def _score_padmakosha(self, lm, hs):
        """Lotus Cup: All 5 fingers half-bent (110-152°) like holding a ball."""
        angles = [
            self._finger_angle(lm, 5, 6, 8),
            self._finger_angle(lm, 9, 10, 12),
            self._finger_angle(lm, 13, 14, 16),
            self._finger_angle(lm, 17, 18, 20),
        ]
        count = sum(1 for a in angles if 110 <= a <= 152)
        ta = self._finger_angle(lm, 1, 2, 4)
        thumb_half = 1 if 100 <= ta <= 155 else 0
        return min((count + thumb_half) * 0.18, 1.0)
    
    def _score_sarpashirsha(self, lm, hs):
        """Snake Head: 4 fingers extended but tips DROOPING downward.
        Tip Y must be greater than DIP Y (drooping in image coords)."""
        s = 0.0
        if self._all_four_extended(lm, 145):                  s += 0.40
        droop = sum([
            lm[8][1]  > lm[7][1],
            lm[12][1] > lm[11][1],
            lm[16][1] > lm[15][1],
            lm[20][1] > lm[19][1],
        ])
        s += 0.15 * droop
        if self._thumb_tucked(lm, hs):               s += 0.10
        if droop < 2:                                 s *= 0.4
        return min(s, 1.0)
    
    def _score_mrigashirsha(self, lm, hs):
        """Deer Head: Index+Pinky extended (horns), Middle+Ring bent."""
        s = 0.0
        if self._is_extended(lm, 5, 6, 8, 148):        s += 0.30
        if self._is_extended(lm, 17, 18, 20, 148):     s += 0.30
        if self._is_bent(lm, 9, 10, 12):               s += 0.20
        if self._is_bent(lm, 13, 14, 16):              s += 0.20
        return min(s, 1.0)
    
    def _score_simhamukha(self, lm, hs):
        """Lion Face: Thumb+Index+Pinky extended, Middle+Ring bent."""
        s = 0.0
        if self._thumb_extended(lm, hs):                  s += 0.20
        if self._is_extended(lm, 5, 6, 8, 148):        s += 0.25
        if self._is_extended(lm, 17, 18, 20, 148):     s += 0.25
        if self._is_bent(lm, 9, 10, 12):               s += 0.15
        if self._is_bent(lm, 13, 14, 16):              s += 0.15
        return min(s, 1.0)
    
    def _score_kangula(self, lm, hs):
        """Bell: ONLY ring finger extended, all others bent."""
        s = 0.0
        if self._is_extended(lm, 13, 14, 16, 148):     s += 0.40
        if self._is_bent(lm, 5, 6, 8):                 s += 0.20
        if self._is_bent(lm, 9, 10, 12):               s += 0.20
        if self._is_bent(lm, 17, 18, 20):              s += 0.10
        if self._thumb_tucked(lm, hs):               s += 0.10
        return min(s, 1.0)
    
    def _score_chatura(self, lm, hs):
        """Four: Index+Middle+Ring extended together, Pinky bent, thumb tucked."""
        s = 0.0
        if self._is_extended(lm, 5, 6, 8):             s += 0.20
        if self._is_extended(lm, 9, 10, 12):           s += 0.20
        if self._is_extended(lm, 13, 14, 16):          s += 0.20
        if self._is_bent(lm, 17, 18, 20):              s += 0.20
        if self._thumb_tucked(lm, hs):               s += 0.10
        if self._spread(lm, 5, 13, hs) < 0.45:      s += 0.10
        return min(s, 1.0)
    
    def _score_bhramara(self, lm, hs):
        """Bee: Index curled to its own base, middle+ring+pinky extended,
        thumb near index DIP."""
        s = 0.0
        if self._dist(lm[8], lm[5]) / hs < 0.25:   s += 0.35
        if self._is_extended(lm, 9, 10, 12, 145):      s += 0.20
        if self._is_extended(lm, 13, 14, 16, 145):     s += 0.20
        if self._is_extended(lm, 17, 18, 20, 145):     s += 0.10
        if self._dist(lm[4], lm[7]) / hs < 0.18:   s += 0.15
        return min(s, 1.0)
    
    def _score_hamsasya(self, lm, hs):
        """Swan Beak: Thumb+Index pinch (<0.10), middle half-out, ring+pinky bent."""
        s = 0.0
        if self._tips_touching(lm, 4, 8, hs, 0.10):     s += 0.45
        angle = self._finger_angle(lm, 9, 10, 12)
        if 130 <= angle <= 158:                         s += 0.25
        if self._is_bent(lm, 13, 14, 16):              s += 0.15
        if self._is_bent(lm, 17, 18, 20):              s += 0.15
        return min(s, 1.0)
    
    def _score_hamsapaksha(self, lm, hs):
        """Swan Wing: Thumb+Index+Middle+Ring extended, Pinky bent."""
        s = 0.0
        if self._thumb_extended(lm, hs):                  s += 0.15
        if self._is_extended(lm, 5, 6, 8):             s += 0.20
        if self._is_extended(lm, 9, 10, 12):           s += 0.20
        if self._is_extended(lm, 13, 14, 16):          s += 0.20
        if self._is_bent(lm, 17, 18, 20):              s += 0.25
        return min(s, 1.0)
    
    def _score_sandamsha(self, lm, hs):
        """Tongs: Very tight thumb+index pinch (<0.08), ALL others bent.
        Tighter than Hamsasya. Penalize if middle is extended."""
        d = self._dist(lm[4], lm[8]) / hs
        s = 0.0
        if d < 0.08:    s += 0.50
        elif d < 0.11:  s += 0.30
        if self._is_bent(lm, 9, 10, 12, 115):          s += 0.15
        if self._is_bent(lm, 13, 14, 16, 115):         s += 0.15
        if self._is_bent(lm, 17, 18, 20, 115):         s += 0.15
        if self._is_extended(lm, 9, 10, 12, 145):      s -= 0.20
        return max(min(s, 1.0), 0.0)
    
    def _score_mukula(self, lm, hs):
        """Flower Bud: All 5 fingertips converging to one point."""
        import numpy as np
        tips = [lm[4], lm[8], lm[12], lm[16], lm[20]]
        cx = np.mean([t[0] for t in tips])
        cy = np.mean([t[1] for t in tips])
        avg_d = np.mean([
            self._dist(t, (cx, cy, 0)) / hs for t in tips
        ])
        if avg_d < 0.08: return 1.0
        if avg_d < 0.12: return 0.75
        if avg_d < 0.16: return 0.50
        return 0.0
    
    def _score_tamrachuda(self, lm, hs):
        """Rooster Comb: Thumb up AND index forward, others bent."""
        s = 0.0
        if self._thumb_extended(lm, hs):                  s += 0.35
        if lm[4][1] < lm[2][1]:                      s += 0.15
        if self._is_extended(lm, 5, 6, 8, 155):        s += 0.25
        if self._is_bent(lm, 9, 10, 12):               s += 0.10
        if self._is_bent(lm, 13, 14, 16):              s += 0.10
        if self._is_bent(lm, 17, 18, 20):              s += 0.05
        return min(s, 1.0)
    
    def _score_trishula(self, lm, hs):
        """Trident: Index+Middle+Ring extended and spread wide, Pinky bent."""
        s = 0.0
        if self._is_extended(lm, 5, 6, 8):             s += 0.20
        if self._is_extended(lm, 9, 10, 12):           s += 0.20
        if self._is_extended(lm, 13, 14, 16):          s += 0.20
        sp = self._spread(lm, 5, 13, hs)
        if sp > 0.35:                                 s += 0.20
        if sp > 0.45:                                 s += 0.10
        if self._is_bent(lm, 17, 18, 20):              s += 0.10
        return min(s, 1.0)
    
    def _score_tarjani(self, lm, hs):
        """Warning Point: Index pointing, THUMB EXTENDED outward.
        Differs from Suchi where thumb is tucked."""
        s = 0.0
        if self._is_extended(lm, 5, 6, 8, 160):        s += 0.40
        if self._is_bent(lm, 9, 10, 12, 115):          s += 0.15
        if self._is_bent(lm, 13, 14, 16, 115):         s += 0.15
        if self._is_bent(lm, 17, 18, 20, 115):         s += 0.15
        if self._thumb_extended(lm, hs):                   s += 0.15
        return min(s, 1.0)
        
    def _score_abhaya(self, lm, hs):
        """Blessing: Pataka shape + palm facing outward (Z coords)."""
        s = self._score_pataka(lm, hs) * 0.70
        import numpy as np
        tips_z = np.mean([lm[8][2], lm[12][2], lm[16][2], lm[20][2]])
        if lm[0][2] - tips_z > 0.05:                s += 0.30
        return min(s, 1.0)
    
    def _score_varada(self, lm, hs):
        """Boon Giving: Open hand, palm DOWN, fingers pointing downward."""
        s = 0.0
        if self._all_four_extended(lm, 145):                  s += 0.40
        import numpy as np
        tips_y = np.mean([lm[8][1], lm[12][1], lm[16][1], lm[20][1]])
        if tips_y > lm[0][1]:                        s += 0.40
        if self._thumb_extended(lm, hs):                   s += 0.20
        return min(s, 1.0)
    
    def _score_vyakhyana(self, lm, hs):
        """Explanation: Thumb+Index pinch, palm out, others extended."""
        s = 0.0
        if self._tips_touching(lm, 4, 8, hs, 0.11):     s += 0.35
        if self._is_extended(lm, 9, 10, 12, 145):      s += 0.20
        if self._is_extended(lm, 13, 14, 16, 145):     s += 0.20
        if self._is_extended(lm, 17, 18, 20, 145):     s += 0.15
        import numpy as np
        tips_z = np.mean([lm[9][2], lm[13][2]])
        if lm[0][2] - tips_z > 0.03:                s += 0.10
        return min(s, 1.0)
    
    def _score_vismaya(self, lm, hs):
        """Wonder: All 5 spread like Alapadma but palm facing inward."""
        s = self._score_alapadma(lm, hs) * 0.70
        import numpy as np
        tips_z = np.mean([lm[8][2], lm[12][2], lm[16][2], lm[20][2]])
        if tips_z - lm[0][2] > 0.03:                s += 0.30
        return min(s, 1.0)
    # ==================== MUDRA GROUPS INITIALIZATION ====================
    
    def _init_mudra_groups(self):
        """Initialize mudra groups mapping"""
        self.mudra_groups = {
            'all_open': ['Alapadma', 'Ardhachandra', 'Arala', 'Vismaya', 
                       'Hamsapaksha', 'Varada', 'Abhaya', 'Sarpashirsha'],
            'four_open': ['Pataka', 'Sarpashirsha', 'Chatura', 'Tripataka', 'Trishula'],
            'fist': ['Mushti', 'Shikhara', 'Tamrachuda', 'Kapittha'],
            'pinch': ['Hamsasya', 'Sandamsha', 'Mayura', 'Mukula', 
                     'Katakamukha', 'Chandrakala', 'Vyakhyana', 
                     'Kapittha', 'Bhramara'],
            'one_up': ['Suchi', 'Tarjani', 'Kangula', 'Shikhara'],
            'two_up': ['Kartarimukha', 'Ardhapataka', 'Mrigashirsha', 
                      'Simhamukha', 'Hamsapaksha'],
            'three_up': ['Tripataka', 'Katakamukha', 'Hamsapaksha', 'Chatura'],
            'mixed': ['Arala', 'Shukatunda', 'Bhramara', 'Padmakosha', 'Kangula']
        }
        
        # Map mudra names to scoring functions
        self.mudra_scorers = {
            'Pataka': self._score_pataka,
            'Alapadma': self._score_alapadma,
            'Mushti': self._score_mushti,
            'Shikhara': self._score_shikhara,
            'Suchi': self._score_suchi,
            'Tripataka': self._score_tripataka,
            'Kartarimukha': self._score_kartarimukha,
            'Hamsasya': self._score_hamsasya,
            'Sandamsha': self._score_sandamsha,
            'Mayura': self._score_mayura,
            'Ardhachandra': self._score_ardhachandra,
            'Arala': self._score_arala,
            'Shukatunda': self._score_shukatunda,
            'Kapittha': self._score_kapittha,
            'Katakamukha': self._score_katakamukha,
            'Chandrakala': self._score_chandrakala,
            'Padmakosha': self._score_padmakosha,
            'Sarpashirsha': self._score_sarpashirsha,
            'Mrigashirsha': self._score_mrigashirsha,
            'Simhamukha': self._score_simhamukha,
            'Kangula': self._score_kangula,
            'Chatura': self._score_chatura,
            'Bhramara': self._score_bhramara,
            'Hamsapaksha': self._score_hamsapaksha,
            'Mukula': self._score_mukula,
            'Tamrachuda': self._score_tamrachuda,
            'Trishula': self._score_trishula,
            'Tarjani': self._score_tarjani,
            'Abhaya': self._score_abhaya,
            'Varada': self._score_varada,
            'Vyakhyana': self._score_vyakhyana,
            'Vismaya': self._score_vismaya,
        }
    
    def _score_tripataka(self, lm, hs):
        """Tripataka: Index+Middle+Pinky extended; Ring finger bent; thumb tucked"""
        s = 0.0
        if self._is_extended(lm, 5, 6, 8):
            s += 0.22
        if self._is_extended(lm, 9, 10, 12):
            s += 0.22
        if self._is_bent(lm, 13, 14, 16):
            s += 0.27
        if self._is_extended(lm, 17, 18, 20):
            s += 0.20
        if self._thumb_tucked(lm, hs):
            s += 0.09
        return min(s, 1.0)
    
    def _score_kartarimukha(self, lm, hs):
        """Kartarimukha: Index+Middle both extended AND spread apart, Ring+Pinky bent"""
        s = 0.0
        if self._is_extended(lm, 5, 6, 8):
            s += 0.20
        if self._is_extended(lm, 9, 10, 12):
            s += 0.20
        spread = self._spread(lm, 5, 9, hs)
        if spread > 0.20:
            s += 0.25
        if spread > 0.27:  # bonus
            s += 0.10
        if self._is_bent(lm, 13, 14, 16):
            s += 0.15
        if self._is_bent(lm, 17, 18, 20):
            s += 0.10
        return min(s, 1.0)
    
    def recognize_single(self, landmarks, handedness, debug=False):
        """
        Main single-hand mudra recognition with scoring system
        Returns: (mudra_name: str, score: float)
        If debug=True: (mudra_name, score, debug_dict)
        """
        if not landmarks or len(landmarks) != 21:
            return ("Unknown", 0.0) if not debug else ("Unknown", 0.0, {})
        
        # Step 1: Normalize hand orientation
        lm = self._normalize_hand(landmarks, handedness)
        hs = self._hand_size(lm)
        
        # Step 2: Stage 1 pre-filter
        groups, flags = self._get_stage1_groups(lm, hs)
        
        # Step 3: Get candidate mudras from matched groups
        candidates = set()
        for group in groups:
            candidates.update(self.mudra_groups.get(group, []))
        
        # Step 4: Score only candidate mudras
        scores = {}
        for mudra_name in candidates:
            if mudra_name in self.mudra_scorers:
                scores[mudra_name] = self.mudra_scorers[mudra_name](lm, hs)
        
        # Step 5: Pick highest scoring mudra above threshold
        if scores:
            best_mudra = max(scores, key=scores.get)
            best_score = scores[best_mudra]
            
            # Apply tie-breaker rules
            if best_score >= 0.60:
                scores = self._apply_tiebreakers(scores, lm, hs)
                best_mudra = max(scores, key=scores.get)
                best_score = scores[best_mudra]
            
            if best_score >= 0.60:
                result = (best_mudra, best_score)
            else:
                result = ("Unknown", best_score)
        else:
            result = ("Unknown", 0.0)
        
        # Step 6: Apply temporal smoothing with hysteresis
        final_mudra = self._apply_temporal_smoothing(handedness, result[0])
        
        if debug:
            debug_info = {
                'stage1_groups': groups,
                'flags': flags,
                'top5': sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5],
                'finger_angles': self._get_finger_angles(lm),
                'thumb_state': 'tucked' if flags['thumb_tuck'] else 'extended',
                'hand_size': hs,
                'candidates': list(candidates)
            }
            return (final_mudra, result[1], debug_info)
        
        return (final_mudra, result[1])
    
    def recognize_two_hand(self, hand1_data, hand2_data):
        """
        Two-hand mudra recognition
        hand1_data = (landmarks, handedness)
        hand2_data = (landmarks, handedness)
        """
        # For now, return best single-hand result
        # TODO: Implement two-hand mudras in Phase 6
        lm1, h1 = hand1_data
        lm2, h2 = hand2_data
        
        result1 = self.recognize_single(lm1, h1)
        result2 = self.recognize_single(lm2, h2)
        
        # Return the higher scoring mudra
        if result1[1] > result2[1]:
            return result1
        else:
            return result2
    
    def _apply_tiebreakers(self, scores, lm, hs):
        """Apply tiebreaker rules to resolve ambiguous cases"""
        # 1. Shikhara vs Mushti — thumb position is decisive
        if scores.get('Shikhara',0) > 0.40 or scores.get('Mushti',0) > 0.40:
            thumb_up   = lm[4][1] < lm[2][1]
            thumb_tuck = self._thumb_tucked(lm, hs)
            if thumb_up and not thumb_tuck:
                scores['Mushti'] = 0.0
            elif thumb_tuck and not thumb_up:
                scores['Shikhara'] = 0.0

        # 2. Alapadma beats Pataka when both high (spread > closed)
        if scores.get('Alapadma',0)>0.55 and scores.get('Pataka',0)>0.55:
            scores['Pataka'] = 0.0

        # 3. Sandamsha vs Hamsasya — middle finger state decides
        if scores.get('Sandamsha',0)>0.55 and scores.get('Hamsasya',0)>0.55:
            if self._is_extended(lm, 9, 10, 12, 145):
                scores['Sandamsha'] = 0.0
            else:
                scores['Hamsasya'] = 0.0

        # 4. Tarjani vs Suchi — thumb state decides
        if scores.get('Tarjani',0)>0.55 and scores.get('Suchi',0)>0.55:
            if self._thumb_extended(lm, hs):
                scores['Suchi'] = 0.0
            else:
                scores['Tarjani'] = 0.0

        # 5. Chatura vs Tripataka — pinky state decides
        if scores.get('Chatura',0)>0.55 and scores.get('Tripataka',0)>0.55:
            if self._is_bent(lm, 17, 18, 20):
                scores['Tripataka'] = 0.0
            else:
                scores['Chatura'] = 0.0

        # 6. Tamrachuda vs Shikhara — index extension decides
        if scores.get('Tamrachuda',0)>0.55 and scores.get('Shikhara',0)>0.55:
            if self._is_extended(lm, 5, 6, 8, 155):
                scores['Shikhara'] = 0.0
            else:
                scores['Tamrachuda'] = 0.0

        return scores
    
    def _apply_temporal_smoothing(self, handedness, current_mudra):
        """Apply temporal smoothing with hysteresis to prevent flickering"""
        if handedness == 'Left':
            history = self.left_hand_history
            confirmed = self.left_confirmed
        else:
            history = self.right_hand_history
            confirmed = self.right_confirmed
        
        # Add current mudra to history
        history.append(current_mudra)
        
        # Hysteresis logic
        if confirmed is not None:
            confirmed_count = history.count(confirmed)
            if confirmed_count >= 3:  # RELEASE_THRESH = 3
                confirmed = None  # release it
        
        # Get most common mudra
        if history:
            candidate = Counter(history).most_common(1)[0][0]
            candidate_count = history.count(candidate)
            
            # Confirm new mudra after 4/7 frames
            if confirmed is None and candidate_count >= 4:  # CONFIRM_THRESH = 4
                confirmed = candidate
        else:
            candidate = "Unknown"
        
        # Update confirmed state
        if handedness == 'Left':
            self.left_confirmed = confirmed
        else:
            self.right_confirmed = confirmed
        
        return confirmed if confirmed is not None else candidate
        
        # Extract finger states and angles
        for finger_name in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            angle = self.get_finger_angle(landmarks, finger_name)
            state = self.get_finger_state_from_angle(angle)
            
            features['finger_angles'][finger_name] = angle
            features['finger_states'][finger_name] = state
        
        # Calculate finger spread
        features['finger_spread'] = self.get_finger_spread(landmarks)
        
        # Calculate thumb-index distance
        thumb_tip = landmarks[self.finger_tip_indices['thumb']]
        index_tip = landmarks[self.finger_tip_indices['index']]
        features['thumb_index_distance'] = self.calculate_distance(thumb_tip, index_tip)
        
        # Calculate palm orientation (simplified)
        wrist = landmarks[0]
        middle_mcp = landmarks[self.finger_mcp_indices['middle']]
        features['palm_orientation'] = math.atan2(middle_mcp[1] - wrist[1], middle_mcp[0] - wrist[0])
        
        return features
    
    def _init_mudra_rules(self):
        """Initialize comprehensive mudra rules dictionary"""
        self.mudra_rules = {
            "Pataka": {
                "extended": ["thumb", "index", "middle", "ring", "pinky"]
            },
            "Tripataka": {
                "extended": ["thumb", "index", "middle", "pinky"],
                "bent": ["ring"]
            },
            "Ardhapataka": {
                "extended": ["thumb", "index", "middle", "ring"],
                "bent": ["pinky"]
            },
            "Kartarimukha": {
                "extended": ["thumb", "index", "middle"],
                "bent": ["ring", "pinky"],
                "thumb_index_touch": True
            },
            "Mayura": {
                "extended": ["thumb", "pinky"],
                "bent": ["index", "middle", "ring"],
                "thumb_pinky_touch": True
            },
            "Ardhachandra": {
                "extended": ["thumb", "index", "middle", "ring", "pinky"],
                "thumb_extended_outward": True
            },
            "Arala": {
                "extended": ["thumb", "index", "middle", "ring"],
                "half-bent": ["pinky"]
            },
            "Shukatunda": {
                "extended": ["thumb", "index"],
                "bent": ["middle", "ring", "pinky"]
            },
            "Mushti": {
                "bent": ["thumb", "index", "middle", "ring", "pinky"]
            },
            "Shikhara": {
                "extended": ["thumb"],
                "bent": ["index", "middle", "ring", "pinky"]
            },
            "Kapitta": {
                "extended": ["thumb"],
                "half-bent": ["index", "middle", "ring", "pinky"],
                "thumb_index_touch": True
            },
            "Katakamukha": {
                "extended": ["index", "middle"],
                "bent": ["thumb", "ring", "pinky"],
                "thumb_middle_touch": True
            },
            "Suchi": {
                "extended": ["thumb", "index"],
                "bent": ["middle", "ring", "pinky"]
            },
            "Chandrakala": {
                "extended": ["thumb", "index"],
                "bent": ["middle", "ring", "pinky"],
                "thumb_index_circle": True
            },
            "Padmakosha": {
                "fingertips_touch": True
            },
            "Sarpashirsha": {
                "extended": ["index", "middle", "ring", "pinky"],
                "bent": ["thumb"],
                "fingers_curved": True
            },
            "Mrigashirsha": {
                "extended": ["thumb", "index", "middle"],
                "bent": ["ring", "pinky"],
                "thumb_middle_touch": True
            },
            "Simhamukha": {
                "extended": ["thumb", "index", "middle"],
                "bent": ["ring", "pinky"],
                "thumb_index_spread": True
            },
            "Kangula": {
                "extended": ["index"],
                "bent": ["thumb", "middle", "ring", "pinky"],
                "thumb_curved": True
            },
            "Alapadma": {
                "extended": ["thumb", "index", "middle", "ring", "pinky"],
                "finger_spread_high": True
            },
            "Chatura": {
                "extended": ["thumb", "index", "middle", "pinky"],
                "bent": ["ring"]
            },
            "Bhramara": {
                "extended": ["thumb", "index", "middle", "ring"],
                "bent": ["pinky"],
                "thumb_middle_touch": True,
                "fingers_curved": True
            },
            "Hamsasya": {
                "thumb_index_touch": True,
                "extended": ["middle", "ring", "pinky"]
            },
            "Hamsapaksha": {
                "extended": ["thumb", "index", "middle"],
                "bent": ["ring", "pinky"],
                "thumb_index_angle": 45
            },
            "Sandamsha": {
                "extended": ["thumb", "index"],
                "bent": ["middle", "ring", "pinky"],
                "thumb_index_opposite": True
            },
            "Mukula": {
                "fingertips_touch": True
            },
            "Tamrachuda": {
                "extended": ["thumb", "index", "middle"],
                "bent": ["ring", "pinky"],
                "thumb_curved": True
            },
            "Trishula": {
                "extended": ["thumb", "index", "middle"],
                "bent": ["ring", "pinky"],
                "thumb_spread": True
            }
        }
    
    def check_mudra_rule(self, mudra_name, features, landmarks):
        """Check if features match a specific mudra rule"""
        rule = self.mudra_rules[mudra_name]
        finger_states = features['finger_states']
        
        # Check finger state requirements
        if 'extended' in rule:
            for finger in rule['extended']:
                if finger_states.get(finger) != 'extended':
                    return False
        
        if 'bent' in rule:
            for finger in rule['bent']:
                if finger_states.get(finger) != 'bent':
                    return False
        
        if 'half-bent' in rule:
            for finger in rule['half-bent']:
                if finger_states.get(finger) != 'half-bent':
                    return False
        
        # Check special conditions
        if rule.get('thumb_index_touch'):
            if not self.is_thumb_touching_index(landmarks):
                return False
        
        if rule.get('thumb_pinky_touch'):
            if not self.is_thumb_touching_pinky(landmarks):
                return False
        
        if rule.get('thumb_middle_touch'):
            if not self.is_thumb_touching_middle(landmarks):
                return False
        
        if rule.get('fingertips_touch'):
            if not self.are_fingertips_touching(landmarks):
                return False
        
        if rule.get('thumb_extended_outward'):
            if not self.is_thumb_extended_outward(landmarks):
                return False
        
        if rule.get('finger_spread_high'):
            if features['finger_spread'] <= 0.12:
                return False
        
        if rule.get('fingers_curved'):
            # Check if fingers are in curved position
            if not self.are_fingers_curved(landmarks):
                return False
        
        return True
    
    def is_thumb_touching_pinky(self, landmarks):
        """Check if thumb tip is touching pinky tip"""
        thumb_tip = landmarks[self.finger_tip_indices['thumb']]
        pinky_tip = landmarks[self.finger_tip_indices['pinky']]
        distance = self.calculate_distance(thumb_tip, pinky_tip)
        return distance < 0.05
    
    def is_thumb_touching_middle(self, landmarks):
        """Check if thumb tip is touching middle tip"""
        thumb_tip = landmarks[self.finger_tip_indices['thumb']]
        middle_tip = landmarks[self.finger_tip_indices['middle']]
        distance = self.calculate_distance(thumb_tip, middle_tip)
        return distance < 0.05
    
    def are_fingers_curved(self, landmarks):
        """Check if fingers are in curved position"""
        curved_count = 0
        for finger in ['index', 'middle', 'ring', 'pinky']:
            angle = self.get_finger_angle(landmarks, finger)
            if 120 < angle < 160:  # Curved but not fully bent
                curved_count += 1
        return curved_count >= 2
    
    def get_finger_spread(self, landmarks):
        index_tip = landmarks[self.finger_tip_indices['index']]
        middle_tip = landmarks[self.finger_tip_indices['middle']]
        ring_tip = landmarks[self.finger_tip_indices['ring']]
        pinky_tip = landmarks[self.finger_tip_indices['pinky']]
        
        spread_1 = self.calculate_distance(index_tip, middle_tip)
        spread_2 = self.calculate_distance(middle_tip, ring_tip)
        spread_3 = self.calculate_distance(ring_tip, pinky_tip)
        
        return (spread_1 + spread_2 + spread_3) / 3
    
    def is_thumb_touching_index(self, landmarks):
        """Check if thumb tip is touching index tip (for Hamsasya)"""
        thumb_tip = landmarks[self.finger_tip_indices['thumb']]
        index_tip = landmarks[self.finger_tip_indices['index']]
        distance = self.calculate_distance(thumb_tip, index_tip)
        return distance < 0.05
    
    def are_fingertips_touching(self, landmarks):
        """Check if all fingertips are close together (for Mukula)"""
        thumb_tip = landmarks[self.finger_tip_indices['thumb']]
        index_tip = landmarks[self.finger_tip_indices['index']]
        middle_tip = landmarks[self.finger_tip_indices['middle']]
        ring_tip = landmarks[self.finger_tip_indices['ring']]
        pinky_tip = landmarks[self.finger_tip_indices['pinky']]
        
        # Calculate distances between all fingertip pairs
        distances = [
            self.calculate_distance(thumb_tip, index_tip),
            self.calculate_distance(index_tip, middle_tip),
            self.calculate_distance(middle_tip, ring_tip),
            self.calculate_distance(ring_tip, pinky_tip)
        ]
        
        # All distances should be small for fingertips to be touching
        return all(dist < 0.08 for dist in distances)
    
    def is_thumb_extended_outward(self, landmarks):
        """Check if thumb is extended outward from hand"""
        thumb_tip = landmarks[self.finger_tip_indices['thumb']]
        index_tip = landmarks[self.finger_tip_indices['index']]
        
        thumb_index_distance = self.calculate_distance(thumb_tip, index_tip)
        return thumb_index_distance > 0.15
    
    def apply_temporal_smoothing(self, hand_type, current_mudra):
        """Apply temporal smoothing with 7-frame window and majority voting"""
        if hand_type == 'Left':
            self.left_hand_history.append(current_mudra)
            history = self.left_hand_history
        else:
            self.right_hand_history.append(current_mudra)
            history = self.right_hand_history
        
        if len(history) < 4:
            return current_mudra
        
        # Use majority voting for stability
        mudra_counts = {}
        for mudra in history:
            mudra_counts[mudra] = mudra_counts.get(mudra, 0) + 1
        
        # Return the most frequent mudra
        most_frequent = max(mudra_counts.items(), key=lambda x: x[1])[0]
        confidence = mudra_counts[most_frequent] / len(history)
        
        # Store confidence for debugging
        self.last_confidence = confidence
        
        return most_frequent
    
    def recognize_mudra(self, landmarks, hand_type='Right'):
        """Main mudra recognition function with improved accuracy"""
        if not landmarks or len(landmarks) != 21:
            return "Unknown"
        
        # Step 1: Normalize hand orientation for consistent behavior
        normalized_landmarks = self.normalize_hand_orientation(landmarks, hand_type)
        
        # Step 2: Extract comprehensive features
        features = self.extract_hand_features(normalized_landmarks)
        
        # Store features for debugging
        self.last_features = features
        
        # Step 3: Check mudra rules
        for mudra_name in self.mudra_rules.keys():
            if self.check_mudra_rule(mudra_name, features, normalized_landmarks):
                # Apply temporal smoothing
                return self.apply_temporal_smoothing(hand_type, mudra_name)
        
        # No mudra matched
        return self.apply_temporal_smoothing(hand_type, "Unknown")
    
    def get_debug_info(self):
        """Get comprehensive debugging information"""
        debug_info = {
            'finger_states': self.last_features.get('finger_states', {}),
            'finger_angles': self.last_features.get('finger_angles', {}),
            'confidence': self.last_confidence,
            'finger_spread': self.last_features.get('finger_spread', 0),
            'thumb_index_distance': self.last_features.get('thumb_index_distance', 0)
        }
        return debug_info
