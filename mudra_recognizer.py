import math
import numpy as np
from collections import deque, Counter


class MudraRecognizer:

    # ─────────────────────────────────────────
    # INIT
    # ─────────────────────────────────────────

    def __init__(self):
        self.left_history   = deque(maxlen=14)
        self.right_history  = deque(maxlen=14)
        self.left_confirmed  = None
        self.right_confirmed = None
        self.CONFIRM_THRESH  = 4
        self.RELEASE_THRESH  = 6
        self.display_scores_right = {}
        self.display_scores_left  = {}
        self.DISPLAY_ALPHA   = 0.15
        self.samyuktha_enabled = False
        self.debug_mode      = False

    # ─────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────

    def _dist(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def _hand_size(self, lm):
        return max(self._dist(lm[0], lm[9]), 0.01)

    def _angle(self, a, b, c):
        v1 = [a[0]-b[0], a[1]-b[1]]
        v2 = [c[0]-b[0], c[1]-b[1]]
        m1 = math.sqrt(v1[0]**2+v1[1]**2)
        m2 = math.sqrt(v2[0]**2+v2[1]**2)
        if m1 < 1e-6 or m2 < 1e-6:
            return 180.0
        dot = v1[0]*v2[0]+v1[1]*v2[1]
        cos_a = max(-1.0, min(1.0, dot/(m1*m2)))
        return math.degrees(math.acos(cos_a))

    def _fangle(self, lm, mcp, pip, tip):
        return self._angle(lm[mcp], lm[pip], lm[tip])

    def _extended(self, lm, mcp, pip, tip, t=148):
        return self._fangle(lm, mcp, pip, tip) > t

    def _bent(self, lm, mcp, pip, tip, t=125):
        return self._fangle(lm, mcp, pip, tip) < t

    def _touching(self, lm, i, j, hs, thresh=0.13):
        return self._dist(lm[i], lm[j]) / hs < thresh

    def _spread(self, lm, i, j, hs):
        return self._dist(lm[i], lm[j]) / hs

    def _four_ext(self, lm, t=145):
        return (self._extended(lm,5,6,8,t) and
                self._extended(lm,9,10,12,t) and
                self._extended(lm,13,14,16,t) and
                self._extended(lm,17,18,20,t))

    def _four_bent(self, lm, t=120):
        return (self._bent(lm,5,6,8,t) and
                self._bent(lm,9,10,12,t) and
                self._bent(lm,13,14,16,t) and
                self._bent(lm,17,18,20,t))

    def _thumb_out(self, lm, hs):
        return self._dist(lm[4], lm[5]) / hs > 0.35

    def _thumb_tucked(self, lm, hs):
        return self._dist(lm[4], lm[5]) / hs < 0.38

    def _normalize(self, lm, handedness):
        # Mirror left hand only. NO rotation.
        if handedness == 'Left':
            lm = [(1.0-p[0], p[1], p[2]) for p in lm]
        return lm

    def _finger_angles(self, lm):
        return {
            'thumb':  self._fangle(lm,1,2,4),
            'index':  self._fangle(lm,5,6,8),
            'middle': self._fangle(lm,9,10,12),
            'ring':   self._fangle(lm,13,14,16),
            'pinky':  self._fangle(lm,17,18,20),
        }

    def _index_curl(self, lm):
        """Better measure of index curl.
        Uses distance from index tip to index MCP
        normalized by finger length."""
        tip_to_mcp = self._dist(lm[8], lm[5])
        mcp_to_pip = self._dist(lm[5], lm[6])
        pip_to_tip = self._dist(lm[6], lm[8])
        finger_len = mcp_to_pip + pip_to_tip
        # When straight: tip_to_mcp ≈ finger_len
        # When curled: tip_to_mcp << finger_len
        if finger_len < 0.001: return 1.0
        return tip_to_mcp / finger_len

    # ─────────────────────────────────────────
    # STAGE 1 — PRE-FILTER GROUPS
    # ─────────────────────────────────────────

    def _get_groups(self, lm, hs):
        fe   = self._four_ext(lm, 142)
        fb   = self._four_bent(lm, 118)
        to   = self._thumb_out(lm, hs)
        tt   = self._thumb_tucked(lm, hs)
        p48  = self._touching(lm, 4, 8, hs, 0.16)
        p416 = self._touching(lm, 4, 16, hs, 0.20)
        ie   = self._extended(lm, 5, 6, 8, 150)
        me   = self._extended(lm, 9, 10, 12, 150)
        re   = self._extended(lm, 13, 14, 16, 150)
        pe   = self._extended(lm, 17, 18, 20, 150)
        ec   = sum([ie, me, re, pe])

        # Use raw angle for ring — more reliable than threshold
        ring_angle = self._fangle(lm, 13, 14, 16)
        ring_bent  = ring_angle < 158  # even slightly bent

        droop = sum([
            lm[8][1]  > lm[7][1],
            lm[12][1] > lm[11][1],
            lm[16][1] > lm[15][1],
            lm[20][1] > lm[19][1],
        ])
        deep_droop = sum([
            lm[8][1]  > lm[5][1],
            lm[12][1] > lm[9][1],
            lm[16][1] > lm[13][1],
            lm[20][1] > lm[17][1],
        ])

        groups = []
        flags = {
            'four_ext': fe, 'four_bent': fb,
            'thumb_ext': to, 'thumb_tuck': tt,
            'pinch_14': p48, 'ext_count': ec,
            'droop': droop, 'ring_bent': ring_bent,
            'ring_angle': ring_angle,
        }

        # Index family: all 4 mudras go in one group
        # Scoring functions separate them by pinch+angle
        ia_val    = self._fangle(lm, 5, 6, 8)
        pinch_val = self._dist(lm[4], lm[8]) / hs

        index_family = (
            ia_val < 162 and
            ia_val > 82 and
            self._extended(lm, 9, 10, 12, 122) and
            self._extended(lm, 13, 14, 16, 122)
        )
        if index_family:
            groups.append('index_family')

        # Check tip cluster FIRST
        tips = [lm[4],lm[8],lm[12],lm[16],lm[20]]
        cx = sum(t[0] for t in tips)/5
        cy = sum(t[1] for t in tips)/5
        tip_cluster = max(
            self._dist(t,(cx,cy,0))/hs for t in tips
        ) < 0.28
        if tip_cluster:
            groups.append('tip_cluster')

        # Check half_bent BEFORE all_open
        all_half_bent = all([
            not self._extended(lm, 5, 6, 8, 155) and
            not self._bent(lm, 5, 6, 8, 88),
            not self._extended(lm, 9, 10, 12, 155) and
            not self._bent(lm, 9, 10, 12, 88),
            not self._extended(lm, 13, 14, 16, 155) and
            not self._bent(lm, 13, 14, 16, 88),
            not self._extended(lm, 17, 18, 20, 155) and
            not self._bent(lm, 17, 18, 20, 88),
        ])
        if all_half_bent:
            groups.append('half_bent')

        # ── OPEN HAND GROUPS ──────────────────────────
        # four_open: ALL fingers straight, thumb tucked
        # Only fires if ring is NOT bent (else Tripataka)
        if fe and tt and droop < 2 and deep_droop < 2 and not ring_bent:
            groups.append('four_open')

        # all_open only fires if NOT tip_cluster or half_bent
        if fe and to and droop < 2 and deep_droop < 2 and not tip_cluster and not all_half_bent:
            groups.append('all_open')

        # drooping: fingers extended but tips drooping
        if fe and (droop >= 2 or deep_droop >= 2):
            groups.append('drooping')

        # ring_bent: 3 fingers up, ring bent (Tripataka/Trishula)
        # Only fire if index+middle+pinky are extended
        if ring_bent and ie and me and pe and droop < 2:
            sp = self._spread(lm, 5, 17, hs)
            if sp <= 0.55:
                groups.append('ring_bent_together')  # Tripataka
            else:
                groups.append('ring_bent_spread')    # Trishula

        # ── FIST GROUPS ───────────────────────────────
        if fb:
            # Thumb tucked over fist = Mushti
            thumb_over_fist = (fb and tt)
            if thumb_over_fist:
                groups.append('thumb_tucked_fist')

            # Thumb pointing up from fist = Shikhara
            thumb_up_fist = (fb and to and lm[4][1] < lm[2][1])
            if thumb_up_fist:
                groups.append('thumb_up_fist')

            # Regular fist = Kapittha
            groups.append('fist')

        # ── PINCH GROUPS ──────────────────────────────
        if p48:
            groups.append('pinch')
        if p416:
            groups.append('ring_pinch')

        # ── TWO FINGER GROUPS ─────────────────────────
        # Use TIP spread (lm[8] to lm[12]) to separate
        # Ardhapataka (tips together) from Kartarimukha (tips spread)
        tip_spread_812 = self._dist(lm[8], lm[12]) / hs

        if ie and me and not re and not pe:
            if tip_spread_812 <= 0.32:
                groups.append('idx_mid_up')     # Ardhapataka
            else:
                groups.append('idx_mid_spread') # Kartarimukha

        # index+pinky up, middle+ring bent = Simhamukha
        me_strict = self._extended(lm, 9, 10, 12, 148)
        if ie and not me_strict and not re and pe:
            groups.append('idx_pinky_up')


        # ── ONE FINGER UP GROUPS ──────────────────────
        # index up + thumb out = Chandrakala
        if ie and not me and not re and not pe and to:
            groups.append('idx_up_thumb_out')

        # index up + thumb tucked = Suchi
        if ie and not me and not re and not pe and tt:
            groups.append('idx_up_thumb_in')

        # ── THREE BENT + PINKY UP ─────────────────────
        if (self._bent(lm, 5, 6, 8, 135) and
                self._bent(lm, 9, 10, 12, 135) and
                self._bent(lm, 13, 14, 16, 135) and
                self._extended(lm, 17, 18, 20, 132)):
            groups.append('three_mid_bent_pinky_up')

        # Middle-thumb touch = Tamrachuda territory
        p412 = self._touching(lm, 4, 12, hs, 0.22)
        if p412:
            groups.append('mid_pinch')

        # three fingers up + thumb tucked = Trishula
        three_up_imr = (ie and me and re and not pe)
        if three_up_imr and tt:
            groups.append('three_up_tuck')  # Trishula

        # Only pinky extended, others bent = Chatura
        pinky_only_up = (pe and not ie and not me and not re)
        if pinky_only_up:
            groups.append('pinky_only')

        if not groups:
            groups.append('mixed')

        return groups, flags

    GROUPS = {
        'all_open':               ['Alapadma', 'Ardhachandra', 'Arala'],
        'four_open':              ['Pataka'],
        'ring_bent_together':     ['Tripataka'],
        'ring_bent_spread':       ['Trishula'],
        'idx_mid_up':             ['Ardhapataka'],
        'idx_mid_spread':         ['Kartarimukha'],
        'idx_pinky_up':           ['Simhamukha'],
        'index_family': ['Arala', 'Shukatunda',
                         'Hamsasya', 'Katakamukha'],
        'idx_up_thumb_out':       ['Chandrakala'],
        'idx_up_thumb_in':        ['Suchi'],
        'three_mid_bent_pinky_up':['Mrigashirsha', 'Hamsapaksha'],
        'three_up_tuck':          ['Trishula'],
        'pinky_only':             ['Chatura'],
        'drooping':               ['Sarpashirsha'],
        'thumb_tucked_fist':      ['Mushti'],
        'thumb_up_fist':          ['Shikhara', 'Tamrachuda'],
        'fist':                   ['Kapittha'],
        'pinch':                  ['Hamsasya', 'Sandamsha', 'Mukula',
                                   'Katakamukha', 'Chandrakala',
                                   'Kapittha'],
        'ring_pinch':             ['Mayura'],
        'mid_pinch':              ['Tamrachuda', 'Bhramara'],
        'tip_cluster':            ['Mukula', 'Sandamsha'],
        'half_bent':              ['Padmakosha', 'Kangula'],
        'mixed':                  ['Shukatunda', 'Bhramara',
                                   'Padmakosha', 'Kangula'],
    }

    # ─────────────────────────────────────────
    # STAGE 2 — SCORING FUNCTIONS
    # Each returns 0.0–1.0
    # Hard gates at top: return 0.0 if key condition missing
    # ─────────────────────────────────────────

    def _s_pataka(self, lm, hs):
        """All 4 fingers very straight together. Thumb TUCKED."""
        # Hard gate: ALL fingers must be very straight
        if self._fangle(lm, 5, 6, 8) < 155:   return 0.0
        if self._fangle(lm, 9, 10, 12) < 155:  return 0.0
        if self._fangle(lm, 13, 14, 16) < 155: return 0.0
        if self._fangle(lm, 17, 18, 20) < 155: return 0.0
        if not self._thumb_tucked(lm, hs):      return 0.0
        sp = self._spread(lm, 5, 17, hs)
        if sp > 0.58:                           return 0.0
        s = 0.70
        if sp < 0.42: s += 0.30
        elif sp < 0.58: s += 0.15
        return min(s, 1.0)

    def _s_alapadma(self, lm, hs):
        """All 5 extended AND maximally spread — true fan shape."""
        if not self._four_ext(lm, 145): return 0.0
        if not self._extended(lm, 1, 2, 4, 138): return 0.0
        # ALL adjacent pairs must be spread — not just average
        s1 = self._spread(lm, 1, 5, hs)   # thumb-index
        s2 = self._spread(lm, 5, 9, hs)   # index-middle
        s3 = self._spread(lm, 9, 13, hs)  # middle-ring
        s4 = self._spread(lm, 13, 17, hs) # ring-pinky
        # Hard gate: EVERY pair must be spread
        if s1 < 0.12: return 0.0
        if s2 < 0.12: return 0.0
        if s3 < 0.12: return 0.0
        if s4 < 0.12: return 0.0
        avg = (s1+s2+s3+s4)/4
        total = self._spread(lm, 5, 17, hs)
        if avg < 0.15: return 0.0
        s = 0.30
        if avg > 0.18: s += 0.25
        if avg > 0.22: s += 0.15
        if total > 0.55: s += 0.30
        return min(s, 1.0)

    def _s_ardhachandra(self, lm, hs):
        """All 4 fingers extended together. Thumb OUT."""
        if not self._four_ext(lm, 142): return 0.0
        if not self._thumb_out(lm, hs): return 0.0
        sp = self._spread(lm, 5, 17, hs)
        if sp > 0.58: return 0.0
        s = 0.35 + 0.35
        if sp < 0.42: s += 0.30
        elif sp < 0.58: s += 0.15
        return min(s, 1.0)

    def _s_tripataka(self, lm, hs):
        """Pataka with ring finger bent."""
        ring_angle = self._fangle(lm, 13, 14, 16)
        if ring_angle > 155: return 0.0
        if not self._extended(lm, 5, 6, 8, 140): return 0.0
        if not self._extended(lm, 9, 10, 12, 140): return 0.0
        if not self._extended(lm, 17, 18, 20, 140): return 0.0
        if self._spread(lm, 5, 17, hs) > 0.48: return 0.0
        s = 0.0
        s += 0.28 if self._extended(lm, 5, 6, 8, 150) else 0.15
        s += 0.28 if self._extended(lm, 9, 10, 12, 150) else 0.15
        s += 0.28 if self._extended(lm, 17, 18, 20, 150) else 0.15
        s += 0.16 if ring_angle < 130 else 0.08
        return min(s, 1.0)

    def _s_ardhapataka(self, lm, hs):
        """Index+Middle extended TOGETHER (tips close). Ring+Pinky bent."""
        if not self._bent(lm, 13, 14, 16, 132): return 0.0
        if not self._bent(lm, 17, 18, 20, 132): return 0.0
        if not self._extended(lm, 5, 6, 8, 135): return 0.0
        if not self._extended(lm, 9, 10, 12, 135): return 0.0
        # Hard gate: tips must be together
        tip_spread = self._dist(lm[8], lm[12]) / hs
        if tip_spread > 0.35: return 0.0
        s = 0.0
        s += 0.28 if self._extended(lm, 5, 6, 8, 148) else 0.15
        s += 0.28 if self._extended(lm, 9, 10, 12, 148) else 0.15
        s += 0.22 if self._bent(lm, 13, 14, 16, 122) else 0.10
        s += 0.22 if self._bent(lm, 17, 18, 20, 122) else 0.10
        return min(s, 1.0)

    def _s_kartarimukha(self, lm, hs):
        """Index+Middle scissors spread apart. Ring+Pinky bent."""
        if not self._extended(lm, 5, 6, 8, 135): return 0.0
        if not self._extended(lm, 9, 10, 12, 135): return 0.0
        if not self._bent(lm, 13, 14, 16, 138): return 0.0
        if not self._bent(lm, 17, 18, 20, 138): return 0.0
        # Hard gate: tips must be spread apart
        tip_spread = self._dist(lm[8], lm[12]) / hs
        if tip_spread < 0.30: return 0.0
        s = 0.0
        s += 0.25 if self._extended(lm, 5, 6, 8, 145) else 0.12
        s += 0.25 if self._extended(lm, 9, 10, 12, 145) else 0.12
        s += 0.25 if self._bent(lm, 13, 14, 16, 128) else 0.10
        s += 0.25 if self._bent(lm, 17, 18, 20, 128) else 0.10
        return min(s, 1.0)

    def _s_mayura(self, lm, hs):
        """Ring+Thumb touch. Index+Middle+Pinky extended."""
        # Hard gate: ring tip must touch thumb tip
        if not self._touching(lm, 4, 16, hs, 0.18): return 0.0
        # Hard gate: index must be extended
        if not self._extended(lm, 5, 6, 8, 132): return 0.0
        # Hard gate: middle must be extended
        if not self._extended(lm, 9, 10, 12, 132): return 0.0
        # Hard gate: pinky must be extended
        if not self._extended(lm, 17, 18, 20, 132): return 0.0
        # Hard gate: ring finger must be bent (touching thumb)
        if not self._bent(lm, 13, 14, 16, 145): return 0.0
        s = 0.0
        s += 0.55 if self._touching(lm, 4, 16, hs, 0.13) else 0.35
        s += 0.15 if self._extended(lm, 5, 6, 8, 142) else 0.08
        s += 0.15 if self._extended(lm, 9, 10, 12, 142) else 0.08
        s += 0.15 if self._extended(lm, 17, 18, 20, 142) else 0.08
        return min(s, 1.0)

    def _s_arala(self, lm, hs):
        ia = self._fangle(lm, 5, 6, 8)
        pinch = self._dist(lm[4], lm[8]) / hs
        if ia > 155 or ia < 95: return 0.0
        if pinch < 0.16: return 0.0
        if not self._extended(lm, 9, 10, 12, 128): return 0.0
        if not self._extended(lm, 13, 14, 16, 128): return 0.0
        if not self._extended(lm, 17, 18, 20, 122): return 0.0
        s = 0.0
        s += 0.50 if 95 <= ia <= 155 else 0.0
        s += 0.25 if pinch >= 0.18 else 0.15
        s += 0.15 if self._extended(lm, 9, 10, 12, 142) else 0.08
        s += 0.10 if self._extended(lm, 13, 14, 16, 142) else 0.05
        return min(s, 1.0)

    def _s_shukatunda(self, lm, hs):
        ia = self._fangle(lm, 5, 6, 8)
        pinch = self._dist(lm[4], lm[8]) / hs
        if pinch > 0.18: return 0.0
        if ia > 135 or ia < 75: return 0.0
        if not self._extended(lm, 9, 10, 12, 125): return 0.0
        if self._extended(lm, 13, 14, 16, 158): return 0.0
        s = 0.0
        if pinch < 0.09: s += 0.55
        elif pinch < 0.13: s += 0.45
        elif pinch < 0.18: s += 0.35
        if ia < 100: s += 0.30
        elif ia < 118: s += 0.22
        elif ia < 135: s += 0.15
        s += 0.15 if self._extended(lm, 9, 10, 12, 142) else 0.08
        return min(s, 1.0)

    def _s_mushti(self, lm, hs):
        if not self._four_bent(lm, 118): return 0.0
        if not self._thumb_tucked(lm, hs): return 0.0
        if lm[4][1] < lm[2][1]: return 0.0
        if self._thumb_out(lm, hs): return 0.0
        s = 0.60
        s += 0.25 if self._thumb_tucked(lm, hs) else 0.0
        if self._dist(lm[4], lm[5]) / hs < 0.30: s += 0.15
        return min(s, 1.0)

    def _s_shikhara(self, lm, hs):
        if not self._four_bent(lm, 118): return 0.0
        if not self._thumb_out(lm, hs): return 0.0
        if not lm[4][1] < lm[2][1]: return 0.0
        if self._thumb_tucked(lm, hs): return 0.0
        s = 0.45
        s += 0.30 if lm[4][1] < lm[2][1] else 0.0
        s += 0.15 if self._thumb_out(lm, hs) else 0.0
        s += 0.10 if lm[4][1] < lm[0][1] else 0.0
        return min(s, 1.0)

    def _s_kapittha(self, lm, hs):
        """Shikhara + index curved pressing thumb.
        Thumb extended upward. Index curved ON thumb.
        Middle+Ring+Pinky curled into palm."""
        # Hard gate: middle ring pinky must be bent
        if not self._bent(lm, 9, 10, 12, 125): return 0.0
        if not self._bent(lm, 13, 14, 16, 125): return 0.0
        if not self._bent(lm, 17, 18, 20, 125): return 0.0
        # Hard gate: thumb must be extended
        if not self._thumb_out(lm, hs): return 0.0
        # Hard gate: index must be curved (not straight, not fully bent)
        ia = self._fangle(lm, 5, 6, 8)
        if ia > 150 or ia < 60: return 0.0
        # Index tip must be near thumb tip (pressing on it)
        if self._dist(lm[8], lm[4]) / hs > 0.25: return 0.0
        s = 0.0
        s += 0.30 if self._bent(lm, 9, 10, 12, 115) else 0.15
        s += 0.20 if self._bent(lm, 13, 14, 16, 115) else 0.10
        s += 0.20 if self._bent(lm, 17, 18, 20, 115) else 0.10
        s += 0.15 if self._thumb_out(lm, hs) else 0.0
        s += 0.15 if self._dist(lm[8], lm[4]) / hs < 0.18 else 0.08
        return min(s, 1.0)

    def _s_katakamukha(self, lm, hs):
        """Bracelet: tight pinch < 0.13.
        Index angle < 122. Ring+pinky extended.
        Three tips (thumb+index+middle) all close."""
        pinch = self._dist(lm[4], lm[8]) / hs
        ia = self._fangle(lm, 5, 6, 8)
        # Hard gate: tight pinch
        if pinch > 0.14: return 0.0
        # Hard gate: index curved
        if ia > 122: return 0.0
        # Hard gate: ring and pinky extended
        if not self._extended(lm, 13, 14, 16, 140): return 0.0
        if not self._extended(lm, 17, 18, 20, 140): return 0.0
        d48  = self._dist(lm[4], lm[8]) / hs
        d812 = self._dist(lm[8], lm[12]) / hs
        d412 = self._dist(lm[4], lm[12]) / hs
        s = 0.0
        if d48  < 0.14: s += 0.22
        if d812 < 0.18: s += 0.18
        if d412 < 0.18: s += 0.15
        s += 0.25 if self._extended(lm,13,14,16,148) else 0.12
        s += 0.20 if self._extended(lm,17,18,20,148) else 0.10
        return min(s, 1.0)

    def _s_suchi(self, lm, hs):
        """Index pointing. All others bent. Thumb TUCKED (not released)."""
        if not self._extended(lm, 5, 6, 8, 155): return 0.0
        if not self._bent(lm, 9, 10, 12, 120): return 0.0
        if not self._bent(lm, 13, 14, 16, 120): return 0.0
        if not self._bent(lm, 17, 18, 20, 120): return 0.0
        # Hard gate: thumb must be TUCKED (Chandrakala has thumb out)
        if not self._thumb_tucked(lm, hs): return 0.0
        s = 0.40
        s += 0.20 if self._bent(lm, 9, 10, 12, 108) else 0.10
        s += 0.20 if self._bent(lm, 13, 14, 16, 108) else 0.10
        s += 0.10 if self._bent(lm, 17, 18, 20, 108) else 0.05
        s += 0.10 if self._thumb_tucked(lm, hs) else 0.0
        return min(s, 1.0)

    def _s_chandrakala(self, lm, hs):
        """Suchi with thumb released/open. Index pointing, thumb free."""
        # Hard gate: index must be pointing (like Suchi)
        if not self._extended(lm, 5, 6, 8, 150): return 0.0
        # Hard gate: middle ring pinky must be bent (like Suchi)
        if not self._bent(lm, 9, 10, 12, 130): return 0.0
        if not self._bent(lm, 13, 14, 16, 130): return 0.0
        if not self._bent(lm, 17, 18, 20, 130): return 0.0
        # Hard gate: thumb must be OUT/released (unlike Suchi)
        if not self._thumb_out(lm, hs): return 0.0
        s = 0.0
        s += 0.35 if self._extended(lm, 5, 6, 8, 158) else 0.20
        s += 0.30 if self._thumb_out(lm, hs) else 0.0
        s += 0.15 if self._bent(lm, 9, 10, 12, 118) else 0.08
        s += 0.10 if self._bent(lm, 13, 14, 16, 118) else 0.05
        s += 0.10 if self._bent(lm, 17, 18, 20, 118) else 0.05
        return min(s, 1.0)

    def _s_padmakosha(self, lm, hs):
        """All 5 half-bent. No fully extended, no fully bent."""
        if self._extended(lm, 5, 6, 8, 155): return 0.0
        if self._extended(lm, 9, 10, 12, 155): return 0.0
        if self._extended(lm, 13, 14, 16, 155): return 0.0
        if self._extended(lm, 17, 18, 20, 155): return 0.0
        if self._bent(lm, 5, 6, 8, 85): return 0.0
        if self._bent(lm, 9, 10, 12, 85): return 0.0
        angles = [
            self._fangle(lm, 5, 6, 8),
            self._fangle(lm, 9, 10, 12),
            self._fangle(lm, 13, 14, 16),
            self._fangle(lm, 17, 18, 20),
        ]
        count = sum(1 for a in angles if 100 <= a <= 158)
        ta = self._fangle(lm, 1, 2, 4)
        if 95 <= ta <= 158: count += 1
        if count < 3: return 0.0
        return min(count * 0.20, 1.0)

    def _s_sarpashirsha(self, lm, hs):
        if not self._four_ext(lm, 138): return 0.0
        if not self._thumb_tucked(lm, hs): return 0.0
        droop = sum([
            lm[8][1]  > lm[7][1],
            lm[12][1] > lm[11][1],
            lm[16][1] > lm[15][1],
            lm[20][1] > lm[19][1],
        ])
        deep_droop = sum([
            lm[8][1]  > lm[5][1],
            lm[12][1] > lm[9][1],
            lm[16][1] > lm[13][1],
            lm[20][1] > lm[17][1],
        ])
        if droop < 2 and deep_droop < 1: return 0.0
        s = 0.30
        s += 0.12 * droop
        s += 0.08 * deep_droop
        s += 0.10 if self._thumb_tucked(lm, hs) else 0.0
        return min(s, 1.0)

    def _s_mrigashirsha(self, lm, hs):
        """Index+Middle+Ring drooping. Pinky+Thumb UP."""
        if not self._bent(lm, 5, 6, 8, 135): return 0.0
        if not self._bent(lm, 9, 10, 12, 135): return 0.0
        if not self._bent(lm, 13, 14, 16, 135): return 0.0
        if not self._extended(lm, 17, 18, 20, 132): return 0.0
        if not self._thumb_out(lm, hs): return 0.0
        s = 0.0
        s += 0.20 if self._bent(lm, 5, 6, 8, 125) else 0.10
        s += 0.20 if self._bent(lm, 9, 10, 12, 125) else 0.10
        s += 0.20 if self._bent(lm, 13, 14, 16, 125) else 0.10
        s += 0.25 if self._extended(lm, 17, 18, 20, 142) else 0.12
        s += 0.15 if self._thumb_out(lm, hs) else 0.0
        return min(s, 1.0)

    def _s_simhamukha(self, lm, hs):
        """Middle+Ring bent to thumb. Index+Pinky extended."""
        if not self._bent(lm, 9, 10, 12, 132): return 0.0
        if not self._bent(lm, 13, 14, 16, 132): return 0.0
        if not self._extended(lm, 5, 6, 8, 132): return 0.0
        if not self._extended(lm, 17, 18, 20, 132): return 0.0
        mid_near = self._dist(lm[12], lm[4]) / hs < 0.38
        ring_near = self._dist(lm[16], lm[4]) / hs < 0.38
        if not (mid_near or ring_near): return 0.0
        s = 0.0
        s += 0.18 if mid_near else 0.05
        s += 0.18 if ring_near else 0.05
        s += 0.25 if self._extended(lm, 5, 6, 8, 142) else 0.12
        s += 0.25 if self._extended(lm, 17, 18, 20, 142) else 0.12
        s += 0.14 if self._bent(lm, 9, 10, 12, 122) else 0.07
        return min(s, 1.0)

    def _s_kangula(self, lm, hs):
        """Padmakosha with ring bent inward. Others half-bent."""
        if not self._bent(lm, 13, 14, 16, 128): return 0.0
        if self._extended(lm, 5, 6, 8, 156): return 0.0
        if self._extended(lm, 9, 10, 12, 156): return 0.0
        if self._extended(lm, 17, 18, 20, 156): return 0.0
        angles = [
            self._fangle(lm, 5, 6, 8),
            self._fangle(lm, 9, 10, 12),
            self._fangle(lm, 17, 18, 20),
        ]
        hb = sum(1 for a in angles if 100 <= a <= 156)
        if hb < 2: return 0.0
        s = 0.0
        s += 0.42 if self._bent(lm, 13, 14, 16, 118) else 0.22
        s += 0.15 * hb
        if self._dist(lm[16], lm[4]) / hs < 0.42: s += 0.13
        return min(s, 1.0)

    def _s_chatura(self, lm, hs):
        """Pinky UP only. Index+Middle+Ring bent. Thumb near ring base."""
        if not self._extended(lm, 17, 18, 20, 135): return 0.0
        if not self._bent(lm, 5, 6, 8, 135): return 0.0
        if not self._bent(lm, 9, 10, 12, 135): return 0.0
        if not self._bent(lm, 13, 14, 16, 135): return 0.0
        s = 0.0
        s += 0.40 if self._extended(lm, 17, 18, 20, 145) else 0.20
        s += 0.20 if self._bent(lm, 5, 6, 8, 125) else 0.10
        s += 0.20 if self._bent(lm, 9, 10, 12, 125) else 0.10
        tb = self._dist(lm[4], lm[13]) / hs < 0.38
        s += 0.15 if tb else 0.05
        s += 0.05 if self._bent(lm, 13, 14, 16, 125) else 0.02
        return min(s, 1.0)

    def _s_bhramara(self, lm, hs):
        """Index curled to base. Middle+Thumb touch. Ring+Pinky out."""
        # Hard gate: middle+thumb must touch
        if not self._touching(lm, 4, 12, hs, 0.22): return 0.0
        # Hard gate: ring and pinky must be extended
        if not self._extended(lm, 13, 14, 16, 130): return 0.0
        if not self._extended(lm, 17, 18, 20, 130): return 0.0
        # Hard gate: index must be curled
        if self._dist(lm[8], lm[5]) / hs > 0.32: return 0.0
        s = 0.0
        cd = self._dist(lm[8], lm[5]) / hs
        if cd < 0.20: s += 0.30
        elif cd < 0.32: s += 0.18
        s += 0.30 if self._touching(lm, 4, 12, hs, 0.16) else 0.15
        s += 0.20 if self._extended(lm, 13, 14, 16, 140) else 0.10
        s += 0.20 if self._extended(lm, 17, 18, 20, 140) else 0.10
        return min(s, 1.0)

    def _s_hamsasya(self, lm, hs):
        """Swan beak: medium pinch + index less curved.
        Index angle > 120, pinch 0.12-0.20.
        Middle+ring+pinky all clearly extended."""
        ia = self._fangle(lm, 5, 6, 8)
        pinch = self._dist(lm[4], lm[8]) / hs
        # Hard gate: medium pinch — not too tight not too loose
        if pinch > 0.22: return 0.0
        if pinch < 0.10: return 0.0
        # Hard gate: index less curved than Shukatunda
        if ia < 118: return 0.0
        if ia > 158: return 0.0
        # Hard gate: middle ring pinky all extended
        if not self._extended(lm, 9, 10, 12, 135): return 0.0
        if not self._extended(lm, 13, 14, 16, 135): return 0.0
        if not self._extended(lm, 17, 18, 20, 135): return 0.0
        s = 0.0
        s += 0.40
        s += 0.10 if pinch < 0.16 else 0.05
        s += 0.20 if self._extended(lm,9,10,12,148) else 0.10
        s += 0.18 if self._extended(lm,13,14,16,148) else 0.09
        s += 0.12 if self._extended(lm,17,18,20,148) else 0.06
        return min(s, 1.0)

    def _s_hamsapaksha(self, lm, hs):
        if not self._bent(lm, 5, 6, 8, 138): return 0.0
        if not self._bent(lm, 9, 10, 12, 138): return 0.0
        if not self._bent(lm, 13, 14, 16, 138): return 0.0
        if not self._extended(lm, 17, 18, 20, 138): return 0.0
        if self._thumb_out(lm, hs): return 0.0
        if lm[20][1] > lm[17][1]: return 0.0
        s = 0.0
        s += 0.22 if self._bent(lm, 5, 6, 8, 125) else 0.10
        s += 0.22 if self._bent(lm, 9, 10, 12, 125) else 0.10
        s += 0.18 if self._bent(lm, 13, 14, 16, 125) else 0.09
        s += 0.38 if self._extended(lm, 17, 18, 20, 148) else 0.18
        return min(s, 1.0)

    def _s_sandamsha(self, lm, hs):
        """Pincers: All 5 tips closing together.
        Slightly more open than Mukula."""
        tips = [lm[4], lm[8], lm[12], lm[16], lm[20]]
        cx = sum(t[0] for t in tips) / 5
        cy = sum(t[1] for t in tips) / 5
        dists = [self._dist(t, (cx,cy,0)) / hs for t in tips]
        max_d = max(dists)
        avg_d = sum(dists) / 5
        # Must be tighter than open hand but looser than Mukula
        if max_d > 0.25: return 0.0
        if avg_d < 0.04: return 0.0  # too tight = Mukula
        if avg_d < 0.10: return 1.0
        if avg_d < 0.16: return 0.75
        if avg_d < 0.22: return 0.50
        return 0.0

    def _s_mukula(self, lm, hs):
        """All 5 tips converging to one point."""
        tips = [lm[4], lm[8], lm[12], lm[16], lm[20]]
        cx = sum(t[0] for t in tips)/5
        cy = sum(t[1] for t in tips)/5
        dists = [self._dist(t, (cx,cy,0))/hs for t in tips]
        if max(dists) > 0.15: return 0.0
        avg = sum(dists)/5
        if avg < 0.04: return 1.0
        if avg < 0.07: return 0.85
        if avg < 0.11: return 0.65
        return 0.0

    def _s_tamrachuda(self, lm, hs):
        """Middle+Thumb crossed/touching. Index bent/curved.
        Ring+Pinky bent at palm. Like Mukula with index open."""
        # Hard gate: middle must be touching/near thumb
        if not self._touching(lm, 4, 12, hs, 0.20): return 0.0
        # Hard gate: ring and pinky must be bent
        if not self._bent(lm, 13, 14, 16, 128): return 0.0
        if not self._bent(lm, 17, 18, 20, 128): return 0.0
        # Index must be bent/curved (not straight)
        ia = self._fangle(lm, 5, 6, 8)
        if ia > 158: return 0.0
        s = 0.0
        s += 0.45 if self._touching(lm, 4, 12, hs, 0.15) else 0.25
        if ia < 155: s += 0.20
        s += 0.18 if self._bent(lm, 13, 14, 16, 118) else 0.09
        s += 0.17 if self._bent(lm, 17, 18, 20, 118) else 0.09
        return min(s, 1.0)

    def _s_trishula(self, lm, hs):
        """Trident: Index+Middle+Ring spread open.
        Thumb+Pinky bent and joined together."""
        # Hard gate: index middle ring must be extended
        if not self._extended(lm, 5, 6, 8, 135): return 0.0
        if not self._extended(lm, 9, 10, 12, 135): return 0.0
        if not self._extended(lm, 13, 14, 16, 135): return 0.0
        # Hard gate: pinky must be bent
        if not self._bent(lm, 17, 18, 20, 130): return 0.0
        # Hard gate: thumb must be bent/tucked
        if not self._thumb_tucked(lm, hs): return 0.0
        # Hard gate: three fingers must be spread
        if self._spread(lm, 5, 13, hs) < 0.25: return 0.0
        s = 0.0
        s += 0.20 if self._extended(lm, 5, 6, 8, 145) else 0.12
        s += 0.20 if self._extended(lm, 9, 10, 12, 145) else 0.12
        s += 0.20 if self._extended(lm, 13, 14, 16, 145) else 0.12
        s += 0.20 if self._bent(lm, 17, 18, 20, 118) else 0.10
        s += 0.10 if self._thumb_tucked(lm, hs) else 0.05
        sp = self._spread(lm, 5, 13, hs)
        if sp > 0.38: s += 0.10
        elif sp > 0.25: s += 0.05
        return min(s, 1.0)

    # ─────────────────────────────────────────
    # DISPATCH
    # ─────────────────────────────────────────

    SCORERS = {
        'Pataka':       '_s_pataka',
        'Alapadma':     '_s_alapadma',
        'Ardhachandra': '_s_ardhachandra',
        'Tripataka':    '_s_tripataka',
        'Ardhapataka':  '_s_ardhapataka',
        'Kartarimukha': '_s_kartarimukha',
        'Mayura':       '_s_mayura',
        'Arala':        '_s_arala',
        'Shukatunda':   '_s_shukatunda',
        'Mushti':       '_s_mushti',
        'Shikhara':     '_s_shikhara',
        'Kapittha':     '_s_kapittha',
        'Katakamukha':  '_s_katakamukha',
        'Suchi':        '_s_suchi',
        'Chandrakala':  '_s_chandrakala',
        'Padmakosha':   '_s_padmakosha',
        'Sarpashirsha': '_s_sarpashirsha',
        'Mrigashirsha': '_s_mrigashirsha',
        'Simhamukha':   '_s_simhamukha',
        'Kangula':      '_s_kangula',
        'Chatura':      '_s_chatura',
        'Bhramara':     '_s_bhramara',
        'Hamsasya':     '_s_hamsasya',
        'Hamsapaksha':  '_s_hamsapaksha',
        'Sandamsha':    '_s_sandamsha',
        'Mukula':       '_s_mukula',
        'Tamrachuda':   '_s_tamrachuda',
        'Trishula':     '_s_trishula',
    }

    # ─────────────────────────────────────────
    # TIEBREAKERS
    # ─────────────────────────────────────────

    def _tiebreak(self, scores, lm, hs):

        def zero(name):
            if name in scores: scores[name] = 0.0

        # Index family tiebreakers
        # Resolve based on pinch distance + index angle
        pinch_tb = self._dist(lm[4], lm[8]) / hs
        ia_tb    = self._fangle(lm, 5, 6, 8)

        # Arala wins if pinch is loose
        if scores.get('Arala',0) > 0.40:
            if pinch_tb >= 0.15:
                scores['Shukatunda']  = 0.0
                scores['Hamsasya']    = 0.0
                scores['Katakamukha'] = 0.0
            else:
                scores['Arala'] = 0.0

        # Shukatunda wins if very tight + very curved
        if (scores.get('Shukatunda',0) > 0.40 and
                pinch_tb < 0.12 and ia_tb < 125):
            scores['Arala']       = 0.0
            scores['Hamsasya']    = 0.0
            scores['Katakamukha'] = 0.0

        # Hamsasya vs Katakamukha
        # Hamsasya: pinch 0.12-0.20, idx > 118
        # Katakamukha: pinch < 0.13, idx < 122
        if (scores.get('Hamsasya',0) > 0.40 and
                scores.get('Katakamukha',0) > 0.40):
            if pinch_tb > 0.13 or ia_tb > 122:
                scores['Katakamukha'] = 0.0
            else:
                scores['Hamsasya'] = 0.0

        # Kartarimukha vs Simhamukha — thumb and spread
        if scores.get('Kartarimukha',0)>0.50 and scores.get('Simhamukha',0)>0.50:
            mid_near = self._dist(lm[12], lm[4]) / hs < 0.38
            ring_near = self._dist(lm[16], lm[4]) / hs < 0.38
            if mid_near or ring_near:
                scores['Kartarimukha'] = 0.0
            else:
                scores['Simhamukha'] = 0.0

        # Chandrakala vs Suchi — thumb is decisive
        if scores.get('Chandrakala',0)>0.50 and scores.get('Suchi',0)>0.50:
            if self._thumb_out(lm, hs):
                scores['Suchi'] = 0.0
            else:
                scores['Chandrakala'] = 0.0

        # Hamsasya vs Katakamukha
        if (scores.get('Hamsasya',0)>0.50 and
                scores.get('Katakamukha',0)>0.50):
            if (self._bent(lm,13,14,16,128) and
                    self._bent(lm,17,18,20,128)):
                scores['Katakamukha'] = 0.0
            else:
                scores['Hamsasya'] = 0.0

        # Shukatunda vs Arala — ring bent decides
        if scores.get('Shukatunda',0)>0.50 and scores.get('Arala',0)>0.50:
            if self._bent(lm, 13, 14, 16, 140):
                scores['Arala'] = 0.0
            else:
                scores['Shukatunda'] = 0.0

        # Trishula vs Tripataka — spread decides
        if scores.get('Trishula',0)>0.50 and scores.get('Tripataka',0)>0.50:
            if self._spread(lm, 5, 17, hs) > 0.30:
                scores['Tripataka'] = 0.0
            else:
                scores['Trishula'] = 0.0

        # Tripataka vs Pataka — ring angle decides
        if scores.get('Tripataka',0)>0.50 and scores.get('Pataka',0)>0.50:
            if self._fangle(lm, 13, 14, 16) < 158:
                scores['Pataka'] = 0.0
            else:
                scores['Tripataka'] = 0.0

        # Pataka vs Ardhachandra — thumb state
        if scores.get('Pataka',0)>0.50 and scores.get('Ardhachandra',0)>0.50:
            if self._thumb_out(lm,hs): zero('Pataka')
            else: zero('Ardhachandra')

        # Alapadma vs Ardhachandra — spread
        if scores.get('Alapadma',0)>0.50 and scores.get('Ardhachandra',0)>0.50:
            avg = sum([self._spread(lm,5,9,hs),
                       self._spread(lm,9,13,hs),
                       self._spread(lm,13,17,hs)])/3
            if avg > 0.20: zero('Ardhachandra')
            else: zero('Alapadma')

        # Alapadma vs Pataka — spread
        if scores.get('Alapadma',0)>0.55 and scores.get('Pataka',0)>0.55:
            zero('Pataka')

        # Sarpashirsha vs Pataka — droop
        if scores.get('Sarpashirsha',0)>0.50 and scores.get('Pataka',0)>0.50:
            droop = sum([lm[8][1]>lm[7][1], lm[12][1]>lm[11][1],
                         lm[16][1]>lm[15][1], lm[20][1]>lm[19][1]])
            if droop >= 3: zero('Pataka')
            else: zero('Sarpashirsha')

        # Mrigashirsha vs Hamsapaksha — thumb
        if scores.get('Mrigashirsha',0)>0.50 and scores.get('Hamsapaksha',0)>0.50:
            if self._thumb_out(lm,hs): zero('Hamsapaksha')
            else: zero('Mrigashirsha')

        # Sarpashirsha vs Hamsapaksha — deep droop decides
        if scores.get('Sarpashirsha',0)>0.50 and scores.get('Hamsapaksha',0)>0.50:
            deep_droop = sum([
                lm[8][1]  > lm[5][1],
                lm[12][1] > lm[9][1],
                lm[16][1] > lm[13][1],
            ])
            if deep_droop >= 2: scores['Hamsapaksha'] = 0.0
            else: scores['Sarpashirsha'] = 0.0

        # Shikhara vs Mushti — thumb up
        if scores.get('Shikhara',0)>0.40 or scores.get('Mushti',0)>0.40:
            if lm[4][1] < lm[2][1] and self._thumb_out(lm,hs):
                zero('Mushti')
            elif self._thumb_tucked(lm,hs):
                zero('Shikhara')

        # Tamrachuda vs Shikhara — index extended
        if scores.get('Tamrachuda',0)>0.50 and scores.get('Shikhara',0)>0.50:
            if self._extended(lm,5,6,8,145): zero('Shikhara')
            else: zero('Tamrachuda')

        # Tripataka vs Shukatunda — ring angle and index curve decides
        if scores.get('Tripataka',0)>0.50 and scores.get('Shukatunda',0)>0.30:
            ring_angle = self._fangle(lm, 13, 14, 16)
            idx_angle = self._fangle(lm, 5, 6, 8)
            # If ring is clearly bent AND index is curved, favor Shukatunda
            if ring_angle < 140 and idx_angle < 150:
                scores['Tripataka'] = 0.0
            else:
                scores['Shukatunda'] = 0.0

        # Arala vs Pataka — index angle decides
        if scores.get('Arala',0)>0.40 and scores.get('Pataka',0)>0.40:
            if self._fangle(lm, 5, 6, 8) < 158:
                scores['Pataka'] = 0.0
            else:
                scores['Arala'] = 0.0

        # Shukatunda vs Arala — ring bent decides
        if scores.get('Shukatunda',0)>0.50 and scores.get('Arala',0)>0.50:
            if self._bent(lm, 13, 14, 16, 132):
                scores['Arala'] = 0.0
            else:
                scores['Shukatunda'] = 0.0

        # Ardhapataka vs Trishula — middle finger state
        if scores.get('Ardhapataka',0)>0.50 and scores.get('Trishula',0)>0.50:
            if self._bent(lm, 17, 18, 20, 130):
                scores['Trishula'] = 0.0
            else:
                scores['Ardhapataka'] = 0.0

        # Chatura vs Tripataka — index bent
        if scores.get('Chatura',0)>0.50 and scores.get('Tripataka',0)>0.50:
            if self._bent(lm,5,6,8,135): zero('Tripataka')
            else: zero('Chatura')

        # Kartarimukha vs Simhamukha — thumb
        if scores.get('Kartarimukha',0)>0.50 and scores.get('Simhamukha',0)>0.50:
            if self._thumb_out(lm,hs): zero('Kartarimukha')
            else: zero('Simhamukha')

        # Chandrakala vs Suchi — thumb
        if scores.get('Chandrakala',0)>0.50 and scores.get('Suchi',0)>0.50:
            if self._thumb_out(lm,hs): zero('Suchi')
            else: zero('Chandrakala')

        # Chandrakala vs Shikhara — index state
        if scores.get('Chandrakala',0)>0.50 and scores.get('Shikhara',0)>0.50:
            if self._extended(lm, 5, 6, 8, 140):
                scores['Shikhara'] = 0.0
            else:
                scores['Chandrakala'] = 0.0

        # Hamsasya vs Katakamukha — ring+pinky
        if scores.get('Hamsasya',0)>0.40 and scores.get('Katakamukha',0)>0.40:
            ring_ext = self._extended(lm, 13, 14, 16, 135)
            pinky_ext = self._extended(lm, 17, 18, 20, 135)
            pinch_tb = self._dist(lm[4], lm[8]) / hs
            ia_tb = self._fangle(lm, 5, 6, 8)
            if ring_ext and pinky_ext:
                # Ring+pinky extended = Hamsasya
                zero('Katakamukha')
            elif pinch_tb < 0.13 and ia_tb < 122:
                # Very tight pinch + curved index = Katakamukha
                zero('Hamsasya')
            else:
                zero('Katakamukha')

        # Bhramara vs Katakamukha — index curl
        if scores.get('Bhramara',0)>0.50 and scores.get('Katakamukha',0)>0.50:
            if self._dist(lm[8],lm[5])/hs < 0.28:
                zero('Katakamukha')
            else:
                zero('Bhramara')

        # Padmakosha vs Alapadma
        if scores.get('Padmakosha',0)>0.50 and scores.get('Alapadma',0)>0.50:
            if any([not self._extended(lm,5,6,8,155),
                    not self._extended(lm,9,10,12,155),
                    not self._extended(lm,13,14,16,155),
                    not self._extended(lm,17,18,20,155)]):
                zero('Alapadma')
            else:
                zero('Padmakosha')

        # Kangula vs Padmakosha
        if scores.get('Kangula',0)>0.50 and scores.get('Padmakosha',0)>0.50:
            if self._bent(lm,13,14,16,125): zero('Padmakosha')
            else: zero('Kangula')

        # Sandamsha vs Mukula — tightness decides
        if scores.get('Sandamsha',0)>0.50 and scores.get('Mukula',0)>0.50:
            tips = [lm[4],lm[8],lm[12],lm[16],lm[20]]
            cx = sum(t[0] for t in tips)/5
            cy = sum(t[1] for t in tips)/5
            avg = sum(self._dist(t,(cx,cy,0)) for t in tips)/(5*hs)
            if avg < 0.06:
                scores['Sandamsha'] = 0.0  # too tight = Mukula
            else:
                scores['Mukula'] = 0.0

        # Bhramara vs Tamrachuda — index curl decides
        if scores.get('Bhramara',0)>0.50 and scores.get('Tamrachuda',0)>0.50:
            if self._dist(lm[8], lm[5]) / hs < 0.30:
                scores['Tamrachuda'] = 0.0
            else:
                scores['Bhramara'] = 0.0

        # Kapittha vs Shikhara — index near thumb decides
        if scores.get('Kapittha',0)>0.50 and scores.get('Shikhara',0)>0.50:
            if self._dist(lm[8], lm[4]) / hs < 0.25:
                scores['Shikhara'] = 0.0
            else:
                scores['Kapittha'] = 0.0

        # Padmakosha vs Shukatunda — uniformity decides
        if scores.get('Padmakosha',0)>0.50 and scores.get('Shukatunda',0)>0.50:
            angles = [
                self._fangle(lm,5,6,8),
                self._fangle(lm,9,10,12),
                self._fangle(lm,13,14,16),
                self._fangle(lm,17,18,20),
            ]
            half = sum(1 for a in angles if 100 <= a <= 155)
            if half >= 3:
                scores['Shukatunda'] = 0.0
            else:
                scores['Padmakosha'] = 0.0

        return scores

    # ─────────────────────────────────────────
    # TEMPORAL SMOOTHING
    # ─────────────────────────────────────────

    def _smooth(self, raw, handedness):
        if handedness == 'Left':
            hist = self.left_history
            conf = self.left_confirmed
        else:
            hist = self.right_history
            conf = self.right_confirmed

        # Don't let Unknown flush history quickly
        if raw == 'Unknown':
            unk_count = sum(1 for x in hist if x == 'Unknown')
            # Require 6 consecutive Unknowns before releasing
            if unk_count < 6 and hist:
                raw = hist[-1]

        hist.append(raw)

        if conf is not None:
            if hist.count(conf) >= self.RELEASE_THRESH:
                pass  # keep confirmed
            else:
                conf = None

        if conf is None:
            cand = Counter(hist).most_common(1)[0][0]
            if hist.count(cand) >= self.CONFIRM_THRESH:
                conf = cand

        if handedness == 'Left':
            self.left_confirmed = conf
        else:
            self.right_confirmed = conf

        return conf if conf else Counter(hist).most_common(1)[0][0]

    # ─────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────

    def recognize_single(self, landmarks, handedness, debug=False):
        if not landmarks or len(landmarks) != 21:
            return ('Unknown', 0.0, {}) if debug else ('Unknown', 0.0)

        lm = self._normalize(landmarks, handedness)
        hs = self._hand_size(lm)

        groups, flags = self._get_groups(lm, hs)

        candidates = set()
        for g in groups:
            candidates.update(self.GROUPS.get(g, []))

        scores = {}
        for name in candidates:
            fn_name = self.SCORERS.get(name)
            if fn_name:
                scores[name] = getattr(self, fn_name)(lm, hs)

        # Smooth display buffer
        dbuf = (self.display_scores_right if handedness=='Right'
                else self.display_scores_left)
        for name, sc in scores.items():
            dbuf[name] = (self.DISPLAY_ALPHA * sc +
                          (1-self.DISPLAY_ALPHA) * dbuf.get(name, sc))
        for name in list(dbuf.keys()):
            if name not in scores:
                dbuf[name] *= 0.85
                if dbuf[name] < 0.05:
                    del dbuf[name]

        if scores:
            scores = self._tiebreak(scores, lm, hs)
            best = max(scores, key=scores.get)
            best_score = scores[best]
            raw = best if best_score >= 0.45 else 'Unknown'
        else:
            raw = 'Unknown'
            best_score = 0.0

        final = self._smooth(raw, handedness)

        if debug:
            top5 = sorted(dbuf.items(), key=lambda x:x[1], reverse=True)[:5]
            return (final, best_score, {
                'stage1_groups': groups,
                'flags': flags,
                'top5': top5,
                'finger_angles': self._finger_angles(lm),
                'thumb_state': 'tucked' if flags['thumb_tuck'] else 'extended',
                'hand_size': hs,
                'candidates': list(candidates),
            })

        return (final, best_score)

    def recognize_two_hand(self, hand1, hand2):
        r1 = self.recognize_single(hand1[0], hand1[1])
        r2 = self.recognize_single(hand2[0], hand2[1])
        return r1 if r1[1] >= r2[1] else r2

    def get_debug_info(self):
        return {}
