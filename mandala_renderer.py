import cv2
import numpy as np
import math
import time


class MandalaRenderer:

    # Mudra category → mandala pattern
    MUDRA_CATEGORY = {
        # Auspicious
        'Pataka':       'lotus',
        'Tripataka':    'lotus',
        'Ardhapataka':  'lotus',
        'Ardhachandra': 'lotus',
        'Alapadma':     'lotus',
        # Power
        'Mushti':       'yantra',
        'Shikhara':     'yantra',
        'Kartarimukha': 'yantra',
        'Trishula':     'yantra',
        'Tamrachuda':   'yantra',
        # Grace
        'Arala':        'floral',
        'Shukatunda':   'floral',
        'Chandrakala':  'floral',
        'Hamsasya':     'floral',
        'Kapittha':     'floral',
        'Katakamukha':  'floral',
        # Nature
        'Mayura':       'spiral',
        'Mrigashirsha': 'spiral',
        'Simhamukha':   'spiral',
        'Sarpashirsha': 'spiral',
        'Hamsapaksha':  'spiral',
        # Divine
        'Suchi':        'star',
        'Padmakosha':   'star',
        'Kangula':      'star',
        'Chatura':      'star',
        'Bhramara':     'star',
    }

    # Pattern → color palette (BGR)
    PATTERN_COLORS = {
        'lotus':  [(30, 180, 255), (60, 140, 220), (90, 100, 200)],
        'yantra': [(20,  60, 200), (10,  30, 160), (30,  80, 220)],
        'floral': [(60, 20, 120), (40, 10, 90), (80, 30, 150)],
        'spiral': [(50, 185,  80), (30, 150,  60), (70, 200, 100)],
        'star':   [(255, 230, 180), (240, 200, 140), (255, 245, 200)],
        'default':[(40, 160, 220), (20, 120, 180), (60, 180, 240)],
    }

    def __init__(self, width=640, height=480):
        self.w = width
        self.h = height
        self.cx = width // 2
        self.cy = height // 2
        self.mandala_scale     = 0.8   # starts small, grows while spinning
        self.rotation    = 0.0      # current rotation angle
        self.spin_speed  = 0.0      # driven by dancer spin
        self.base_speed  = 0.008    # ambient slow rotation
        self.breathe_t   = 0.0      # breathing phase
        self.current_pattern = 'floral'
        self.current_colors  = [(100, 60, 180), (70, 40, 140), (120, 80, 200)]

    def set_mudra(self, mudra):
        # Always use floral pattern regardless of mudra
        self.current_pattern = 'floral'
        self.current_colors  = [(100, 60, 180), (70, 40, 140), (120, 80, 200)]

    def update(self, dt=0.033, movement_energy=0.0):
        self.rotation  += self.base_speed + self.spin_speed
        # Breathing speed tied to movement energy
        # Still = slow breath, moving = faster pulse
        breath_speed = 0.6 + movement_energy * 1.2
        self.breathe_t += dt * breath_speed
        # Decay spin speed back to zero
        self.spin_speed *= 0.96

    def add_spin(self, speed):
        """Called when dancer spin detected."""
        self.spin_speed = speed

    # ── Drawing helpers ───────────────────────────

    def _ngon(self, img, cx, cy, r, n, angle_offset,
              color, thickness, alpha):
        pts = []
        for i in range(n):
            a = angle_offset + i * 2 * math.pi / n
            pts.append((int(cx + math.cos(a) * r),
                        int(cy + math.sin(a) * r)))
        ov = img.copy()
        cv2.polylines(ov,
            [np.array(pts, np.int32).reshape(-1,1,2)],
            True, color, thickness)
        cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)

    def _radial_lines(self, img, cx, cy, r1, r2, n,
                      angle_offset, color, thickness, alpha):
        ov = img.copy()
        for i in range(n):
            a = angle_offset + i * 2 * math.pi / n
            x1 = int(cx + math.cos(a) * r1)
            y1 = int(cy + math.sin(a) * r1)
            x2 = int(cx + math.cos(a) * r2)
            y2 = int(cy + math.sin(a) * r2)
            cv2.line(ov, (x1,y1), (x2,y2), color, thickness)
        cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)

    def _petal(self, img, cx, cy, r, angle,
               width_r, color, alpha):
        tip_x = int(cx + math.cos(angle) * r)
        tip_y = int(cy + math.sin(angle) * r)
        perp  = angle + math.pi/2
        lx = int(cx + math.cos(perp)*width_r
                 + math.cos(angle)*r*0.45)
        ly = int(cy + math.sin(perp)*width_r
                 + math.sin(angle)*r*0.45)
        rx = int(cx - math.cos(perp)*width_r
                 + math.cos(angle)*r*0.45)
        ry = int(cy - math.sin(perp)*width_r
                 + math.sin(angle)*r*0.45)
        pts = np.array([[cx,cy],[lx,ly],
                        [tip_x,tip_y],[rx,ry]],
                       np.int32).reshape(-1,1,2)
        ov = img.copy()
        cv2.fillPoly(ov, [pts], color)
        cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)
        cv2.polylines(img, [pts], True, color, 1)

    # ── Pattern drawers ───────────────────────────

    def _draw_lotus(self, img, rot, breathe, colors):
        c1, c2, c3 = colors
        bs = 1.0 + 0.04 * math.sin(breathe)
        R  = int(min(self.w, self.h) * 0.42 * bs)
        # Outer petals
        for i in range(12):
            a = rot + i * 2*math.pi/12
            self._petal(img, self.cx, self.cy,
                        R, a, int(R*0.18), c1, 0.12)
        # Mid petals
        for i in range(8):
            a = rot*1.3 + i * 2*math.pi/8
            self._petal(img, self.cx, self.cy,
                        int(R*0.68), a, int(R*0.14), c2, 0.15)
        # Inner petals
        for i in range(6):
            a = rot*1.7 + i * 2*math.pi/6
            self._petal(img, self.cx, self.cy,
                        int(R*0.40), a, int(R*0.10), c3, 0.18)
        # Concentric rings
        for r_mult, alpha in [(1.0,0.12),(0.70,0.14),(0.42,0.18)]:
            self._ngon(img, self.cx, self.cy,
                       int(R*r_mult), 12, rot, c1, 1, alpha)
        # Radial spokes
        self._radial_lines(img, self.cx, self.cy,
                           int(R*0.15), R, 12, rot, c2, 1, 0.14)

    def _draw_yantra(self, img, rot, breathe, colors):
        c1, c2, c3 = colors
        bs = 1.0 + 0.03 * math.sin(breathe)
        R  = int(min(self.w, self.h) * 0.40 * bs)
        # Outer circle
        self._ngon(img, self.cx, self.cy,
                   int(R*1.08), 36, rot, c1, 1, 0.50)
        # 4 overlapping triangles
        for i in range(4):
            self._ngon(img, self.cx, self.cy,
                       R, 3,
                       rot + i*math.pi/6,
                       c1, 1, 0.55)
        # 3 squares rotated
        for i in range(3):
            self._ngon(img, self.cx, self.cy,
                       int(R*0.75), 4,
                       rot*1.5 + i*math.pi/6,
                       c2, 1, 0.55)
        # Hexagons
        for r_mult in [0.55, 0.38]:
            self._ngon(img, self.cx, self.cy,
                       int(R*r_mult), 6,
                       rot*2, c3, 1, 0.60)
        # Inner square
        self._ngon(img, self.cx, self.cy,
                   int(R*0.25), 4, rot*3, c1, 1, 0.65)
        # Dense radial lines — 2 sets
        self._radial_lines(img, self.cx, self.cy,
                           int(R*0.10), R, 12,
                           rot, c1, 1, 0.50)
        self._radial_lines(img, self.cx, self.cy,
                           int(R*0.10), int(R*0.75), 12,
                           rot + math.pi/12, c2, 1, 0.45)
        # Concentric rings
        for r_mult, alpha in [(0.90,0.45),(0.68,0.50),(0.45,0.55),(0.25,0.60)]:
            self._ngon(img, self.cx, self.cy,
                       int(R*r_mult), 36, rot, c2, 1, alpha)
        # Center dot
        ov = img.copy()
        cv2.circle(ov, (self.cx, self.cy), int(R*0.06), c3, -1)
        cv2.addWeighted(ov, 0.80, img, 0.20, 0, img)

    def _draw_floral(self, img, rot, breathe, colors):
        c1, c2, c3 = colors
        bs = 1.0 + 0.05 * math.sin(breathe)
        R  = int(min(self.w, self.h) * 0.41 * bs)
        # Large outer petals
        for i in range(16):
            a = rot + i * 2*math.pi/16
            self._petal(img, self.cx, self.cy,
                        R, a, int(R*0.12), c1, 0.10)
        # Medium petals offset
        for i in range(12):
            a = rot*1.2 + i * 2*math.pi/12 + math.pi/12
            self._petal(img, self.cx, self.cy,
                        int(R*0.65), a, int(R*0.10), c2, 0.13)
        # Small inner petals
        for i in range(8):
            a = rot*1.5 + i * 2*math.pi/8
            self._petal(img, self.cx, self.cy,
                        int(R*0.38), a, int(R*0.08), c3, 0.16)
        # Rings
        for r_mult, alpha in [(1.0,0.10),(0.65,0.13),(0.38,0.16)]:
            self._ngon(img, self.cx, self.cy,
                       int(R*r_mult), 16, rot, c1, 1, alpha)

    def _draw_spiral(self, img, rot, breathe, colors):
        c1, c2, c3 = colors
        bs = 1.0 + 0.04 * math.sin(breathe)
        R  = int(min(self.w, self.h) * 0.42 * bs)
        # 6 Archimedean spirals
        for s in range(6):
            phase = s * 2*math.pi/6
            pts   = []
            for t in range(200):
                frac  = t/199
                r_cur = int(R * 0.05 + R * 0.95 * frac)
                a     = rot + phase + frac * 5*math.pi
                pts.append((
                    int(self.cx + math.cos(a)*r_cur),
                    int(self.cy + math.sin(a)*r_cur),
                ))
            ov = img.copy()
            for i in range(1, len(pts)):
                cv2.line(ov, pts[i-1], pts[i], c1, 1)
            cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)
        # Dense concentric rings
        for r_mult in [0.20, 0.35, 0.50, 0.65, 0.80, 0.95, 1.05]:
            self._ngon(img, self.cx, self.cy,
                       int(R*r_mult), 36, rot, c2, 1, 0.45)
        # Radial spokes — 2 sets
        self._radial_lines(img, self.cx, self.cy,
                           0, R, 16, rot, c3, 1, 0.45)
        self._radial_lines(img, self.cx, self.cy,
                           int(R*0.35), R, 16,
                           rot + math.pi/16, c1, 1, 0.35)
        # Petal layer
        for i in range(12):
            a = rot*1.2 + i * 2*math.pi/12
            self._petal(img, self.cx, self.cy,
                        int(R*0.55), a, int(R*0.08), c2, 0.35)
        # Center circle
        ov = img.copy()
        cv2.circle(ov, (self.cx, self.cy), int(R*0.10), c3, -1)
        cv2.addWeighted(ov, 0.70, img, 0.30, 0, img)

    def _draw_star(self, img, rot, breathe, colors):
        c1, c2, c3 = colors
        bs = 1.0 + 0.04 * math.sin(breathe)
        R  = int(min(self.w, self.h) * 0.41 * bs)
        # Outer ring
        self._ngon(img, self.cx, self.cy,
                   int(R*1.08), 36, rot, c3, 1, 0.45)
        # Dense radial lines
        self._radial_lines(img, self.cx, self.cy,
                           0, R, 24, rot, c2, 1, 0.45)
        self._radial_lines(img, self.cx, self.cy,
                           int(R*0.40), R, 24,
                           rot + math.pi/24, c1, 1, 0.35)
        # Nested star polygons — more layers
        for n, r_mult, alpha in [
            (12, 1.00, 0.50),
            (10, 0.88, 0.52),
            (8,  0.75, 0.55),
            (7,  0.62, 0.58),
            (6,  0.50, 0.60),
            (5,  0.38, 0.63),
            (4,  0.26, 0.65),
            (3,  0.16, 0.68),
        ]:
            self._ngon(img, self.cx, self.cy,
                       int(R*r_mult), n,
                       rot + n*0.12, c1, 1, alpha)
        # Petal layer between stars
        for i in range(16):
            a = rot*1.3 + i * 2*math.pi/16
            self._petal(img, self.cx, self.cy,
                        int(R*0.70), a, int(R*0.06), c2, 0.30)
        # Concentric rings
        for r_mult in [0.85, 0.65, 0.45, 0.28]:
            self._ngon(img, self.cx, self.cy,
                       int(R*r_mult), 36, rot, c3, 1, 0.40)
        # Bright center
        ov = img.copy()
        cv2.circle(ov, (self.cx, self.cy), int(R*0.08), c1, -1)
        cv2.addWeighted(ov, 0.85, img, 0.15, 0, img)

    # ── Public API ────────────────────────────────

    def render(self, frame):
        """Draw mandala onto frame in-place."""
        breathe = self.breathe_t
        rot     = self.rotation
        colors  = self.current_colors
        self._draw_floral(frame, rot, breathe, colors)
        return frame
