import cv2
import numpy as np
import math
import time
from collections import deque


# ─────────────────────────────────────────────────────
# TRAIL SYSTEM
# ─────────────────────────────────────────────────────

class TrailSystem:
    def __init__(self, maxlen=30):
        self.points = deque(maxlen=maxlen)

    def update(self, x, y):
        self.points.append((x, y, time.time()))

    def draw(self, frame, color, thickness=2):
        pts = list(self.points)
        if len(pts) < 2:
            return
        now = time.time()
        for i in range(1, len(pts)):
            age = now - pts[i][2]
            alpha = max(0, 1.0 - age * 3.0)
            if alpha <= 0:
                continue
            c = tuple(int(ch * alpha) for ch in color)
            cv2.line(frame,
                (pts[i-1][0], pts[i-1][1]),
                (pts[i][0],   pts[i][1]),
                c, thickness)


# ─────────────────────────────────────────────────────
# PARTICLE SYSTEM
# ─────────────────────────────────────────────────────

class ParticleSystem:
    def __init__(self, max_particles=200):
        self.particles = []
        self.MAX = max_particles

    def spawn(self, x, y, color, behavior,
              speed_factor=1.0, count=3):
        for _ in range(count):
            if len(self.particles) >= self.MAX:
                break

            if behavior == 'drift_up':
                vx = np.random.uniform(-0.8, 0.8)
                vy = np.random.uniform(-2.0, -0.8)
            elif behavior == 'orbit':
                angle = np.random.uniform(0, 2*math.pi)
                spd = np.random.uniform(0.5, 1.5)
                vx = math.cos(angle) * spd
                vy = math.sin(angle) * spd
            elif behavior == 'bloom':
                angle = np.random.uniform(0, 2*math.pi)
                spd = np.random.uniform(1.0, 3.0)
                vx = math.cos(angle) * spd
                vy = math.sin(angle) * spd
            elif behavior == 'rise':
                vx = np.random.uniform(-0.5, 0.5)
                vy = np.random.uniform(-2.5, -1.2)
            else:
                angle = np.random.uniform(0, 2*math.pi)
                vx = math.cos(angle)
                vy = math.sin(angle)

            self.particles.append({
                'x': float(x + np.random.randint(-8, 8)),
                'y': float(y + np.random.randint(-8, 8)),
                'vx': vx * speed_factor,
                'vy': vy * speed_factor,
                'life': 1.0,
                'decay': np.random.uniform(0.015, 0.035),
                'size': np.random.randint(2, 5),
                'color': color,
            })

    def update_draw(self, frame):
        alive = []
        for p in self.particles:
            p['x']  += p['vx']
            p['y']  += p['vy']
            p['vx'] *= 0.97
            p['vy'] *= 0.97
            p['life'] -= p['decay']
            if p['life'] > 0:
                alpha = p['life']
                c = tuple(int(ch * alpha) for ch in p['color'])
                cx, cy = int(p['x']), int(p['y'])
                h, w = frame.shape[:2]
                if 0 <= cx < w and 0 <= cy < h:
                    cv2.circle(frame, (cx,cy),
                               p['size'], c, -1)
                alive.append(p)
        self.particles = alive


# ─────────────────────────────────────────────────────
# GLOW HELPER
# ─────────────────────────────────────────────────────

def draw_glow(frame, cx, cy, radius, color):
    """Soft multi-layer glow around a point."""
    for r_mult, alpha in [(2.2,0.05),(1.7,0.09),
                          (1.3,0.14),(1.0,0.22)]:
        ov = frame.copy()
        cv2.circle(ov, (cx,cy), int(radius*r_mult),
                   color, -1)
        cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)
    cv2.circle(frame, (cx,cy),
               int(radius*0.25), color, 2)


# ─────────────────────────────────────────────────────
# FINGERTIP GLOW HELPER
# ─────────────────────────────────────────────────────

def draw_fingertip_dots(frame, landmarks, color, w, h):
    """Small glow dot at each fingertip."""
    for tip_idx in [4, 8, 12, 16, 20]:
        tx = int(landmarks[tip_idx][0] * w)
        ty = int(landmarks[tip_idx][1] * h)
        cv2.circle(frame, (tx,ty), 4, color, -1)
        ov = frame.copy()
        cv2.circle(ov, (tx,ty), 9, color, -1)
        cv2.addWeighted(ov, 0.12, frame, 0.88, 0, frame)


# ─────────────────────────────────────────────────────
# GROUP 1 GEOMETRY FUNCTIONS
# ─────────────────────────────────────────────────────

def draw_pataka_rays(frame, cx, cy, radius, color, now):
    """16 bright rays shooting from palm like sunburst.
    Each ray has bright core + wider soft glow behind it.
    Rays slowly rotate. Concentric rings around palm."""

    # Outer soft glow rings (4 rings, large radius)
    for r_mult, alpha in [
        (3.5, 0.15), (2.8, 0.22),
        (2.0, 0.35), (1.4, 0.50)
    ]:
        ov = frame.copy()
        cv2.circle(ov, (cx,cy),
                   int(radius * r_mult), color, 1)
        cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)

    # 16 rays — each has wide soft ray + bright core ray
    n = 16
    for i in range(n):
        angle = (2*math.pi/n)*i + now*0.25

        # Soft wide ray (low alpha, thick line)
        ex_far = int(cx + math.cos(angle) * radius * 3.2)
        ey_far = int(cy + math.sin(angle) * radius * 3.2)
        ov = frame.copy()
        cv2.line(ov, (cx,cy), (ex_far, ey_far), color, 4)
        cv2.addWeighted(ov, 0.18, frame, 0.82, 0, frame)

        # Bright core ray (high alpha, thin line)
        ex_mid = int(cx + math.cos(angle) * radius * 2.5)
        ey_mid = int(cy + math.sin(angle) * radius * 2.5)
        ov = frame.copy()
        cv2.line(ov, (cx,cy), (ex_mid, ey_mid), color, 2)
        cv2.addWeighted(ov, 0.60, frame, 0.40, 0, frame)

    # Bright center circle
    cv2.circle(frame, (cx,cy), int(radius*0.4), color, 2)
    ov = frame.copy()
    cv2.circle(ov, (cx,cy), int(radius*0.25), color, -1)
    cv2.addWeighted(ov, 0.70, frame, 0.30, 0, frame)


def draw_tripataka_flames(frame, cx, cy, radius,
                           color, now):
    """Three elongated flame tongues above hand.
    Middle flame taller. Flames flicker with sine wave.
    Each flame is a filled pointed polygon."""

    flame_configs = [
        # (x_offset, height_mult, width_mult)
        (-int(radius*0.55), 1.6, 0.28),  # left flame
        (0,                  2.2, 0.35),  # center flame (tallest)
        ( int(radius*0.55), 1.6, 0.28),  # right flame
    ]

    for i, (x_off, h_mult, w_mult) in \
            enumerate(flame_configs):
        # Flicker using sine wave — each flame independent
        flicker = math.sin(now * 4.0 + i * 1.2) * 0.12
        h = int(radius * (h_mult + flicker) * 1.8)
        w = int(radius * w_mult)

        base_x = cx + x_off
        base_y = cy

        # Flame polygon: wide base, narrow tip
        tip_wobble = int(math.sin(now*5+i)*radius*0.08)
        pts = np.array([
            [base_x - w,     base_y],
            [base_x - w//2,  base_y - h//2],
            [base_x + tip_wobble, base_y - h],  # tip
            [base_x + w//2,  base_y - h//2],
            [base_x + w,     base_y],
        ], np.int32).reshape((-1,1,2))

        # Outer flame (low alpha, wider)
        ov = frame.copy()
        cv2.fillPoly(ov, [pts], color)
        cv2.addWeighted(ov, 0.35, frame, 0.65, 0, frame)

        # Inner bright core (higher alpha)
        core_pts = np.array([
            [base_x - w//2,      base_y],
            [base_x - w//4,      base_y - h//2],
            [base_x + tip_wobble, base_y - h + h//5],
            [base_x + w//4,      base_y - h//2],
            [base_x + w//2,      base_y],
        ], np.int32).reshape((-1,1,2))

        ov = frame.copy()
        cv2.fillPoly(ov, [core_pts], color)
        cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)

    # Base glow at palm
    ov = frame.copy()
    cv2.circle(ov, (cx,cy), int(radius*0.8), color, -1)
    cv2.addWeighted(ov, 0.25, frame, 0.75, 0, frame)


def draw_ardhapataka_pillars(frame, cx, cy, radius,
                              color, now):
    """Two sharp vertical pillars extending far upward.
    Bright architectural lines with subtle fill between.
    Clean cap connecting them at top. No curves."""

    offset  = int(radius * 0.35)
    pillar_h = int(radius * 3.5)
    top_y    = cy - pillar_h

    # Fill between pillars (subtle)
    ov = frame.copy()
    cv2.rectangle(ov,
        (cx - offset, top_y),
        (cx + offset, cy),
        color, -1)
    cv2.addWeighted(ov, 0.12, frame, 0.88, 0, frame)

    # Left pillar
    ov = frame.copy()
    cv2.line(ov, (cx-offset, cy),
                 (cx-offset, top_y), color, 3)
    cv2.addWeighted(ov, 0.80, frame, 0.20, 0, frame)

    # Right pillar
    ov = frame.copy()
    cv2.line(ov, (cx+offset, cy),
                 (cx+offset, top_y), color, 3)
    cv2.addWeighted(ov, 0.80, frame, 0.20, 0, frame)

    # Cap at top
    ov = frame.copy()
    cv2.line(ov, (cx-offset-8, top_y),
                 (cx+offset+8, top_y), color, 3)
    cv2.addWeighted(ov, 0.80, frame, 0.20, 0, frame)

    # Glow along pillar edges
    for x_off in [-(offset+4), (offset+4)]:
        ov = frame.copy()
        cv2.line(ov, (cx+x_off, cy),
                     (cx+x_off, top_y), color, 6)
        cv2.addWeighted(ov, 0.20, frame, 0.80, 0, frame)


def draw_ardhachandra_moon(frame, cx, cy, radius,
                            color, now):
    """Filled crescent moon — outer filled circle minus
    inner offset circle = real crescent shape.
    Soft blue-silver glow. Star dots near tips."""

    r_out  = int(radius * 2.2)
    r_in   = int(radius * 1.7)
    shift  = int(radius * 0.9)

    # Outer filled circle (the moon disc)
    ov = frame.copy()
    cv2.circle(ov, (cx, cy), r_out, color, -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

    # Cut inner circle (black) offset to create crescent
    # Offset to the right to create left-facing crescent
    ov = frame.copy()
    cv2.circle(ov, (cx + shift, cy), r_in, (0,0,0), -1)
    cv2.addWeighted(ov, 0.85, frame, 0.15, 0, frame)

    # Bright crescent edge outline
    cv2.circle(frame, (cx, cy), r_out, color, 2)

    # Soft outer glow around crescent
    ov = frame.copy()
    cv2.circle(ov, (cx, cy),
               int(r_out * 1.3), color, -1)
    cv2.addWeighted(ov, 0.10, frame, 0.90, 0, frame)

    # Star dots near crescent tips
    # Top tip approx at (cx - r_out*0.3, cy - r_out*0.9)
    # Bottom tip approx at (cx - r_out*0.3, cy + r_out*0.9)
    tip_x = cx - int(r_out * 0.25)
    for sign in [-1, 1]:
        tip_y = cy + sign * int(r_out * 0.88)
        cv2.circle(frame, (tip_x, tip_y), 4, color, -1)
        ov = frame.copy()
        cv2.circle(ov, (tip_x, tip_y), 10, color, -1)
        cv2.addWeighted(ov, 0.40, frame, 0.60, 0, frame)

    # Small scattered stars around crescent
    rng = np.random.RandomState(42)  # fixed seed = stable
    for _ in range(6):
        sx = cx + rng.randint(-int(r_out*1.5),
                               int(r_out*1.5))
        sy = cy + rng.randint(-int(r_out*1.5),
                               int(r_out*1.5))
        cv2.circle(frame, (sx, sy), 2, color, -1)


def draw_alapadma_lotus(frame, cx, cy, radius, color, now):
    """Low-poly geometric lotus like the reference image.
    Three distinct petal layers:
    - Outer: wide flat pale white-pink petals spread low
    - Middle: medium pink petals more upright
    - Inner: deep magenta tight petals
    - Center: bright orange-yellow stamen
    Each petal made of triangular facets for low-poly look.
    NO particles — geometry only."""

    breathe = 1.0 + 0.04 * math.sin(now * 1.0)

    def draw_faceted_petal(frame, cx, cy, angle,
                           length, base_width,
                           tip_color, mid_color,
                           base_color, alpha):
        """Low-poly petal made of 3 triangular facets.
        Creates the faceted gem/low-poly look."""

        perp = angle + math.pi/2

        # Key points
        tip_x  = int(cx + math.cos(angle)*length)
        tip_y  = int(cy + math.sin(angle)*length)
        base_x = int(cx + math.cos(angle)*length*0.05)
        base_y = int(cy + math.sin(angle)*length*0.05)

        # Side points at widest (45% from base)
        w_dist = length * 0.40
        left_x  = int(cx + math.cos(angle)*w_dist
                      + math.cos(perp)*base_width)
        left_y  = int(cy + math.sin(angle)*w_dist
                      + math.sin(perp)*base_width)
        right_x = int(cx + math.cos(angle)*w_dist
                      - math.cos(perp)*base_width)
        right_y = int(cy + math.sin(angle)*w_dist
                      - math.sin(perp)*base_width)

        # Mid ridge point (creates the facet split)
        mid_x = int(cx + math.cos(angle)*length*0.62)
        mid_y = int(cy + math.sin(angle)*length*0.62)

        # FACET 1: Left triangle (base to left to mid)
        f1 = np.array([
            [base_x, base_y],
            [left_x, left_y],
            [mid_x,  mid_y],
        ], np.int32).reshape((-1,1,2))
        ov = frame.copy()
        cv2.fillPoly(ov, [f1], mid_color)
        cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)

        # FACET 2: Right triangle (base to right to mid)
        f2 = np.array([
            [base_x,  base_y],
            [right_x, right_y],
            [mid_x,   mid_y],
        ], np.int32).reshape((-1,1,2))
        ov = frame.copy()
        # Slightly lighter for right facet
        lighter = tuple(min(255, c+30) for c in mid_color)
        cv2.fillPoly(ov, [f2], lighter)
        cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)

        # FACET 3: Tip triangle (left to tip to right via mid)
        f3 = np.array([
            [left_x,  left_y],
            [tip_x,   tip_y],
            [mid_x,   mid_y],
        ], np.int32).reshape((-1,1,2))
        ov = frame.copy()
        cv2.fillPoly(ov, [f3], tip_color)
        cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)

        f4 = np.array([
            [right_x, right_y],
            [tip_x,   tip_y],
            [mid_x,   mid_y],
        ], np.int32).reshape((-1,1,2))
        ov = frame.copy()
        lighter2 = tuple(min(255, c+20) for c in tip_color)
        cv2.fillPoly(ov, [f4], lighter2)
        cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)

        # Thin facet lines for low-poly look
        all_pts = [
            (base_x, base_y), (left_x, left_y),
            (mid_x, mid_y), (tip_x, tip_y),
            (right_x, right_y)
        ]
        edges = [
            (0,1),(1,2),(2,3),(3,4),(4,2),(2,0),(1,3),(4,0)
        ]
        for a_idx, b_idx in edges:
            cv2.line(frame,
                all_pts[a_idx], all_pts[b_idx],
                (255,255,255), 1)

    # ── LAYER 1: Outer petals ────────────────────────
    # Wide, flat, pale white-pink spread outward
    # BGR colors:
    outer_tip  = (235, 220, 255)  # pale pink-white
    outer_mid  = (210, 180, 240)  # soft pink
    outer_base = (180, 140, 210)  # medium pink

    n1 = 8
    l1 = int(radius * 2.8 * breathe)
    w1 = int(radius * 0.52)
    for i in range(n1):
        angle = (2*math.pi/n1)*i + now*0.04
        draw_faceted_petal(frame, cx, cy, angle,
                           l1, w1,
                           outer_tip, outer_mid,
                           outer_base, 0.70)

    # ── LAYER 2: Middle petals ───────────────────────
    # Medium pink, slightly more upright
    mid_tip  = (180, 100, 230)  # bright pink
    mid_mid  = (150, 70,  210)  # deep pink
    mid_base = (120, 50,  190)  # magenta

    n2 = 8
    l2 = int(radius * 1.9 * breathe)
    w2 = int(radius * 0.38)
    for i in range(n2):
        angle = (2*math.pi/n2)*i + \
                (math.pi/n2) + now*0.04
        draw_faceted_petal(frame, cx, cy, angle,
                           l2, w2,
                           mid_tip, mid_mid,
                           mid_base, 0.75)

    # ── LAYER 3: Inner petals ────────────────────────
    # Deep magenta, tight and tall
    inner_tip  = (140, 60,  200)  # bright magenta
    inner_mid  = (100, 30,  170)  # deep magenta
    inner_base = ( 80, 20,  150)  # dark magenta

    n3 = 6
    l3 = int(radius * 1.2 * breathe)
    w3 = int(radius * 0.26)
    for i in range(n3):
        angle = (2*math.pi/n3)*i + now*0.04
        draw_faceted_petal(frame, cx, cy, angle,
                           l3, w3,
                           inner_tip, inner_mid,
                           inner_base, 0.80)

    # ── CENTER: Orange-yellow stamen ─────────────────
    # Like the reference — bright warm orange center

    # Outer stamen ring — orange
    ov = frame.copy()
    cv2.circle(ov, (cx,cy),
               int(radius*0.45), (0, 160, 255), -1)
    cv2.addWeighted(ov, 0.85, frame, 0.15, 0, frame)

    # Inner stamen — bright yellow
    ov = frame.copy()
    cv2.circle(ov, (cx,cy),
               int(radius*0.28), (0, 220, 255), -1)
    cv2.addWeighted(ov, 0.90, frame, 0.10, 0, frame)

    # Stamen facet lines
    for i in range(8):
        angle = (2*math.pi/8)*i
        sx = int(cx + math.cos(angle)*radius*0.42)
        sy = int(cy + math.sin(angle)*radius*0.42)
        cv2.line(frame, (cx,cy), (sx,sy),
                 (0,255,255), 1)

    # Bright white hot center
    ov = frame.copy()
    cv2.circle(ov, (cx,cy),
               int(radius*0.12), (255,255,255), -1)
    cv2.addWeighted(ov, 0.95, frame, 0.05, 0, frame)


# ─────────────────────────────────────────────────────
# MUDRA THEMES — Group 1 only
# ─────────────────────────────────────────────────────

MUDRA_THEMES = {
    'Pataka': {
        'color':       (215, 245, 255),  # bright warm ivory
        'glow_color':  (200, 235, 250),
        'p_color':     (220, 248, 255),
        'geometry_fn': draw_pataka_rays,
        'p_behavior':  'drift_up',
        'p_count':     4,
        'trail_color': (200, 235, 245),
    },
    'Tripataka': {
        'color':       (30, 130, 255),   # bright amber-flame
        'glow_color':  (20, 110, 240),
        'p_color':     (50, 150, 255),
        'geometry_fn': draw_tripataka_flames,
        'p_behavior':  'rise',
        'p_count':     5,
        'trail_color': (30, 120, 245),
    },
    'Ardhapataka': {
        'color':       (235, 235, 255),  # bright cool silver
        'glow_color':  (220, 220, 245),
        'p_color':     (240, 240, 255),
        'geometry_fn': draw_ardhapataka_pillars,
        'p_behavior':  'drift_up',
        'p_count':     2,
        'trail_color': (220, 220, 240),
    },
    'Ardhachandra': {
        'color':       (255, 230, 190),  # bright moonlight
        'glow_color':  (245, 220, 180),
        'p_color':     (255, 235, 200),
        'geometry_fn': draw_ardhachandra_moon,
        'p_behavior':  'orbit',
        'p_count':     3,
        'trail_color': (245, 220, 185),
    },
    'Alapadma': {
        'color':       (130, 150, 255),
        'glow_color':  (120, 140, 245),
        'p_color':     (140, 160, 255),
        'geometry_fn': draw_alapadma_lotus,
        'p_behavior':  'none',
        'p_count':     0,
        'trail_color': (125, 145, 250),
    },
}


# ─────────────────────────────────────────────────────
# MAIN RENDERER
# ─────────────────────────────────────────────────────

class MudraRenderer:
    def __init__(self, width=640, height=480):
        self.w = width
        self.h = height
        self.particles  = ParticleSystem(max_particles=200)
        self.trail      = TrailSystem(maxlen=30)
        self.frame_count = 0

    def _palm_center(self, landmarks):
        cx = int(landmarks[9][0] * self.w)
        cy = int(landmarks[9][1] * self.h)
        return cx, cy

    def _hand_radius(self, landmarks):
        dx = (landmarks[9][0]-landmarks[0][0]) * self.w
        dy = (landmarks[9][1]-landmarks[0][1]) * self.h
        return max(int(math.sqrt(dx*dx+dy*dy)), 25)

    def render(self, frame, mudra, score, landmarks,
               handedness='Right'):
        if not landmarks:
            self.particles.update_draw(frame)
            return frame

        self.frame_count += 1
        now = time.time()

        cx, cy  = self._palm_center(landmarks)
        radius  = self._hand_radius(landmarks)

        # Always update trail
        self.trail.update(cx, cy)

        # If unknown or low score — draw minimal
        if mudra == 'Unknown' or score < 0.55:
            self.trail.draw(frame, (100,100,100), 1)
            self.particles.update_draw(frame)
            return frame

        theme = MUDRA_THEMES.get(mudra)
        if not theme:
            self.particles.update_draw(frame)
            return frame

        color    = theme['color']
        g_color  = theme['glow_color']
        p_color  = theme['p_color']
        geo_fn   = theme['geometry_fn']
        behavior = theme['p_behavior']
        p_count  = theme['p_count']
        t_color  = theme['trail_color']

        # LAYER 1 — Trail
        self.trail.draw(frame, t_color, thickness=2)

        # LAYER 2 — Glow
        draw_glow(frame, cx, cy, radius, g_color)

        # LAYER 3 — Geometry
        geo_fn(frame, cx, cy, radius, color, now)

        # LAYER 4 — Particles (every other frame)
        if self.frame_count % 2 == 0 and \
                theme.get('p_count', 0) > 0:
            self.particles.spawn(
                cx, cy, p_color, behavior,
                speed_factor=0.8, count=p_count)
        self.particles.update_draw(frame)

        # LAYER 5 — Fingertip dots
        draw_fingertip_dots(
            frame, landmarks, color, self.w, self.h)

        return frame
