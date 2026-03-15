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


def draw_ardhapataka_river(frame, cx, cy, radius,
                            color, now,
                            target_x=None,
                            target_y=None):
    """Ardhapataka: River flowing between two bank lines.
    If target_x/y given: river flows toward that point
    (second hand position).
    If no target: river flows upward from palm center.

    Two parallel bank lines define the river edges.
    Multiple sine wave lines flow between the banks.
    Water color: deep blue-teal with lighter wave crests."""

    # ── Determine river direction ─────────────────────
    if target_x is not None and target_y is not None:
        # Direction from this hand toward other hand
        dx = target_x - cx
        dy = target_y - cy
        river_length = math.sqrt(dx*dx + dy*dy)
        if river_length < 1:
            river_length = 1
        # Unit vector along river
        ux = dx / river_length
        uy = dy / river_length
    else:
        # Default: flow upward
        ux = 0.0
        uy = -1.0
        river_length = int(radius * 3.5)

    # Perpendicular to river direction (bank direction)
    px = -uy
    py =  ux

    # River width — scales with hand size
    bank_gap = int(radius * 0.55)

    # ── Draw river bank lines ─────────────────────────
    # Left bank
    left_start_x = int(cx + px * bank_gap)
    left_start_y = int(cy + py * bank_gap)
    left_end_x   = int(cx + px * bank_gap
                       + ux * river_length)
    left_end_y   = int(cy + py * bank_gap
                       + uy * river_length)

    # Right bank
    right_start_x = int(cx - px * bank_gap)
    right_start_y = int(cy - py * bank_gap)
    right_end_x   = int(cx - px * bank_gap
                        + ux * river_length)
    right_end_y   = int(cy - py * bank_gap
                        + uy * river_length)

    # Draw bank lines — bright silver
    bank_color = (235, 235, 255)
    ov = frame.copy()
    cv2.line(ov,
        (left_start_x,  left_start_y),
        (left_end_x,    left_end_y),
        bank_color, 3)
    cv2.addWeighted(ov, 0.80, frame, 0.20, 0, frame)

    ov = frame.copy()
    cv2.line(ov,
        (right_start_x, right_start_y),
        (right_end_x,   right_end_y),
        bank_color, 3)
    cv2.addWeighted(ov, 0.80, frame, 0.20, 0, frame)

    # Soft bank glow
    for start, end in [
        ((left_start_x,  left_start_y),
         (left_end_x,    left_end_y)),
        ((right_start_x, right_start_y),
         (right_end_x,   right_end_y)),
    ]:
        ov = frame.copy()
        cv2.line(ov, start, end, bank_color, 8)
        cv2.addWeighted(ov, 0.15, frame, 0.85, 0, frame)

    # ── Draw flowing water sine waves ────────────────
    # Multiple sine wave lines between the banks
    # Each wave offset in phase for flowing animation

    n_waves    = 6      # number of wave lines
    n_steps    = 80     # points per wave line
    wave_amp   = bank_gap * 0.35  # wave height
    wave_freq  = 2.5    # cycles along river length
    flow_speed = 2.2    # animation speed

    # Water colors — deep teal to bright cyan
    wave_colors = [
        (180, 120,  40),   # deep teal
        (200, 150,  60),   # teal
        (220, 180,  80),   # medium teal
        (235, 200, 100),   # lighter teal
        (245, 220, 130),   # pale teal
        (255, 240, 160),   # near white-blue
    ]

    for w in range(n_waves):
        # Position each wave evenly across river width
        t_across = (w + 0.5) / n_waves  # 0.0 to 1.0
        # Offset from right bank to left bank
        offset_mult = -bank_gap + t_across * bank_gap * 2
        phase_offset = w * 0.8  # stagger wave phases

        pts = []
        for s in range(n_steps):
            # Position along river (0 to river_length)
            t_along = s / (n_steps - 1)
            dist_along = t_along * river_length

            # Sine wave perpendicular to flow direction
            sine_val = math.sin(
                t_along * wave_freq * 2 * math.pi
                - now * flow_speed
                + phase_offset
            ) * wave_amp

            # Final point position
            px_pos = int(
                cx
                + ux * dist_along        # along river
                + px * offset_mult       # across river
                + px * sine_val          # wave wobble
            )
            py_pos = int(
                cy
                + uy * dist_along
                + py * offset_mult
                + py * sine_val
            )

            # Clamp to frame bounds
            h, fw = frame.shape[:2]
            px_pos = max(0, min(fw-1, px_pos))
            py_pos = max(0, min(h-1,  py_pos))
            pts.append((px_pos, py_pos))

        if len(pts) < 2:
            continue

        # Draw wave line with alpha
        wave_c = wave_colors[w % len(wave_colors)]

        # Fade alpha based on wave position
        # Center waves brighter, edge waves more faint
        center_dist = abs(t_across - 0.5) * 2  # 0=center
        alpha = 0.65 - center_dist * 0.20

        ov = frame.copy()
        for i in range(1, len(pts)):
            cv2.line(ov, pts[i-1], pts[i], wave_c, 1)
        cv2.addWeighted(ov, alpha,
                        frame, 1-alpha, 0, frame)

    # ── Bright wave crests ────────────────────────────
    # Highlight brightest points of top wave
    crest_pts = []
    for s in range(n_steps):
        t_along = s / (n_steps - 1)
        dist_along = t_along * river_length
        sine_val = math.sin(
            t_along * wave_freq * 2 * math.pi
            - now * flow_speed
        )
        # Only draw crests (top of sine)
        if sine_val > 0.75:
            px_pos = int(cx + ux*dist_along
                         + px * sine_val * wave_amp * 0.3)
            py_pos = int(cy + uy*dist_along
                         + py * sine_val * wave_amp * 0.3)
            h, fw = frame.shape[:2]
            if 0 <= px_pos < fw and 0 <= py_pos < h:
                cv2.circle(frame, (px_pos, py_pos),
                           2, (255,255,255), -1)

    # ── Soft water fill between banks ────────────────
    # Semi-transparent blue fill for water body
    fill_pts = []
    # Left bank points
    for s in range(0, n_steps, 4):
        t_along = s / (n_steps-1)
        dist = t_along * river_length
        fill_pts.append((
            int(cx + ux*dist + px*bank_gap),
            int(cy + uy*dist + py*bank_gap)
        ))
    # Right bank points (reversed)
    for s in range(n_steps-1, 0, -4):
        t_along = s / (n_steps-1)
        dist = t_along * river_length
        fill_pts.append((
            int(cx + ux*dist - px*bank_gap),
            int(cy + uy*dist - py*bank_gap)
        ))

    if len(fill_pts) > 3:
        poly = np.array(fill_pts,
                        np.int32).reshape((-1,1,2))
        ov = frame.copy()
        cv2.fillPoly(ov, [poly], (180, 100, 30))
        cv2.addWeighted(ov, 0.18,
                        frame, 0.82, 0, frame)

    # ── Palm glow at river source ─────────────────────
    ov = frame.copy()
    cv2.circle(ov, (cx,cy),
               int(radius*0.9), bank_color, -1)
    cv2.addWeighted(ov, 0.20, frame, 0.80, 0, frame)
    cv2.circle(frame, (cx,cy),
               int(radius*0.35), bank_color, 2)


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
    """Low-poly lotus matching reference image shape.
    NOT a perfect circle — crown/flower shape where:
    - Bottom outer petals spread wide and horizontal
    - Upper petals rise more vertically
    - 3 layers: outer white-pink, middle pink, inner magenta
    - Orange-yellow center stamen
    - Each petal made of triangular facets (low-poly)"""

    breathe = 1.0 + 0.04 * math.sin(now * 1.0)

    def draw_faceted_petal(frame, cx, cy,
                           angle, length, width,
                           tip_color, mid_color,
                           base_color, alpha):
        """Single low-poly faceted petal."""
        perp = angle + math.pi/2

        tip_x  = int(cx + math.cos(angle)*length)
        tip_y  = int(cy + math.sin(angle)*length)
        base_x = int(cx + math.cos(angle)*length*0.06)
        base_y = int(cy + math.sin(angle)*length*0.06)

        w_dist = length * 0.42
        left_x  = int(cx + math.cos(angle)*w_dist
                      + math.cos(perp)*width)
        left_y  = int(cy + math.sin(angle)*w_dist
                      + math.sin(perp)*width)
        right_x = int(cx + math.cos(angle)*w_dist
                      - math.cos(perp)*width)
        right_y = int(cy + math.sin(angle)*w_dist
                      - math.sin(perp)*width)

        mid_x = int(cx + math.cos(angle)*length*0.60)
        mid_y = int(cy + math.sin(angle)*length*0.60)

        # 4 triangular facets
        facets = [
            ([base_x,base_y],[left_x,left_y],
             [mid_x,mid_y],   mid_color, 0),
            ([base_x,base_y],[right_x,right_y],
             [mid_x,mid_y],
             tuple(min(255,c+25) for c in mid_color), 0),
            ([left_x,left_y],[tip_x,tip_y],
             [mid_x,mid_y],   tip_color, 0),
            ([right_x,right_y],[tip_x,tip_y],
             [mid_x,mid_y],
             tuple(min(255,c+20) for c in tip_color), 0),
        ]

        for p1,p2,p3,fc,_ in facets:
            pts = np.array([p1,p2,p3],
                np.int32).reshape((-1,1,2))
            ov = frame.copy()
            cv2.fillPoly(ov, [pts], fc)
            cv2.addWeighted(ov, alpha,
                frame, 1-alpha, 0, frame)

        # Low-poly edge lines
        all_pts = [(base_x,base_y),(left_x,left_y),
                   (mid_x,mid_y),(tip_x,tip_y),
                   (right_x,right_y)]
        for a_i,b_i in [(0,1),(1,2),(2,3),(3,4),
                         (4,2),(0,4),(1,3)]:
            cv2.line(frame, all_pts[a_i], all_pts[b_i],
                (200,200,200), 1)

    # ══════════════════════════════════════════════
    # PETAL ANGLE LAYOUT — crown/flower shape
    # NOT evenly spaced circle
    # Mimics reference: wide bottom, rising sides, top
    # Angles in degrees, 0=right, 90=down, -90=up
    # ══════════════════════════════════════════════

    # OUTER LAYER — white to pale pink
    # Wide flat petals, spread like a crown base
    outer_angles_deg = [
        -90,   # straight up — center top petal
        -130,  # upper left
        -50,   # upper right
        180,   # straight left
        0,     # straight right
        130,   # lower left
        50,    # lower right
        -160,  # far upper left
        -20,   # far upper right
    ]
    outer_tip  = (245, 235, 255)  # near white-pink
    outer_mid  = (220, 195, 245)  # pale pink
    outer_base = (190, 155, 225)  # soft pink

    l1 = int(radius * 2.8 * breathe)
    for i, deg in enumerate(outer_angles_deg):
        angle = math.radians(deg)
        # Horizontal petals wider, vertical petals narrower
        # Petals near 0/180 degrees are wider (horizontal)
        horiz_factor = abs(math.cos(angle))
        w = int(radius * (0.35 + 0.22 * horiz_factor))
        # Horizontal petals also longer
        l = int(l1 * (0.85 + 0.20 * horiz_factor))
        draw_faceted_petal(frame, cx, cy, angle,
                           l, w,
                           outer_tip, outer_mid,
                           outer_base, 0.68)

    # MIDDLE LAYER — medium pink
    # More upright petals, crown shape
    mid_angles_deg = [
        -90,   # top center
        -115,  # upper left
        -65,   # upper right
        -145,  # far upper left
        -35,   # far upper right
        160,   # lower left
        20,    # lower right
        -170,  # far lower left
        10,    # far lower right
    ]
    mid_tip  = (180, 100, 230)
    mid_mid  = (150,  65, 205)
    mid_base = (120,  45, 180)

    l2 = int(radius * 1.95 * breathe)
    for deg in mid_angles_deg:
        angle = math.radians(deg)
        horiz_factor = abs(math.cos(angle))
        w = int(radius * (0.28 + 0.14 * horiz_factor))
        draw_faceted_petal(frame, cx, cy, angle,
                           l2, w,
                           mid_tip, mid_mid,
                           mid_base, 0.75)

    # INNER LAYER — deep magenta, tight upright petals
    inner_angles_deg = [
        -90,   # top
        -112,  # left of top
        -68,   # right of top
        -135,  # left
        -45,   # right
        160,   # lower left
        20,    # lower right
    ]
    inner_tip  = (150,  55, 210)
    inner_mid  = (110,  25, 175)
    inner_base = ( 85,  15, 150)

    l3 = int(radius * 1.25 * breathe)
    for deg in inner_angles_deg:
        angle = math.radians(deg)
        horiz_factor = abs(math.cos(angle))
        w = int(radius * (0.20 + 0.08 * horiz_factor))
        draw_faceted_petal(frame, cx, cy, angle,
                           l3, w,
                           inner_tip, inner_mid,
                           inner_base, 0.82)

    # ── ORANGE-YELLOW STAMEN CENTER ──────────────────
    # Outer orange ring
    ov = frame.copy()
    cv2.circle(ov, (cx,cy),
               int(radius*0.44), (0,150,245), -1)
    cv2.addWeighted(ov, 0.88, frame, 0.12, 0, frame)

    # Facet lines on stamen
    for i in range(6):
        angle = (2*math.pi/6)*i
        sx = int(cx + math.cos(angle)*radius*0.40)
        sy = int(cy + math.sin(angle)*radius*0.40)
        cv2.line(frame,(cx,cy),(sx,sy),
                 (0,200,255),1)

    # Inner yellow
    ov = frame.copy()
    cv2.circle(ov,(cx,cy),
               int(radius*0.26),(0,220,255),-1)
    cv2.addWeighted(ov,0.92,frame,0.08,0,frame)

    # Bright white hot center
    ov = frame.copy()
    cv2.circle(ov,(cx,cy),
               int(radius*0.11),(255,255,255),-1)
    cv2.addWeighted(ov,0.95,frame,0.05,0,frame)

    # Soft outer bloom glow
    ov = frame.copy()
    cv2.circle(ov,(cx,cy),
               int(radius*3.2),(190,140,255),-1)
    cv2.addWeighted(ov,0.07,frame,0.93,0,frame)


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
        'color':       (235, 235, 255),
        'glow_color':  (220, 220, 245),
        'p_color':     (240, 240, 255),
        'geometry_fn': draw_ardhapataka_river,
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
        self.second_hand_pos = None

    def _palm_center(self, landmarks):
        cx = int(landmarks[9][0] * self.w)
        cy = int(landmarks[9][1] * self.h)
        return cx, cy

    def _hand_radius(self, landmarks):
        dx = (landmarks[9][0]-landmarks[0][0]) * self.w
        dy = (landmarks[9][1]-landmarks[0][1]) * self.h
        return max(int(math.sqrt(dx*dx+dy*dy)), 25)

    def render(self, frame, mudra, score,
               landmarks, handedness='Right',
               second_landmarks=None):
        if not landmarks:
            self.particles.update_draw(frame)
            return frame

        self.frame_count += 1
        now = time.time()

        cx, cy  = self._palm_center(landmarks)
        radius  = self._hand_radius(landmarks)

        # Track second hand position for river effect
        if second_landmarks:
            s_cx = int(second_landmarks[9][0] * self.w)
            s_cy = int(second_landmarks[9][1] * self.h)
            self.second_hand_pos = (s_cx, s_cy)
        else:
            self.second_hand_pos = None

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
        # Special case: Ardhapataka needs second hand pos
        if mudra == 'Ardhapataka' and \
                hasattr(self, 'second_hand_pos') and \
                self.second_hand_pos is not None:
            tx, ty = self.second_hand_pos
            geo_fn(frame, cx, cy, radius,
                   color, now, tx, ty)
        else:
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
