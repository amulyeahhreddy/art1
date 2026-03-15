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
# GROUP 2 GEOMETRY FUNCTIONS
# ─────────────────────────────────────────────────────

def draw_katakamukha_petals(frame, cx, cy,
                             radius, color, now):
    """Real flower petals falling from pinch point.
    Each flower is drawn as 5 oval ellipse petals
    arranged around a yellow center — like jasmine
    or a simple daisy. Petals fall with sideways drift.
    Flowers are large and clearly recognizable."""

    pinch_x = cx
    pinch_y = cy - int(radius * 0.5)

    # Glow at pinch point
    ov = frame.copy()
    cv2.circle(ov, (pinch_x, pinch_y),
               int(radius*0.50), color, -1)
    cv2.addWeighted(ov, 0.28, frame, 0.72, 0, frame)
    cv2.circle(frame, (pinch_x, pinch_y),
               int(radius*0.18), color, 2)

    # Flower petal colors — rose and jasmine
    petal_colors = [
        (160, 130, 255),  # deep rose
        (190, 155, 255),  # medium rose
        (210, 185, 255),  # light rose
        (175, 210, 255),  # jasmine white-pink
        (140, 195, 240),  # jasmine teal
        (220, 200, 255),  # pale pink
    ]
    center_color = (0, 220, 255)   # bright yellow center

    n_flowers = 10
    rng = np.random.RandomState(33)

    for i in range(n_flowers):
        # Fall parameters
        drift_dir  = rng.uniform(-1.0, 1.0)
        fall_spd   = 0.30 + rng.random()*0.40
        petal_long = rng.randint(10, 18)  # ellipse long axis
        petal_short= int(petal_long * 0.55) # ellipse short axis
        color_idx  = rng.randint(0, len(petal_colors))
        flower_rot = rng.uniform(0, 2*math.pi)
        rot_spd    = rng.uniform(-0.4, 0.4)

        # Animation
        t = (now * fall_spd + i * 0.70) % 1.0

        fall_dist  = int(t * radius * 4.2)
        drift_dist = int(drift_dir * t
                         * radius * 1.6)

        fx = pinch_x + drift_dist
        fy = pinch_y + fall_dist

        # Fade
        if t < 0.15:
            alpha = (t/0.15) * 0.88
        elif t > 0.72:
            alpha = ((1.0-t)/0.28) * 0.88
        else:
            alpha = 0.88

        if alpha < 0.05:
            continue

        # Current rotation
        current_rot = flower_rot + now * rot_spd
        pc = petal_colors[color_idx]

        # ── Draw flower: 5 oval petals ───────────────
        # Each petal is an ellipse offset from center
        n_petals  = 5
        petal_dist = int(petal_long * 0.72)

        for p in range(n_petals):
            petal_angle = current_rot + \
                          p*(2*math.pi/n_petals)

            # Petal center position
            pet_cx = int(fx + math.cos(petal_angle)
                         * petal_dist)
            pet_cy = int(fy + math.sin(petal_angle)
                         * petal_dist)

            # Draw oval petal
            # Ellipse oriented along petal_angle
            angle_deg = int(math.degrees(petal_angle))

            ov = frame.copy()
            cv2.ellipse(ov,
                (pet_cx, pet_cy),
                (petal_long, petal_short),
                angle_deg,
                0, 360,
                pc, -1)
            cv2.addWeighted(ov, alpha,
                            frame, 1-alpha,
                            0, frame)

            # Petal outline slightly darker
            darker = tuple(max(0,c-40) for c in pc)
            cv2.ellipse(frame,
                (pet_cx, pet_cy),
                (petal_long, petal_short),
                angle_deg,
                0, 360,
                darker, 1)

        # ── Flower center — bright yellow circle ────
        center_r = int(petal_long * 0.38)
        ov = frame.copy()
        cv2.circle(ov, (int(fx),int(fy)),
                   center_r, center_color, -1)
        cv2.addWeighted(ov, alpha*0.95,
                        frame, 1-alpha*0.95,
                        0, frame)

        # Center dot ring — tiny stamens
        if center_r > 4:
            for d in range(6):
                da = d*(2*math.pi/6)+current_rot
                dx = int(fx + math.cos(da)*center_r*0.65)
                dy = int(fy + math.sin(da)*center_r*0.65)
                cv2.circle(frame,(dx,dy),
                           1,(255,255,255),-1)

    # Garland arc below pinch
    arc_pts = []
    for t in np.linspace(-math.pi*0.55,
                          math.pi*0.55, 35):
        ax = int(pinch_x + math.sin(t)*radius*1.1)
        ay = int(pinch_y + int(radius*0.9)
                 - math.cos(t)*radius*0.25)
        arc_pts.append((ax,ay))
    if len(arc_pts) > 1:
        ov = frame.copy()
        for i in range(1, len(arc_pts)):
            cv2.line(ov, arc_pts[i-1],
                     arc_pts[i], color, 1)
        cv2.addWeighted(ov, 0.40,
                        frame, 0.60, 0, frame)


def draw_mushti_core(frame, cx, cy,
                     radius, color, now):
    """Heavy pulsing radial glow from inside fist.
    Sine wave oscillates intensity like a heartbeat.
    Deep crimson core that breathes in and out."""

    # Pulse using sine wave — heartbeat rhythm
    pulse_fast  = (math.sin(now * 3.5) + 1) / 2
    pulse_slow  = (math.sin(now * 1.2) + 1) / 2
    combined    = pulse_fast * 0.6 + pulse_slow * 0.4

    # Core radius pulses
    core_r  = int(radius * (0.5 + combined * 0.4))
    inner_r = int(radius * (0.25 + combined * 0.20))

    # Outer shockwave ring — appears at pulse peak
    if pulse_fast > 0.85:
        shock_r = int(radius * (1.5 + pulse_fast * 0.8))
        ov = frame.copy()
        cv2.circle(ov, (cx,cy), shock_r, color, 2)
        cv2.addWeighted(ov, 0.60,
                        frame, 0.40, 0, frame)

    # Multiple glow layers — radial gradient simulation
    glow_layers = [
        (radius * 2.2, 0.08 * combined),
        (radius * 1.7, 0.15 * combined),
        (radius * 1.3, 0.25 * combined),
        (radius * 1.0, 0.40 * combined),
        (core_r,       0.60 * combined),
        (inner_r,      0.80),
    ]
    for r_mult, alpha in glow_layers:
        r = int(r_mult)
        if r < 1:
            continue
        ov = frame.copy()
        cv2.circle(ov, (cx,cy), r, color, -1)
        cv2.addWeighted(ov, alpha,
                        frame, 1-alpha, 0, frame)

    # Bright white hot center at peak
    if combined > 0.6:
        white_r = int(radius * 0.12 * combined)
        ov = frame.copy()
        cv2.circle(ov, (cx,cy),
                   white_r, (255,255,255), -1)
        cv2.addWeighted(ov, combined * 0.90,
                        frame, 1-combined*0.90,
                        0, frame)

    # Radiating crack lines at pulse peak
    if pulse_fast > 0.70:
        n_cracks = 8
        crack_len = int(radius * (1.5 + combined))
        for i in range(n_cracks):
            angle = (2*math.pi/n_cracks)*i
            ex = int(cx + math.cos(angle)*crack_len)
            ey = int(cy + math.sin(angle)*crack_len)
            ov = frame.copy()
            cv2.line(ov, (cx,cy), (ex,ey),
                     color, 1)
            cv2.addWeighted(ov, 0.45 * pulse_fast,
                            frame,
                            1-0.45*pulse_fast,
                            0, frame)


def draw_shikhara_pillar(frame, cx, cy,
                              radius, color, now):
        """Sacred pillar / Shiva lingam rising upward.
        Vertical beam with glowing column.
        Bow effect deferred until body landmarks added."""

        pillar_h = int(radius * 3.8)
        top_y    = cy - pillar_h
        pulse    = (math.sin(now * 1.5) + 1) / 2

        # Outer soft glow column
        for w, alpha in [(20,0.08),(14,0.14),(8,0.25),(4,0.55)]:
            ov = frame.copy()
            cv2.rectangle(ov,
                (cx - w, top_y),
                (cx + w, cy),
                color, -1)
            cv2.addWeighted(ov, alpha,
                            frame, 1-alpha, 0, frame)

        # Bright core line
        ov = frame.copy()
        cv2.line(ov, (cx, top_y), (cx, cy), color, 2)
        cv2.addWeighted(ov, 0.85, frame, 0.15, 0, frame)

        # Pulsing base glow
        base_r = int(radius * (0.7 + pulse*0.3))
        ov = frame.copy()
        cv2.circle(ov, (cx,cy), base_r, color, -1)
        cv2.addWeighted(ov, 0.25 + pulse*0.15,
                        frame,
                        1-(0.25+pulse*0.15),
                        0, frame)

        # Top cap glow
        ov = frame.copy()
        cv2.circle(ov, (cx, top_y),
                   int(radius*0.3), color, -1)
        cv2.addWeighted(ov, 0.60, frame, 0.40, 0, frame)

        # Sparkles along pillar
        if int(now*20) % 2 == 0:
            sy = np.random.randint(top_y, cy)
            sx = cx + np.random.randint(-6, 6)
            cv2.circle(frame, (sx,sy), 2,
                       (255,255,255), -1)


def draw_trishula_flames(frame, cx, cy,
                          radius, color, now,
                          tip1=None, tip2=None,
                          tip3=None):
    # Shiva's trishula: gold metallic shaft + red fire
    trident_color = (30, 180, 220)   # gold BGR
    shaft_color   = (40, 200, 230)   # bright gold
    glow_color    = (20, 140, 180)   # deep gold

    # ── TRIDENT GEOMETRY ─────────────────────────────

    shaft_len   = int(radius * 2.5)
    prong_h     = int(radius * 1.8)
    prong_gap   = int(radius * 0.55)
    cross_y     = cy - int(radius * 0.6)

    # Shaft — vertical center line
    shaft_bot = cy + int(radius * 0.8)
    shaft_top = cy - int(radius * 0.5)

    for w, alpha in [(8,0.12),(4,0.25),(2,0.60)]:
        ov = frame.copy()
        cv2.line(ov, (cx, shaft_bot),
                     (cx, shaft_bot - shaft_len),
                     shaft_color, w)
        cv2.addWeighted(ov, alpha,
                        frame, 1-alpha, 0, frame)

    # Three prong bases — spread from shaft top
    prong_bases = [
        (cx - prong_gap, shaft_top),  # left
        (cx,             shaft_top),  # center
        (cx + prong_gap, shaft_top),  # right
    ]

    # Three prong tips — points going up
    prong_tips = [
        (cx - prong_gap, shaft_top - prong_h),
        (cx,             shaft_top - prong_h
                         - int(radius*0.4)),  # center taller
        (cx + prong_gap, shaft_top - prong_h),
    ]

    # Draw prongs
    for base, tip in zip(prong_bases, prong_tips):
        for w, alpha in [(6,0.15),(3,0.30),(1,0.75)]:
            ov = frame.copy()
            cv2.line(ov, base, tip, trident_color, w)
            cv2.addWeighted(ov, alpha,
                            frame, 1-alpha, 0, frame)

    # Prong curve tips — small outward hooks
    for i, (base, tip) in enumerate(
            zip(prong_bases, prong_tips)):
        if i != 1:  # only outer prongs get hooks
            hook_dir = -1 if i == 0 else 1
            hook_x = tip[0] + hook_dir*int(radius*0.25)
            hook_y = tip[1] + int(radius*0.20)
            ov = frame.copy()
            cv2.line(ov, tip,
                         (hook_x, hook_y),
                         trident_color, 2)
            cv2.addWeighted(ov, 0.65,
                            frame, 0.35, 0, frame)

    # Crossbar connecting outer prong bases
    ov = frame.copy()
    cv2.line(ov,
        (prong_bases[0][0], cross_y),
        (prong_bases[2][0], cross_y),
        trident_color, 2)
    cv2.addWeighted(ov, 0.65,
                    frame, 0.35, 0, frame)

    # Second crossbar slightly below
    cross_y2 = cross_y + int(radius*0.3)
    ov = frame.copy()
    cv2.line(ov,
        (prong_bases[0][0] - int(radius*0.1),
         cross_y2),
        (prong_bases[2][0] + int(radius*0.1),
         cross_y2),
        trident_color, 1)
    cv2.addWeighted(ov, 0.40,
                    frame, 0.60, 0, frame)

    # ── FLAMES AT PRONG TIPS ─────────────────────────

    flame_colors = [
        (0,   40, 180),   # deep red
        (0,   80, 220),   # bright red
        (20, 130, 240),   # orange-red
        (40, 170, 245),   # orange
        (60, 200, 255),   # bright orange tip
    ]

    for i, tip in enumerate(prong_tips):
        tip_x, tip_y = tip
        phase = i * 1.3

        # Flame height varies
        f_h = int(radius * (1.2 if i==1 else 0.9))
        f_w = int(radius * 0.22)

        for seg in range(6):
            t      = seg / 6
            t_next = (seg+1) / 6
            y1 = int(tip_y - t * f_h)
            y2 = int(tip_y - t_next * f_h)
            w1 = int(f_w * (1.0-t))
            w2 = int(f_w * (1.0-t_next))
            wb1 = int(math.sin(
                now*5+t*4+phase)*f_w*0.35)
            wb2 = int(math.sin(
                now*5+t_next*4+phase)*f_w*0.35)

            seg_pts = np.array([
                [tip_x-w1, y1],
                [tip_x+w1, y1],
                [tip_x+w2+wb2, y2],
                [tip_x-w2+wb2, y2],
            ], np.int32).reshape((-1,1,2))

            fc_idx = min(seg, len(flame_colors)-1)
            fc = flame_colors[fc_idx]
            alpha = (1.0-t) * 0.65

            ov = frame.copy()
            cv2.fillPoly(ov, [seg_pts], fc)
            cv2.addWeighted(ov, alpha,
                            frame, 1-alpha,
                            0, frame)

        # Glow at tip
        ov = frame.copy()
        cv2.circle(ov, (tip_x, tip_y),
                   int(radius*0.20), glow_color, -1)
        cv2.addWeighted(ov, 0.55,
                        frame, 0.45, 0, frame)

    # ── SHAFT GLOW ───────────────────────────────────
    ov = frame.copy()
    cv2.circle(ov, (cx, shaft_bot),
               int(radius*0.45), glow_color, -1)
    cv2.addWeighted(ov, 0.25,
                    frame, 0.75, 0, frame)


def draw_tamrachuda_crest(frame, cx, cy,
                           radius, color, now):
    """Clear rooster silhouette:
    - Large sharp beak pointing right
    - Rooster comb: 3 bumps on top
    - Bright eye dot
    - Tail feathers fanning left/down
    - Body glow at palm center
    Fire-red colors throughout."""

    pulse = (math.sin(now*2.5)+1)/2

    # ── ROOSTER BODY GLOW ────────────────────────────
    body_r = int(radius * 1.0)
    ov = frame.copy()
    cv2.circle(ov, (cx,cy), body_r, color, -1)
    cv2.addWeighted(ov, 0.20, frame, 0.80, 0, frame)

    # ── BEAK — large sharp triangle ──────────────────
    beak_color  = (0, 80, 220)   # rooster red
    beak_length = int(radius * 2.0)
    beak_width  = int(radius * 0.45)

    beak_tip_x  = cx + beak_length
    beak_tip_y  = cy - int(radius * 0.1)
    beak_top_x  = cx + int(radius * 0.3)
    beak_top_y  = cy - beak_width
    beak_bot_x  = cx + int(radius * 0.3)
    beak_bot_y  = cy + int(beak_width * 0.5)

    beak_pts = np.array([
        [beak_top_x, beak_top_y],
        [beak_tip_x, beak_tip_y],
        [beak_bot_x, beak_bot_y],
    ], np.int32).reshape((-1,1,2))

    ov = frame.copy()
    cv2.fillPoly(ov, [beak_pts], beak_color)
    cv2.addWeighted(ov, 0.85,
                    frame, 0.15, 0, frame)
    cv2.polylines(frame, [beak_pts],
                  True, (0,120,255), 1)

    # Beak glow
    ov = frame.copy()
    cv2.fillPoly(ov, [beak_pts], beak_color)
    cv2.addWeighted(ov, 0.20,
                    frame, 0.80, 0, frame)

    # ── EYE — bright white with pupil ────────────────
    eye_x = cx + int(radius * 0.55)
    eye_y = cy - int(radius * 0.45)

    # Eye white
    ov = frame.copy()
    cv2.circle(ov, (eye_x, eye_y),
               int(radius*0.18), (255,255,255), -1)
    cv2.addWeighted(ov, 0.90,
                    frame, 0.10, 0, frame)
    # Pupil
    cv2.circle(frame, (eye_x, eye_y),
               int(radius*0.09), (0,0,0), -1)
    # Eye glow ring
    ov = frame.copy()
    cv2.circle(ov, (eye_x, eye_y),
               int(radius*0.25), beak_color, 1)
    cv2.addWeighted(ov, 0.50,
                    frame, 0.50, 0, frame)

    # ── COMB — 3 bumps on top of head ────────────────
    comb_color = (0, 60, 220)  # deep red
    comb_base_y = cy - int(radius * 0.6)
    comb_x_positions = [
        cx + int(radius*0.4),
        cx + int(radius*0.15),
        cx - int(radius*0.1),
    ]
    comb_heights = [
        int(radius * 1.1),
        int(radius * 0.85),
        int(radius * 0.65),
    ]

    for i, (comb_x, comb_h) in enumerate(
            zip(comb_x_positions, comb_heights)):
        # Animate comb with slight wobble
        wobble = int(math.sin(now*3+i)*radius*0.05)
        comb_tip_y = comb_base_y - comb_h + wobble

        comb_pts = np.array([
            [comb_x - int(radius*0.12),
             comb_base_y],
            [comb_x,
             comb_tip_y],
            [comb_x + int(radius*0.12),
             comb_base_y],
        ], np.int32).reshape((-1,1,2))

        ov = frame.copy()
        cv2.fillPoly(ov, [comb_pts], comb_color)
        cv2.addWeighted(ov, 0.80,
                        frame, 0.20, 0, frame)

        # Comb glow
        ov = frame.copy()
        cv2.fillPoly(ov, [comb_pts],
                     (20, 100, 240))
        cv2.addWeighted(ov, 0.20,
                        frame, 0.80, 0, frame)

    # ── TAIL FEATHERS — fan left and down ────────────
    tail_colors = [
        (0,   70, 200),
        (0,  110, 230),
        (20, 160, 240),
        (40, 190, 220),
        (60, 210, 190),
    ]

    n_tail = 8
    for i in range(n_tail):
        t = i / (n_tail-1)
        # Fan: 150 to 240 degrees (left/down side)
        angle = math.radians(150 + t*90)
        center_t = 1.0 - abs(t-0.5)*2
        t_len = int(radius*(1.5 + center_t*1.0))

        wave = math.sin(now*1.5+i*0.6)*0.05
        angle += wave

        tip_x = int(cx + math.cos(angle)*t_len)
        tip_y = int(cy + math.sin(angle)*t_len)

        ctrl_x = int(
            cx + math.cos(angle)*t_len*0.5
            + math.cos(angle+math.pi/2)*radius*0.25)
        ctrl_y = int(
            cy + math.sin(angle)*t_len*0.5
            + math.sin(angle+math.pi/2)*radius*0.25)

        f_pts = []
        for s in range(18):
            st = s/17
            bx = int((1-st)**2*cx
                     + 2*(1-st)*st*ctrl_x
                     + st**2*tip_x)
            by = int((1-st)**2*cy
                     + 2*(1-st)*st*ctrl_y
                     + st**2*tip_y)
            f_pts.append((bx,by))

        fc = tail_colors[
            int(t*(len(tail_colors)-1))]
        alpha = 0.55 + center_t*0.25

        ov = frame.copy()
        for j in range(1, len(f_pts)):
            fade = 1.0 - j/len(f_pts)
            fc_f = tuple(int(c*fade) for c in fc)
            cv2.line(ov, f_pts[j-1], f_pts[j],
                     fc_f, 2)
        cv2.addWeighted(ov, alpha,
                        frame, 1-alpha, 0, frame)

        if center_t > 0.3:
            cv2.circle(frame, (tip_x,tip_y),
                       2, fc, -1)

    # ── WATTLE — small red dewlap below beak ─────────
    wattle_pts = np.array([
        [cx + int(radius*0.35),
         cy + int(radius*0.05)],
        [cx + int(radius*0.25),
         cy + int(radius*0.55)],
        [cx + int(radius*0.55),
         cy + int(radius*0.25)],
    ], np.int32).reshape((-1,1,2))

    ov = frame.copy()
    cv2.fillPoly(ov, [wattle_pts], beak_color)
    cv2.addWeighted(ov, 0.70,
                    frame, 0.30, 0, frame)


def draw_kartarimukha_lightning(frame, cx, cy,
                                 radius, color, now):
    """Two jagged lightning bolts diverging from palm.
    Like separation or lightning splitting.
    Each bolt is a jagged polyline that re-randomizes
    every few frames for flickering effect."""

    def jagged_bolt(start_x, start_y,
                    end_x, end_y,
                    jag, n_segs, seed):
        """Generate jagged lightning bolt points."""
        rng = np.random.RandomState(seed)
        pts = [(start_x, start_y)]
        for i in range(1, n_segs):
            t  = i / n_segs
            mx = int(start_x + (end_x-start_x)*t
                     + rng.randint(-jag, jag))
            my = int(start_y + (end_y-start_y)*t
                     + rng.randint(-jag, jag))
            pts.append((mx, my))
        pts.append((end_x, end_y))
        return pts

    # Seed changes every 3 frames for flicker
    flicker_seed = int(now * 10) % 8

    # Two bolt endpoints — diverge outward
    spread = int(radius * 2.5)
    bolt_len = int(radius * 3.0)

    # Left bolt — goes up-left
    l_end_x = cx - spread
    l_end_y = cy - bolt_len
    # Right bolt — goes up-right
    r_end_x = cx + spread
    r_end_y = cy - bolt_len

    jag_amount = int(radius * 0.25)

    for bolt_end, seed_offset in [
        ((l_end_x, l_end_y), 0),
        ((r_end_x, r_end_y), 4),
    ]:
        bolt_pts = jagged_bolt(
            cx, cy,
            bolt_end[0], bolt_end[1],
            jag_amount, 12,
            flicker_seed + seed_offset)

        # Outer glow bolt
        ov = frame.copy()
        for i in range(1, len(bolt_pts)):
            cv2.line(ov, bolt_pts[i-1],
                     bolt_pts[i], color, 5)
        cv2.addWeighted(ov, 0.25,
                        frame, 0.75, 0, frame)

        # Mid bolt
        ov = frame.copy()
        for i in range(1, len(bolt_pts)):
            cv2.line(ov, bolt_pts[i-1],
                     bolt_pts[i], color, 2)
        cv2.addWeighted(ov, 0.65,
                        frame, 0.35, 0, frame)

        # Bright core
        for i in range(1, len(bolt_pts)):
            cv2.line(frame,
                     bolt_pts[i-1], bolt_pts[i],
                     (255,255,255), 1)

        # Bright flash at bolt tip
        ov = frame.copy()
        cv2.circle(ov, bolt_end,
                   int(radius*0.2), color, -1)
        cv2.addWeighted(ov, 0.70,
                        frame, 0.30, 0, frame)

    # Central flash at palm
    flash = (math.sin(now * 8) + 1) / 2
    ov = frame.copy()
    cv2.circle(ov, (cx,cy),
               int(radius * (0.4 + flash*0.3)),
               color, -1)
    cv2.addWeighted(ov, 0.35 + flash*0.25,
                    frame,
                    1-(0.35+flash*0.25),
                    0, frame)


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
    'Katakamukha': {
        'color':       (160, 130, 230),  # flower pink
        'glow_color':  (150, 120, 220),
        'p_color':     (170, 140, 235),
        'geometry_fn': draw_katakamukha_petals,
        'p_behavior':  'none',
        'p_count':     0,
        'trail_color': (155, 125, 225),
    },
    'Mushti': {
        'color':       (30,  20, 180),   # deep crimson
        'glow_color':  (20,  10, 160),
        'p_color':     (40,  30, 195),
        'geometry_fn': draw_mushti_core,
        'p_behavior':  'none',
        'p_count':     0,
        'trail_color': (25,  15, 170),
    },
    'Shikhara': {
        'color':       (0,  140, 255),   # saffron
        'glow_color':  (0,  120, 240),
        'p_color':     (20, 160, 255),
        'geometry_fn': draw_shikhara_pillar,
        'p_behavior':  'rise',
        'p_count':     3,
        'trail_color': (0,  130, 245),
    },
    'Trishula': {
        'color':       (30, 180, 220),   # gold
        'glow_color':  (20, 150, 200),
        'p_color':     (40, 190, 230),
        'geometry_fn': draw_trishula_flames,
        'p_behavior':  'none',
        'p_count':     0,
        'trail_color': (25, 160, 210),
    },
    'Tamrachuda': {
        'color':       (0,  100, 220),   # fire red
        'glow_color':  (0,   80, 200),
        'p_color':     (20, 120, 235),
        'geometry_fn': draw_tamrachuda_crest,
        'p_behavior':  'none',
        'p_count':     0,
        'trail_color': (0,   90, 210),
    },
    'Kartarimukha': {
        'color':       (255, 200, 100),  # electric blue
        'glow_color':  (240, 185,  90),
        'p_color':     (255, 210, 120),
        'geometry_fn': draw_kartarimukha_lightning,
        'p_behavior':  'scatter',
        'p_count':     4,
        'trail_color': (245, 190,  95),
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
        # Special cases that need landmark positions
        if mudra == 'Ardhapataka' and \
                hasattr(self, 'second_hand_pos') and \
                self.second_hand_pos is not None:
            tx, ty = self.second_hand_pos
            geo_fn(frame, cx, cy, radius,
                   color, now, tx, ty)
        elif mudra == 'Trishula':
            # Pass three fingertip positions
            t1 = (int(landmarks[8][0]*self.w),
                  int(landmarks[8][1]*self.h))
            t2 = (int(landmarks[12][0]*self.w),
                  int(landmarks[12][1]*self.h))
            t3 = (int(landmarks[16][0]*self.w),
                  int(landmarks[16][1]*self.h))
            geo_fn(frame, cx, cy, radius, color, now,
                   t1, t2, t3)
        elif mudra == 'Shikhara':
            geo_fn(frame, cx, cy, radius, color, now)
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
