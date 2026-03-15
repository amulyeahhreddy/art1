import cv2
import numpy as np
import math
import time
from collections import deque


# ─────────────────────────────────────────────────────
# TRAIL SYSTEM
# ─────────────────────────────────────────────────────

class TrailSystem:
    def __init__(self, maxlen=55):
        self.points = deque(maxlen=maxlen)
        self._sparks = []

    def update(self, x, y):
        self.points.append((x, y, time.time()))
        # Spawn sparkles at current position
        for _ in range(3):
            angle = np.random.uniform(0, 2 * math.pi)
            spd   = np.random.uniform(0.4, 2.2)
            self._sparks.append({
                'x':     float(x + np.random.randint(-6, 6)),
                'y':     float(y + np.random.randint(-6, 6)),
                'vx':    math.cos(angle) * spd,
                'vy':    math.sin(angle) * spd - 0.8,
                'life':  1.0,
                'decay': np.random.uniform(0.04, 0.09),
                'size':  np.random.randint(1, 4),
            })

    def draw(self, frame, color, thickness=3):
        now = time.time()
        pts = list(self.points)

        # ── Curved trail using Bezier smoothing ──────
        # Build smoothed curve through recent points
        if len(pts) >= 3:
            for i in range(1, len(pts) - 1):
                age  = now - pts[i][2]
                alpha = max(0.0, 1.0 - age * 1.2)
                if alpha <= 0:
                    continue

                t = i / len(pts)  # 0=oldest 1=newest

                # Smooth midpoint between neighbours
                px0, py0 = pts[i-1][0], pts[i-1][1]
                px1, py1 = pts[i][0],   pts[i][1]
                px2, py2 = pts[i+1][0], pts[i+1][1]
                mid_x = int((px0 + px2) / 2)
                mid_y = int((py0 + py2) / 2)

                # Width tapers — thick at tip, thin at tail
                w = max(1, int(thickness * (0.3 + t * 1.4)))

                # Gold color — bright at tip, deep at tail
                gold_bright = (40,  210, 255)   # BGR bright gold
                gold_deep   = (10,  140, 200)   # BGR deep gold
                blend = t
                gc = tuple(int(gold_deep[c] + blend *
                               (gold_bright[c] - gold_deep[c]))
                           for c in range(3))

                # Outer wide glow
                ov = frame.copy()
                cv2.line(ov, (px0, py0), (mid_x, mid_y),
                         gc, w + 8)
                cv2.addWeighted(ov, alpha * 0.15,
                                frame, 1 - alpha * 0.15,
                                0, frame)

                # Mid glow
                ov = frame.copy()
                cv2.line(ov, (px0, py0), (mid_x, mid_y),
                         gc, w + 3)
                cv2.addWeighted(ov, alpha * 0.30,
                                frame, 1 - alpha * 0.30,
                                0, frame)

                # Bright core
                ov = frame.copy()
                cv2.line(ov, (px0, py0), (mid_x, mid_y),
                         gc, w)
                cv2.addWeighted(ov, alpha * 0.75,
                                frame, 1 - alpha * 0.75,
                                0, frame)

                # White hot center line
                if t > 0.65:
                    cv2.line(frame, (px1, py1), (mid_x, mid_y),
                             (255, 255, 255), max(1, w - 1))

        # ── Sparkles ─────────────────────────────────
        alive = []
        for sp in self._sparks:
            sp['x']  += sp['vx']
            sp['vy'] += sp['vy']
            sp['vy'] += 0.05   # gravity
            sp['vx'] *= 0.96
            sp['life'] -= sp['decay']
            if sp['life'] <= 0:
                continue
            a = sp['life']
            sx, sy = int(sp['x']), int(sp['y'])
            h2, w2 = frame.shape[:2]
            if not (0 <= sx < w2 and 0 <= sy < h2):
                alive.append(sp)
                continue

            # Sparkle color — gold to white as it fades
            sc = tuple(int(min(255, c * (0.7 + a * 0.3)))
                       for c in (40, 200, 255))

            # Sparkle glow
            ov = frame.copy()
            cv2.circle(ov, (sx, sy),
                       sp['size'] + 2, sc, -1)
            cv2.addWeighted(ov, a * 0.35,
                            frame, 1 - a * 0.35,
                            0, frame)

            # Sparkle core
            cv2.circle(frame, (sx, sy),
                       sp['size'], sc, -1)

            # Cross sparkle on larger ones
            if sp['size'] >= 3 and a > 0.5:
                cv2.line(frame,
                         (sx - 4, sy), (sx + 4, sy),
                         (255, 255, 255), 1)
                cv2.line(frame,
                         (sx, sy - 4), (sx, sy + 4),
                         (255, 255, 255), 1)

            alive.append(sp)
        self._sparks = alive

        # ── Bright tip burst at latest point ─────────
        if pts:
            rx, ry, rt = pts[-1]
            age_tip = now - rt
            if age_tip < 0.12:
                ov = frame.copy()
                cv2.circle(ov, (rx, ry), 12,
                           (40, 210, 255), -1)
                cv2.addWeighted(ov, 0.35,
                                frame, 0.65, 0, frame)
                cv2.circle(frame, (rx, ry), 4,
                           (255, 255, 255), -1)


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
            elif behavior == 'float':
                vx = np.random.uniform(-0.3, 0.3)
                vy = np.random.uniform(-0.8, -0.2)
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
    for r_mult, alpha in [
        (3.5, 0.15), (2.8, 0.22),
        (2.0, 0.35), (1.4, 0.50)
    ]:
        ov = frame.copy()
        cv2.circle(ov, (cx,cy),
                   int(radius * r_mult), color, 1)
        cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)

    n = 16
    for i in range(n):
        angle = (2*math.pi/n)*i + now*0.25
        ex_far = int(cx + math.cos(angle) * radius * 3.2)
        ey_far = int(cy + math.sin(angle) * radius * 3.2)
        ov = frame.copy()
        cv2.line(ov, (cx,cy), (ex_far, ey_far), color, 4)
        cv2.addWeighted(ov, 0.18, frame, 0.82, 0, frame)
        ex_mid = int(cx + math.cos(angle) * radius * 2.5)
        ey_mid = int(cy + math.sin(angle) * radius * 2.5)
        ov = frame.copy()
        cv2.line(ov, (cx,cy), (ex_mid, ey_mid), color, 2)
        cv2.addWeighted(ov, 0.60, frame, 0.40, 0, frame)

    cv2.circle(frame, (cx,cy), int(radius*0.4), color, 2)
    ov = frame.copy()
    cv2.circle(ov, (cx,cy), int(radius*0.25), color, -1)
    cv2.addWeighted(ov, 0.70, frame, 0.30, 0, frame)


def draw_tripataka_flames(frame, cx, cy, radius, color, now):
    flame_configs = [
        (-int(radius*0.55), 1.6, 0.28),
        (0,                  2.2, 0.35),
        ( int(radius*0.55), 1.6, 0.28),
    ]
    for i, (x_off, h_mult, w_mult) in enumerate(flame_configs):
        flicker = math.sin(now * 4.0 + i * 1.2) * 0.12
        h = int(radius * (h_mult + flicker) * 1.8)
        w = int(radius * w_mult)
        base_x = cx + x_off
        base_y = cy
        tip_wobble = int(math.sin(now*5+i)*radius*0.08)
        pts = np.array([
            [base_x - w,     base_y],
            [base_x - w//2,  base_y - h//2],
            [base_x + tip_wobble, base_y - h],
            [base_x + w//2,  base_y - h//2],
            [base_x + w,     base_y],
        ], np.int32).reshape((-1,1,2))
        ov = frame.copy()
        cv2.fillPoly(ov, [pts], color)
        cv2.addWeighted(ov, 0.35, frame, 0.65, 0, frame)
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
    ov = frame.copy()
    cv2.circle(ov, (cx,cy), int(radius*0.8), color, -1)
    cv2.addWeighted(ov, 0.25, frame, 0.75, 0, frame)


def draw_ardhapataka_river(frame, cx, cy, radius, color, now,
                            target_x=None, target_y=None):
    if target_x is not None and target_y is not None:
        dx = target_x - cx
        dy = target_y - cy
        river_length = math.sqrt(dx*dx + dy*dy)
        if river_length < 1:
            river_length = 1
        ux = dx / river_length
        uy = dy / river_length
    else:
        ux = 0.0
        uy = -1.0
        river_length = int(radius * 3.5)

    px = -uy
    py =  ux
    bank_gap = int(radius * 0.55)

    left_start_x = int(cx + px * bank_gap)
    left_start_y = int(cy + py * bank_gap)
    left_end_x   = int(cx + px * bank_gap + ux * river_length)
    left_end_y   = int(cy + py * bank_gap + uy * river_length)
    right_start_x = int(cx - px * bank_gap)
    right_start_y = int(cy - py * bank_gap)
    right_end_x   = int(cx - px * bank_gap + ux * river_length)
    right_end_y   = int(cy - py * bank_gap + uy * river_length)

    bank_color = (235, 235, 255)
    for start, end in [
        ((left_start_x, left_start_y), (left_end_x, left_end_y)),
        ((right_start_x, right_start_y), (right_end_x, right_end_y)),
    ]:
        ov = frame.copy()
        cv2.line(ov, start, end, bank_color, 3)
        cv2.addWeighted(ov, 0.80, frame, 0.20, 0, frame)
        ov = frame.copy()
        cv2.line(ov, start, end, bank_color, 8)
        cv2.addWeighted(ov, 0.15, frame, 0.85, 0, frame)

    n_waves    = 6
    n_steps    = 80
    wave_amp   = bank_gap * 0.35
    wave_freq  = 2.5
    flow_speed = 2.2
    wave_colors = [
        (180, 120,  40),(200, 150,  60),(220, 180,  80),
        (235, 200, 100),(245, 220, 130),(255, 240, 160),
    ]
    for w in range(n_waves):
        t_across = (w + 0.5) / n_waves
        offset_mult = -bank_gap + t_across * bank_gap * 2
        phase_offset = w * 0.8
        pts = []
        for s in range(n_steps):
            t_along = s / (n_steps - 1)
            dist_along = t_along * river_length
            sine_val = math.sin(
                t_along * wave_freq * 2 * math.pi
                - now * flow_speed + phase_offset) * wave_amp
            px_pos = int(cx + ux*dist_along + px*offset_mult + px*sine_val)
            py_pos = int(cy + uy*dist_along + py*offset_mult + py*sine_val)
            h2, fw = frame.shape[:2]
            px_pos = max(0, min(fw-1, px_pos))
            py_pos = max(0, min(h2-1, py_pos))
            pts.append((px_pos, py_pos))
        if len(pts) < 2:
            continue
        wave_c = wave_colors[w % len(wave_colors)]
        center_dist = abs(t_across - 0.5) * 2
        alpha = 0.65 - center_dist * 0.20
        ov = frame.copy()
        for i in range(1, len(pts)):
            cv2.line(ov, pts[i-1], pts[i], wave_c, 1)
        cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)

    ov = frame.copy()
    cv2.circle(ov, (cx,cy), int(radius*0.9), bank_color, -1)
    cv2.addWeighted(ov, 0.20, frame, 0.80, 0, frame)
    cv2.circle(frame, (cx,cy), int(radius*0.35), bank_color, 2)


def draw_ardhachandra_moon(frame, cx, cy, radius, color, now):
    r_out  = int(radius * 2.2)
    r_in   = int(radius * 1.7)
    shift  = int(radius * 0.9)
    ov = frame.copy()
    cv2.circle(ov, (cx, cy), r_out, color, -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
    ov = frame.copy()
    cv2.circle(ov, (cx + shift, cy), r_in, (0,0,0), -1)
    cv2.addWeighted(ov, 0.85, frame, 0.15, 0, frame)
    cv2.circle(frame, (cx, cy), r_out, color, 2)
    ov = frame.copy()
    cv2.circle(ov, (cx, cy), int(r_out * 1.3), color, -1)
    cv2.addWeighted(ov, 0.10, frame, 0.90, 0, frame)
    tip_x = cx - int(r_out * 0.25)
    for sign in [-1, 1]:
        tip_y = cy + sign * int(r_out * 0.88)
        cv2.circle(frame, (tip_x, tip_y), 4, color, -1)
        ov = frame.copy()
        cv2.circle(ov, (tip_x, tip_y), 10, color, -1)
        cv2.addWeighted(ov, 0.40, frame, 0.60, 0, frame)
    rng = np.random.RandomState(42)
    for _ in range(6):
        sx = cx + rng.randint(-int(r_out*1.5), int(r_out*1.5))
        sy = cy + rng.randint(-int(r_out*1.5), int(r_out*1.5))
        cv2.circle(frame, (sx, sy), 2, color, -1)


def draw_alapadma_lotus(frame, cx, cy, radius, color, now):
    breathe = 1.0 + 0.04 * math.sin(now * 1.0)

    def draw_faceted_petal(frame, cx, cy, angle, length, width,
                           tip_color, mid_color, base_color, alpha):
        perp = angle + math.pi/2
        tip_x  = int(cx + math.cos(angle)*length)
        tip_y  = int(cy + math.sin(angle)*length)
        base_x = int(cx + math.cos(angle)*length*0.06)
        base_y = int(cy + math.sin(angle)*length*0.06)
        w_dist = length * 0.42
        left_x  = int(cx + math.cos(angle)*w_dist + math.cos(perp)*width)
        left_y  = int(cy + math.sin(angle)*w_dist + math.sin(perp)*width)
        right_x = int(cx + math.cos(angle)*w_dist - math.cos(perp)*width)
        right_y = int(cy + math.sin(angle)*w_dist - math.sin(perp)*width)
        mid_x = int(cx + math.cos(angle)*length*0.60)
        mid_y = int(cy + math.sin(angle)*length*0.60)
        facets = [
            ([base_x,base_y],[left_x,left_y],[mid_x,mid_y], mid_color, 0),
            ([base_x,base_y],[right_x,right_y],[mid_x,mid_y],
             tuple(min(255,c+25) for c in mid_color), 0),
            ([left_x,left_y],[tip_x,tip_y],[mid_x,mid_y], tip_color, 0),
            ([right_x,right_y],[tip_x,tip_y],[mid_x,mid_y],
             tuple(min(255,c+20) for c in tip_color), 0),
        ]
        for p1,p2,p3,fc,_ in facets:
            pts = np.array([p1,p2,p3], np.int32).reshape((-1,1,2))
            ov = frame.copy()
            cv2.fillPoly(ov, [pts], fc)
            cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)
        all_pts = [(base_x,base_y),(left_x,left_y),(mid_x,mid_y),(tip_x,tip_y),(right_x,right_y)]
        for a_i,b_i in [(0,1),(1,2),(2,3),(3,4),(4,2),(0,4),(1,3)]:
            cv2.line(frame, all_pts[a_i], all_pts[b_i], (200,200,200), 1)

    outer_angles_deg = [-90,-130,-50,180,0,130,50,-160,-20]
    outer_tip  = (245, 235, 255)
    outer_mid  = (220, 195, 245)
    outer_base = (190, 155, 225)
    l1 = int(radius * 2.8 * breathe)
    for deg in outer_angles_deg:
        angle = math.radians(deg)
        horiz_factor = abs(math.cos(angle))
        w = int(radius * (0.35 + 0.22 * horiz_factor))
        l = int(l1 * (0.85 + 0.20 * horiz_factor))
        draw_faceted_petal(frame, cx, cy, angle, l, w,
                           outer_tip, outer_mid, outer_base, 0.68)

    mid_angles_deg = [-90,-115,-65,-145,-35,160,20,-170,10]
    mid_tip  = (180, 100, 230)
    mid_mid  = (150,  65, 205)
    mid_base = (120,  45, 180)
    l2 = int(radius * 1.95 * breathe)
    for deg in mid_angles_deg:
        angle = math.radians(deg)
        horiz_factor = abs(math.cos(angle))
        w = int(radius * (0.28 + 0.14 * horiz_factor))
        draw_faceted_petal(frame, cx, cy, angle, l2, w,
                           mid_tip, mid_mid, mid_base, 0.75)

    inner_angles_deg = [-90,-112,-68,-135,-45,160,20]
    inner_tip  = (150,  55, 210)
    inner_mid  = (110,  25, 175)
    inner_base = ( 85,  15, 150)
    l3 = int(radius * 1.25 * breathe)
    for deg in inner_angles_deg:
        angle = math.radians(deg)
        horiz_factor = abs(math.cos(angle))
        w = int(radius * (0.20 + 0.08 * horiz_factor))
        draw_faceted_petal(frame, cx, cy, angle, l3, w,
                           inner_tip, inner_mid, inner_base, 0.82)

    ov = frame.copy()
    cv2.circle(ov, (cx,cy), int(radius*0.44), (0,150,245), -1)
    cv2.addWeighted(ov, 0.88, frame, 0.12, 0, frame)
    for i in range(6):
        angle = (2*math.pi/6)*i
        sx = int(cx + math.cos(angle)*radius*0.40)
        sy = int(cy + math.sin(angle)*radius*0.40)
        cv2.line(frame,(cx,cy),(sx,sy),(0,200,255),1)
    ov = frame.copy()
    cv2.circle(ov,(cx,cy),int(radius*0.26),(0,220,255),-1)
    cv2.addWeighted(ov,0.92,frame,0.08,0,frame)
    ov = frame.copy()
    cv2.circle(ov,(cx,cy),int(radius*0.11),(255,255,255),-1)
    cv2.addWeighted(ov,0.95,frame,0.05,0,frame)
    ov = frame.copy()
    cv2.circle(ov,(cx,cy),int(radius*3.2),(190,140,255),-1)
    cv2.addWeighted(ov,0.07,frame,0.93,0,frame)


# ─────────────────────────────────────────────────────
# GROUP 2 GEOMETRY FUNCTIONS
# ─────────────────────────────────────────────────────

def draw_katakamukha_petals(frame, cx, cy, radius, color, now):
    pinch_x = cx
    pinch_y = cy - int(radius * 0.5)
    ov = frame.copy()
    cv2.circle(ov, (pinch_x, pinch_y), int(radius*0.50), color, -1)
    cv2.addWeighted(ov, 0.28, frame, 0.72, 0, frame)
    cv2.circle(frame, (pinch_x, pinch_y), int(radius*0.18), color, 2)

    petal_colors = [
        (160, 130, 255),(190, 155, 255),(210, 185, 255),
        (175, 210, 255),(140, 195, 240),(220, 200, 255),
    ]
    center_color = (0, 220, 255)
    n_flowers = 10
    rng = np.random.RandomState(33)

    for i in range(n_flowers):
        drift_dir  = rng.uniform(-1.0, 1.0)
        fall_spd   = 0.30 + rng.random()*0.40
        petal_long = rng.randint(10, 18)
        petal_short= int(petal_long * 0.55)
        color_idx  = rng.randint(0, len(petal_colors))
        flower_rot = rng.uniform(0, 2*math.pi)
        rot_spd    = rng.uniform(-0.4, 0.4)
        t = (now * fall_spd + i * 0.70) % 1.0
        fall_dist  = int(t * radius * 4.2)
        drift_dist = int(drift_dir * t * radius * 1.6)
        fx = pinch_x + drift_dist
        fy = pinch_y + fall_dist
        if t < 0.15:
            alpha = (t/0.15) * 0.88
        elif t > 0.72:
            alpha = ((1.0-t)/0.28) * 0.88
        else:
            alpha = 0.88
        if alpha < 0.05:
            continue
        current_rot = flower_rot + now * rot_spd
        pc = petal_colors[color_idx]
        n_petals  = 5
        petal_dist = int(petal_long * 0.72)
        for p in range(n_petals):
            petal_angle = current_rot + p*(2*math.pi/n_petals)
            pet_cx = int(fx + math.cos(petal_angle)*petal_dist)
            pet_cy = int(fy + math.sin(petal_angle)*petal_dist)
            angle_deg = int(math.degrees(petal_angle))
            ov = frame.copy()
            cv2.ellipse(ov,(pet_cx,pet_cy),(petal_long,petal_short),angle_deg,0,360,pc,-1)
            cv2.addWeighted(ov,alpha,frame,1-alpha,0,frame)
            darker = tuple(max(0,c-40) for c in pc)
            cv2.ellipse(frame,(pet_cx,pet_cy),(petal_long,petal_short),angle_deg,0,360,darker,1)
        center_r = int(petal_long * 0.38)
        ov = frame.copy()
        cv2.circle(ov,(int(fx),int(fy)),center_r,center_color,-1)
        cv2.addWeighted(ov,alpha*0.95,frame,1-alpha*0.95,0,frame)
        if center_r > 4:
            for d in range(6):
                da = d*(2*math.pi/6)+current_rot
                dx2 = int(fx+math.cos(da)*center_r*0.65)
                dy2 = int(fy+math.sin(da)*center_r*0.65)
                cv2.circle(frame,(dx2,dy2),1,(255,255,255),-1)

    arc_pts = []
    for t in np.linspace(-math.pi*0.55, math.pi*0.55, 35):
        ax = int(pinch_x + math.sin(t)*radius*1.1)
        ay = int(pinch_y + int(radius*0.9) - math.cos(t)*radius*0.25)
        arc_pts.append((ax,ay))
    if len(arc_pts) > 1:
        ov = frame.copy()
        for i in range(1, len(arc_pts)):
            cv2.line(ov, arc_pts[i-1], arc_pts[i], color, 1)
        cv2.addWeighted(ov, 0.40, frame, 0.60, 0, frame)


def draw_mushti_core(frame, cx, cy, radius, color, now):
    pulse_fast  = (math.sin(now * 3.5) + 1) / 2
    pulse_slow  = (math.sin(now * 1.2) + 1) / 2
    combined    = pulse_fast * 0.6 + pulse_slow * 0.4
    core_r  = int(radius * (0.5 + combined * 0.4))
    inner_r = int(radius * (0.25 + combined * 0.20))
    if pulse_fast > 0.85:
        shock_r = int(radius * (1.5 + pulse_fast * 0.8))
        ov = frame.copy()
        cv2.circle(ov, (cx,cy), shock_r, color, 2)
        cv2.addWeighted(ov, 0.60, frame, 0.40, 0, frame)
    glow_layers = [
        (radius * 2.2, 0.08 * combined),(radius * 1.7, 0.15 * combined),
        (radius * 1.3, 0.25 * combined),(radius * 1.0, 0.40 * combined),
        (core_r,       0.60 * combined),(inner_r,      0.80),
    ]
    for r_mult, alpha in glow_layers:
        r = int(r_mult)
        if r < 1: continue
        ov = frame.copy()
        cv2.circle(ov, (cx,cy), r, color, -1)
        cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)
    if combined > 0.6:
        white_r = int(radius * 0.12 * combined)
        ov = frame.copy()
        cv2.circle(ov, (cx,cy), white_r, (255,255,255), -1)
        cv2.addWeighted(ov, combined * 0.90, frame, 1-combined*0.90, 0, frame)
    if pulse_fast > 0.70:
        n_cracks = 8
        crack_len = int(radius * (1.5 + combined))
        for i in range(n_cracks):
            angle = (2*math.pi/n_cracks)*i
            ex = int(cx + math.cos(angle)*crack_len)
            ey = int(cy + math.sin(angle)*crack_len)
            ov = frame.copy()
            cv2.line(ov, (cx,cy), (ex,ey), color, 1)
            cv2.addWeighted(ov, 0.45 * pulse_fast, frame, 1-0.45*pulse_fast, 0, frame)


def draw_shikhara_pillar(frame, cx, cy, radius, color, now):
    pillar_h = int(radius * 3.8)
    top_y    = cy - pillar_h
    pulse    = (math.sin(now * 1.5) + 1) / 2
    for w, alpha in [(20,0.08),(14,0.14),(8,0.25),(4,0.55)]:
        ov = frame.copy()
        cv2.rectangle(ov, (cx - w, top_y), (cx + w, cy), color, -1)
        cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)
    ov = frame.copy()
    cv2.line(ov, (cx, top_y), (cx, cy), color, 2)
    cv2.addWeighted(ov, 0.85, frame, 0.15, 0, frame)
    base_r = int(radius * (0.7 + pulse*0.3))
    ov = frame.copy()
    cv2.circle(ov, (cx,cy), base_r, color, -1)
    cv2.addWeighted(ov, 0.25 + pulse*0.15, frame, 1-(0.25+pulse*0.15), 0, frame)
    ov = frame.copy()
    cv2.circle(ov, (cx, top_y), int(radius*0.3), color, -1)
    cv2.addWeighted(ov, 0.60, frame, 0.40, 0, frame)
    if int(now*20) % 2 == 0:
        sy = np.random.randint(top_y, cy)
        sx = cx + np.random.randint(-6, 6)
        cv2.circle(frame, (sx,sy), 2, (255,255,255), -1)


def draw_trishula_flames(frame, cx, cy, radius, color, now,
                          tip1=None, tip2=None, tip3=None):
    trident_color = (30, 180, 220)
    shaft_color   = (40, 200, 230)
    glow_color    = (20, 140, 180)
    shaft_len   = int(radius * 2.5)
    prong_h     = int(radius * 1.8)
    prong_gap   = int(radius * 0.55)
    cross_y     = cy - int(radius * 0.6)
    shaft_bot = cy + int(radius * 0.8)
    shaft_top = cy - int(radius * 0.5)
    for w, alpha in [(8,0.12),(4,0.25),(2,0.60)]:
        ov = frame.copy()
        cv2.line(ov, (cx, shaft_bot), (cx, shaft_bot - shaft_len), shaft_color, w)
        cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)
    prong_bases = [(cx - prong_gap, shaft_top),(cx, shaft_top),(cx + prong_gap, shaft_top)]
    prong_tips  = [(cx - prong_gap, shaft_top - prong_h),
                   (cx, shaft_top - prong_h - int(radius*0.4)),
                   (cx + prong_gap, shaft_top - prong_h)]
    for base, tip in zip(prong_bases, prong_tips):
        for w, alpha in [(6,0.15),(3,0.30),(1,0.75)]:
            ov = frame.copy()
            cv2.line(ov, base, tip, trident_color, w)
            cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)
    for i, (base, tip) in enumerate(zip(prong_bases, prong_tips)):
        if i != 1:
            hook_dir = -1 if i == 0 else 1
            hook_x = tip[0] + hook_dir*int(radius*0.25)
            hook_y = tip[1] + int(radius*0.20)
            ov = frame.copy()
            cv2.line(ov, tip, (hook_x, hook_y), trident_color, 2)
            cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
    ov = frame.copy()
    cv2.line(ov, (prong_bases[0][0], cross_y), (prong_bases[2][0], cross_y), trident_color, 2)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
    cross_y2 = cross_y + int(radius*0.3)
    ov = frame.copy()
    cv2.line(ov, (prong_bases[0][0]-int(radius*0.1), cross_y2),
             (prong_bases[2][0]+int(radius*0.1), cross_y2), trident_color, 1)
    cv2.addWeighted(ov, 0.40, frame, 0.60, 0, frame)
    flame_colors = [(0,40,180),(0,80,220),(20,130,240),(40,170,245),(60,200,255)]
    for i, tip in enumerate(prong_tips):
        tip_x, tip_y = tip
        phase = i * 1.3
        f_h = int(radius * (1.2 if i==1 else 0.9))
        f_w = int(radius * 0.22)
        for seg in range(6):
            t      = seg / 6
            t_next = (seg+1) / 6
            y1 = int(tip_y - t * f_h)
            y2 = int(tip_y - t_next * f_h)
            w1 = int(f_w * (1.0-t))
            w2 = int(f_w * (1.0-t_next))
            wb2 = int(math.sin(now*5+t_next*4+phase)*f_w*0.35)
            seg_pts = np.array([[tip_x-w1,y1],[tip_x+w1,y1],[tip_x+w2+wb2,y2],[tip_x-w2+wb2,y2]],
                               np.int32).reshape((-1,1,2))
            fc_idx = min(seg, len(flame_colors)-1)
            fc = flame_colors[fc_idx]
            alpha = (1.0-t) * 0.65
            ov = frame.copy()
            cv2.fillPoly(ov, [seg_pts], fc)
            cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)
        ov = frame.copy()
        cv2.circle(ov, (tip_x, tip_y), int(radius*0.20), glow_color, -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
    ov = frame.copy()
    cv2.circle(ov, (cx, shaft_bot), int(radius*0.45), glow_color, -1)
    cv2.addWeighted(ov, 0.25, frame, 0.75, 0, frame)


def draw_tamrachuda_crest(frame, cx, cy, radius, color, now):
    pulse = (math.sin(now*2.5)+1)/2
    body_r = int(radius * 1.0)
    ov = frame.copy()
    cv2.circle(ov, (cx,cy), body_r, color, -1)
    cv2.addWeighted(ov, 0.20, frame, 0.80, 0, frame)
    beak_color  = (0, 80, 220)
    beak_length = int(radius * 2.0)
    beak_width  = int(radius * 0.45)
    beak_tip_x  = cx + beak_length
    beak_tip_y  = cy - int(radius * 0.1)
    beak_top_x  = cx + int(radius * 0.3)
    beak_top_y  = cy - beak_width
    beak_bot_x  = cx + int(radius * 0.3)
    beak_bot_y  = cy + int(beak_width * 0.5)
    beak_pts = np.array([[beak_top_x,beak_top_y],[beak_tip_x,beak_tip_y],[beak_bot_x,beak_bot_y]],
                         np.int32).reshape((-1,1,2))
    ov = frame.copy()
    cv2.fillPoly(ov, [beak_pts], beak_color)
    cv2.addWeighted(ov, 0.85, frame, 0.15, 0, frame)
    cv2.polylines(frame, [beak_pts], True, (0,120,255), 1)
    eye_x = cx + int(radius * 0.55)
    eye_y = cy - int(radius * 0.45)
    ov = frame.copy()
    cv2.circle(ov, (eye_x, eye_y), int(radius*0.18), (255,255,255), -1)
    cv2.addWeighted(ov, 0.90, frame, 0.10, 0, frame)
    cv2.circle(frame, (eye_x, eye_y), int(radius*0.09), (0,0,0), -1)
    comb_color = (0, 60, 220)
    comb_base_y = cy - int(radius * 0.6)
    comb_x_positions = [cx+int(radius*0.4), cx+int(radius*0.15), cx-int(radius*0.1)]
    comb_heights = [int(radius*1.1), int(radius*0.85), int(radius*0.65)]
    for i, (comb_x, comb_h) in enumerate(zip(comb_x_positions, comb_heights)):
        wobble = int(math.sin(now*3+i)*radius*0.05)
        comb_tip_y = comb_base_y - comb_h + wobble
        comb_pts = np.array([
            [comb_x-int(radius*0.12), comb_base_y],
            [comb_x, comb_tip_y],
            [comb_x+int(radius*0.12), comb_base_y],
        ], np.int32).reshape((-1,1,2))
        ov = frame.copy()
        cv2.fillPoly(ov, [comb_pts], comb_color)
        cv2.addWeighted(ov, 0.80, frame, 0.20, 0, frame)
    tail_colors = [(0,70,200),(0,110,230),(20,160,240),(40,190,220),(60,210,190)]
    n_tail = 8
    for i in range(n_tail):
        t = i / (n_tail-1)
        angle = math.radians(150 + t*90)
        center_t = 1.0 - abs(t-0.5)*2
        t_len = int(radius*(1.5 + center_t*1.0))
        wave = math.sin(now*1.5+i*0.6)*0.05
        angle += wave
        tip_x2 = int(cx + math.cos(angle)*t_len)
        tip_y2 = int(cy + math.sin(angle)*t_len)
        ctrl_x = int(cx+math.cos(angle)*t_len*0.5+math.cos(angle+math.pi/2)*radius*0.25)
        ctrl_y = int(cy+math.sin(angle)*t_len*0.5+math.sin(angle+math.pi/2)*radius*0.25)
        f_pts = []
        for s in range(18):
            st = s/17
            bx2 = int((1-st)**2*cx+2*(1-st)*st*ctrl_x+st**2*tip_x2)
            by2 = int((1-st)**2*cy+2*(1-st)*st*ctrl_y+st**2*tip_y2)
            f_pts.append((bx2,by2))
        fc = tail_colors[int(t*(len(tail_colors)-1))]
        alpha = 0.55 + center_t*0.25
        ov = frame.copy()
        for j in range(1, len(f_pts)):
            fade = 1.0 - j/len(f_pts)
            fc_f = tuple(int(c*fade) for c in fc)
            cv2.line(ov, f_pts[j-1], f_pts[j], fc_f, 2)
        cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)
    wattle_pts = np.array([
        [cx+int(radius*0.35),cy+int(radius*0.05)],
        [cx+int(radius*0.25),cy+int(radius*0.55)],
        [cx+int(radius*0.55),cy+int(radius*0.25)],
    ], np.int32).reshape((-1,1,2))
    ov = frame.copy()
    cv2.fillPoly(ov, [wattle_pts], beak_color)
    cv2.addWeighted(ov, 0.70, frame, 0.30, 0, frame)


def draw_kartarimukha_lightning(frame, cx, cy, radius, color, now):
    def jagged_bolt(start_x, start_y, end_x, end_y, jag, n_segs, seed):
        rng = np.random.RandomState(seed)
        pts = [(start_x, start_y)]
        for i in range(1, n_segs):
            t  = i / n_segs
            mx2 = int(start_x + (end_x-start_x)*t + rng.randint(-jag, jag))
            my2 = int(start_y + (end_y-start_y)*t + rng.randint(-jag, jag))
            pts.append((mx2, my2))
        pts.append((end_x, end_y))
        return pts

    flicker_seed = int(now * 10) % 8
    spread = int(radius * 2.5)
    bolt_len = int(radius * 3.0)
    l_end_x = cx - spread
    l_end_y = cy - bolt_len
    r_end_x = cx + spread
    r_end_y = cy - bolt_len
    jag_amount = int(radius * 0.25)
    for bolt_end, seed_offset in [((l_end_x,l_end_y),0),((r_end_x,r_end_y),4)]:
        bolt_pts = jagged_bolt(cx, cy, bolt_end[0], bolt_end[1],
                               jag_amount, 12, flicker_seed + seed_offset)
        ov = frame.copy()
        for i in range(1, len(bolt_pts)):
            cv2.line(ov, bolt_pts[i-1], bolt_pts[i], color, 5)
        cv2.addWeighted(ov, 0.25, frame, 0.75, 0, frame)
        ov = frame.copy()
        for i in range(1, len(bolt_pts)):
            cv2.line(ov, bolt_pts[i-1], bolt_pts[i], color, 2)
        cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
        for i in range(1, len(bolt_pts)):
            cv2.line(frame, bolt_pts[i-1], bolt_pts[i], (255,255,255), 1)
        ov = frame.copy()
        cv2.circle(ov, bolt_end, int(radius*0.2), color, -1)
        cv2.addWeighted(ov, 0.70, frame, 0.30, 0, frame)
    flash = (math.sin(now * 8) + 1) / 2
    ov = frame.copy()
    cv2.circle(ov, (cx,cy), int(radius*(0.4+flash*0.3)), color, -1)
    cv2.addWeighted(ov, 0.35+flash*0.25, frame, 1-(0.35+flash*0.25), 0, frame)


# ─────────────────────────────────────────────────────
# GROUP 3 GEOMETRY FUNCTIONS
# ─────────────────────────────────────────────────────

def draw_arala_swirl(frame, cx, cy, radius, color, now,
                     index_tip_x=None, index_tip_y=None):
    if index_tip_x is None:
        index_tip_x = cx - int(radius*0.3)
        index_tip_y = cy - int(radius*1.2)
    swirl_cx = index_tip_x
    swirl_cy = index_tip_y
    swirl_r  = int(radius * 2.2)
    h, w = frame.shape[:2]
    x1 = max(0, swirl_cx - swirl_r)
    y1 = max(0, swirl_cy - swirl_r)
    x2 = min(w, swirl_cx + swirl_r)
    y2 = min(h, swirl_cy + swirl_r)
    if x2 > x1 and y2 > y1:
        region = frame[y1:y2, x1:x2].copy()
        rh, rw = region.shape[:2]
        if rw > 4 and rh > 4:
            rcx = swirl_cx - x1
            rcy = swirl_cy - y1
            swirl_angle = (now * 45) % 360
            M = cv2.getRotationMatrix2D((rcx, rcy), swirl_angle, 1.0)
            swirled = cv2.warpAffine(region, M, (rw, rh),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            mask = np.zeros((rh, rw), dtype=np.float32)
            cv2.circle(mask, (rcx, rcy), min(swirl_r, rw//2, rh//2), 1.0, -1)
            mask = cv2.GaussianBlur(mask, (21,21), 0)
            for c in range(3):
                frame[y1:y2, x1:x2, c] = (
                    swirled[:,:,c] * mask + frame[y1:y2, x1:x2, c] * (1-mask)
                ).astype(np.uint8)
    n_spirals = 3
    for s in range(n_spirals):
        phase = s * (2*math.pi/n_spirals)
        pts = []
        for t in range(60):
            angle = (t/60) * 4*math.pi + now*2.0 + phase
            r = (t/60) * swirl_r * 0.85
            px2 = int(swirl_cx + math.cos(angle)*r)
            py2 = int(swirl_cy + math.sin(angle)*r)
            h2, w2 = frame.shape[:2]
            if 0 <= px2 < w2 and 0 <= py2 < h2:
                pts.append((px2,py2))
        if len(pts) > 1:
            ov = frame.copy()
            for i in range(1, len(pts)):
                cv2.line(ov, pts[i-1], pts[i], color, 1)
            cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
    ov = frame.copy()
    cv2.circle(ov, (swirl_cx, swirl_cy), int(radius*0.28), color, -1)
    cv2.addWeighted(ov, 0.60, frame, 0.40, 0, frame)
    cv2.circle(frame, (swirl_cx, swirl_cy), int(radius*0.12), (255,255,255), -1)


def draw_shukatunda_arrow(frame, cx, cy, radius, color, now,
                           index_tip_x=None, index_tip_y=None,
                           index_base_x=None, index_base_y=None):
    if index_tip_x is None:
        index_tip_x = cx
        index_tip_y = cy - int(radius*1.5)
        index_base_x = cx
        index_base_y = cy
    dx = index_tip_x - index_base_x
    dy = index_tip_y - index_base_y
    length = math.sqrt(dx*dx + dy*dy) + 1
    ux = dx/length
    uy = dy/length
    arrow_start_x = index_tip_x
    arrow_start_y = index_tip_y
    arrow_len = int(radius * 5.0)
    for w, alpha in [(12,0.08),(7,0.15),(3,0.35),(1,0.90)]:
        end_x = int(arrow_start_x + ux*arrow_len)
        end_y = int(arrow_start_y + uy*arrow_len)
        ov = frame.copy()
        cv2.line(ov, (arrow_start_x, arrow_start_y), (end_x, end_y), color, w)
        cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)
    tip_x2 = int(arrow_start_x + ux*arrow_len)
    tip_y2 = int(arrow_start_y + uy*arrow_len)
    perp_x = -uy
    perp_y =  ux
    wing   = int(radius*0.5)
    back   = int(radius*0.6)
    arrow_pts = np.array([
        [tip_x2, tip_y2],
        [int(tip_x2-ux*back+perp_x*wing), int(tip_y2-uy*back+perp_y*wing)],
        [int(tip_x2-ux*back*0.5),         int(tip_y2-uy*back*0.5)],
        [int(tip_x2-ux*back-perp_x*wing), int(tip_y2-uy*back-perp_y*wing)],
    ], np.int32).reshape((-1,1,2))
    ov = frame.copy()
    cv2.fillPoly(ov, [arrow_pts], color)
    cv2.addWeighted(ov, 0.85, frame, 0.15, 0, frame)
    n_ghosts = 3
    for g in range(n_ghosts):
        ghost_offset = (g+1) * int(radius*1.2)
        g_start_x = int(arrow_start_x - ux*ghost_offset)
        g_start_y = int(arrow_start_y - uy*ghost_offset)
        g_end_x   = int(tip_x2 - ux*ghost_offset)
        g_end_y   = int(tip_y2 - uy*ghost_offset)
        ghost_alpha = 0.25 - g*0.07
        ov = frame.copy()
        cv2.line(ov, (g_start_x,g_start_y), (g_end_x,g_end_y), color, 2)
        cv2.addWeighted(ov, ghost_alpha, frame, 1-ghost_alpha, 0, frame)
    ov = frame.copy()
    cv2.circle(ov, (arrow_start_x, arrow_start_y), int(radius*0.30), color, -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)


def draw_chandrakala_moon(frame, cx, cy, radius, color, now,
                           thumb_x=None, thumb_y=None,
                           index_x=None, index_y=None):
    if thumb_x is not None and index_x is not None:
        c_dist = math.sqrt((thumb_x-index_x)**2 + (thumb_y-index_y)**2)
        c_norm = min(c_dist / (radius*1.5), 1.0)
    else:
        c_norm = 0.5
    moon_r = int(radius*(0.8 + c_norm*1.6))
    inner_r= int(moon_r*0.75)
    shift  = int(moon_r*0.72)
    if c_norm > 0.3:
        bright_alpha = (c_norm-0.3)/0.7 * 0.18
        ov = frame.copy()
        cv2.rectangle(ov, (0,0), (frame.shape[1],frame.shape[0]), (255,255,255), -1)
        cv2.addWeighted(ov, bright_alpha, frame, 1-bright_alpha, 0, frame)
    ov = frame.copy()
    cv2.circle(ov, (cx,cy), int(moon_r*1.4), color, -1)
    cv2.addWeighted(ov, 0.08, frame, 0.92, 0, frame)
    ov = frame.copy()
    cv2.circle(ov, (cx,cy), moon_r, color, -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
    ov = frame.copy()
    cv2.circle(ov, (cx+shift, cy), inner_r, (0,0,0), -1)
    cv2.addWeighted(ov, 0.90, frame, 0.10, 0, frame)
    cv2.circle(frame, (cx,cy), moon_r, color, 2)
    rng = np.random.RandomState(99)
    for _ in range(8):
        sx = cx + rng.randint(-int(moon_r*2.5), int(moon_r*2.5))
        sy = cy + rng.randint(-int(moon_r*2.5), int(moon_r*2.5))
        if math.sqrt((sx-cx)**2+(sy-cy)**2) > moon_r*1.2:
            star_size = rng.randint(1,3)
            bright = 0.4 + c_norm*0.5
            ov = frame.copy()
            cv2.circle(ov,(sx,sy),star_size*2,color,-1)
            cv2.addWeighted(ov,bright*0.3,frame,1-bright*0.3,0,frame)
            cv2.circle(frame,(sx,sy),star_size,(255,255,255),-1)
    for ta in [math.radians(150), math.radians(210)]:
        tx2 = int(cx + math.cos(ta)*moon_r)
        ty2 = int(cy + math.sin(ta)*moon_r)
        cv2.circle(frame,(tx2,ty2),3,color,-1)
        ov = frame.copy()
        cv2.circle(ov,(tx2,ty2),8,color,-1)
        cv2.addWeighted(ov,0.40,frame,0.60,0,frame)


def draw_hamsasya_flame(frame, cx, cy, radius, color, now,
                         pinch_x=None, pinch_y=None, hand_speed=0.0):
    if pinch_x is None:
        pinch_x = cx
        pinch_y = cy - int(radius*0.8)
    flicker_amt = min(hand_speed/15.0, 1.0)
    flicker = (math.sin(now*8.0)*0.15 + math.sin(now*13.0)*0.10 +
               flicker_amt*math.sin(now*20)*0.15)
    flame_h = int(radius*(1.6 + flicker*0.4))
    flame_w = int(radius*0.40)
    diya_color = (40, 160, 220)
    diya_pts = np.array([
        [pinch_x-int(radius*0.4), pinch_y+int(radius*0.15)],
        [pinch_x+int(radius*0.4), pinch_y+int(radius*0.15)],
        [pinch_x+int(radius*0.25), pinch_y+int(radius*0.40)],
        [pinch_x-int(radius*0.25), pinch_y+int(radius*0.40)],
    ], np.int32).reshape((-1,1,2))
    ov = frame.copy()
    cv2.fillPoly(ov, [diya_pts], diya_color)
    cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)
    cv2.polylines(frame,[diya_pts],True,(60,200,240),1)
    flame_colors_list = [(0,80,200),(0,140,240),(20,200,255),(60,230,255),(120,245,255)]
    for seg in range(8):
        t      = seg/8
        t_next = (seg+1)/8
        y1 = int(pinch_y - t*flame_h)
        y2 = int(pinch_y - t_next*flame_h)
        w1 = int(flame_w*(1.0-t*0.7))
        w2 = int(flame_w*(1.0-t_next*0.7))
        wb2 = int(math.sin(now*6+t_next*5)*flame_w*0.30*(1+flicker_amt))
        seg_pts = np.array([
            [pinch_x-w1,y1],[pinch_x+w1,y1],
            [pinch_x+w2+wb2,y2],[pinch_x-w2+wb2,y2],
        ], np.int32).reshape((-1,1,2))
        fc_idx = min(int(t*len(flame_colors_list)), len(flame_colors_list)-1)
        fc = flame_colors_list[fc_idx]
        alpha = (1.0-t)*0.75
        ov = frame.copy()
        cv2.fillPoly(ov,[seg_pts],fc)
        cv2.addWeighted(ov,alpha,frame,1-alpha,0,frame)
    tip_x2 = pinch_x + int(math.sin(now*6)*flame_w*0.25)
    tip_y2 = pinch_y - flame_h
    cv2.circle(frame,(tip_x2,tip_y2),int(radius*0.08),(255,255,255),-1)
    ov = frame.copy()
    cv2.circle(ov,(pinch_x,pinch_y),int(radius*0.7),(20,180,255),-1)
    cv2.addWeighted(ov,0.20,frame,0.80,0,frame)


def draw_kapittha_coins(frame, cx, cy, radius, color, now):
    emit_x = cx
    emit_y = cy - int(radius*0.3)
    ov = frame.copy()
    cv2.circle(ov,(emit_x,emit_y),int(radius*0.55),color,-1)
    cv2.addWeighted(ov,0.20,frame,0.80,0,frame)
    gold_face   = (0,   195, 255)
    gold_deep   = (0,   155, 210)
    gold_rim    = (0,   130, 190)
    gold_shine  = (120, 235, 255)
    gold_shadow = (0,   100, 160)
    n_coins = 14
    rng = np.random.RandomState(77)
    for i in range(n_coins):
        drift    = rng.uniform(-1.1, 1.1)
        fall_spd = 0.28 + rng.random()*0.42
        coin_r   = rng.randint(9, 17)
        spin_spd = rng.uniform(1.5, 4.0)
        spin_dir = 1 if rng.random()>0.5 else -1
        t = (now*fall_spd + i*0.58) % 1.0
        fall_d  = int(t*radius*4.8)
        drift_d = int(drift*t*radius*1.4)
        coin_x  = int(emit_x + drift_d)
        coin_y  = int(emit_y + fall_d)
        if t < 0.12:    alpha = t/0.12
        elif t > 0.75:  alpha = (1.0-t)/0.25
        else:           alpha = 1.0
        alpha *= 0.88
        if alpha < 0.05: continue
        spin_angle = now*spin_spd*spin_dir + i*1.1
        cos_spin = math.cos(spin_angle)
        face_w = max(2, int(coin_r*abs(cos_spin)))
        face_h = coin_r
        showing_face = cos_spin > 0
        ov = frame.copy()
        cv2.ellipse(ov,(coin_x,coin_y),(face_w+2,face_h+2),0,0,360,gold_shadow,-1)
        cv2.addWeighted(ov,alpha,frame,1-alpha,0,frame)
        face_color = gold_face if showing_face else gold_deep
        ov = frame.copy()
        cv2.ellipse(ov,(coin_x,coin_y),(face_w,face_h),0,0,360,face_color,-1)
        cv2.addWeighted(ov,alpha,frame,1-alpha,0,frame)
        cv2.ellipse(frame,(coin_x,coin_y),(face_w,face_h),0,0,360,gold_rim,1)
        if face_w > 5 and showing_face:
            shine_x = coin_x - int(face_w*0.25)
            shine_y = coin_y - int(face_h*0.30)
            shine_w = max(1,int(face_w*0.45))
            shine_h = max(1,int(face_h*0.22))
            ov = frame.copy()
            cv2.ellipse(ov,(shine_x,shine_y),(shine_w,shine_h),0,0,360,gold_shine,-1)
            cv2.addWeighted(ov,alpha*0.70,frame,1-alpha*0.70,0,frame)
        if face_w > 6 and showing_face and alpha>0.4:
            cv2.circle(frame,(coin_x,coin_y),max(1,face_w//3),gold_rim,1)
            cv2.circle(frame,(coin_x,coin_y),max(1,face_w//6),gold_shine,-1)
        if abs(cos_spin) < 0.25:
            ov = frame.copy()
            cv2.line(ov,(coin_x,coin_y-face_h),(coin_x,coin_y+face_h),gold_shine,2)
            cv2.addWeighted(ov,alpha*0.80,frame,1-alpha*0.80,0,frame)
    if int(now*6)%2 == 0:
        for _ in range(5):
            angle2 = np.random.uniform(0,2*math.pi)
            sr = np.random.randint(int(radius*0.2),int(radius*0.9))
            sx2 = int(emit_x+math.cos(angle2)*sr)
            sy2 = int(emit_y+math.sin(angle2)*sr)
            cv2.circle(frame,(sx2,sy2),2,(255,255,255),-1)
            cv2.line(frame,(sx2-3,sy2),(sx2+3,sy2),gold_shine,1)
            cv2.line(frame,(sx2,sy2-3),(sx2,sy2+3),gold_shine,1)


# ─────────────────────────────────────────────────────
# GROUP 4 — NATURE GEOMETRY FUNCTIONS (REWRITTEN)
# Each anchored to its most characteristic fingertip.
# Dynamic: reacts to hand movement via `now`.
# ─────────────────────────────────────────────────────

def draw_mayura_peacock(frame, cx, cy, radius, color, now,
                        thumb_tip_x=None, thumb_tip_y=None):
    """
    Mayura (Peacock) — full dense peacock fan from THUMB TIP.
    5 rows of feathers with green leaves and geometric mandala eyespots.
    Anchor: thumb tip (lm 4). Colors: green leaves, geometric eyespots, navy body.
    """
    if thumb_tip_x is None:
        thumb_tip_x = cx - int(radius * 0.6)
        thumb_tip_y = cy - int(radius * 0.4)
    ax, ay = thumb_tip_x, thumb_tip_y
    breathe = 1.0 + 0.03 * math.sin(now * 1.1)

    def _draw_feather_leaves(frame, bx, by, angle, flen, row_idx):
        """Draw two green leaf shapes flanking the feather center line."""
        perp = angle + math.pi / 2
        tip_x = int(bx + math.cos(angle) * flen)
        tip_y = int(by + math.sin(angle) * flen)
        
        # Leaf width scales with row: outer rows wider, inner rows narrower
        leaf_width_scale = 1.0 - (row_idx * 0.15)
        base_width = flen * 0.25 * leaf_width_scale
        
        # Left leaf (3-point polygon)
        left_base_x = int(bx + math.cos(perp) * base_width + math.cos(angle) * flen * 0.3)
        left_base_y = int(by + math.sin(perp) * base_width + math.sin(angle) * flen * 0.3)
        left_pts = np.array([[bx,by], [left_base_x, left_base_y], [tip_x,tip_y]], np.int32)
        
        # Right leaf (3-point polygon)
        right_base_x = int(bx - math.cos(perp) * base_width * 0.8 + math.cos(angle) * flen * 0.3)
        right_base_y = int(by - math.sin(perp) * base_width * 0.8 + math.sin(angle) * flen * 0.3)
        right_pts = np.array([[bx,by], [right_base_x, right_base_y], [tip_x,tip_y]], np.int32)
        
        # Third triangle mid facet
        mid_base_x = int(bx + math.cos(angle) * flen * 0.2)
        mid_base_y = int(by + math.sin(angle) * flen * 0.2)
        mid_pts = np.array([[bx,by], [mid_base_x, mid_base_y], [tip_x,tip_y]], np.int32)
        
        # Draw leaves with facets
        dark_green = (30, 120, 30)
        bright_green = (55, 185, 55)
        mid_green = (42, 152, 42)
        outline_color = (20, 100, 20)
        vein_color = (15, 80, 15)
        
        # Shadow facet
        ov = frame.copy()
        cv2.fillPoly(ov, [left_pts.reshape(-1,1,2)], dark_green)
        cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)
        
        # Lit facet
        ov = frame.copy()
        cv2.fillPoly(ov, [right_pts.reshape(-1,1,2)], bright_green)
        cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)
        
        # Mid facet
        ov = frame.copy()
        cv2.fillPoly(ov, [mid_pts.reshape(-1,1,2)], mid_green)
        cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
        
        # Thin polyline outline
        cv2.polylines(frame, [left_pts.reshape(-1,1,2)], True, outline_color, 1)
        cv2.polylines(frame, [right_pts.reshape(-1,1,2)], True, outline_color, 1)
        
        # Center vein lines
        vein_left_x = int(bx + math.cos(perp) * base_width * 0.5 + math.cos(angle) * flen * 0.15)
        vein_left_y = int(by + math.sin(perp) * base_width * 0.5 + math.sin(angle) * flen * 0.15)
        vein_right_x = int(bx - math.cos(perp) * base_width * 0.4 + math.cos(angle) * flen * 0.15)
        vein_right_y = int(by - math.sin(perp) * base_width * 0.4 + math.sin(angle) * flen * 0.15)
        
        # Create overlay for vein lines with alpha
        ov = frame.copy()
        cv2.line(ov, (bx,by), (vein_left_x, vein_left_y), vein_color, 1)
        cv2.line(ov, (bx,by), (vein_right_x, vein_right_y), vein_color, 1)
        cv2.addWeighted(ov, 0.40, frame, 0.60, 0, frame)

    def _draw_mandala_eyespot(frame, ex, ey, eye_r, now):
        """Draw geometric mandala eyespot with concentric layered polygons."""
        base_rotation = now * 0.3
        
        def _polygon_points(cx, cy, r, n_sides, rotation_offset=0):
            points = []
            for i in range(n_sides):
                angle = (2 * math.pi / n_sides) * i + rotation_offset
                px = int(cx + math.cos(angle) * r)
                py = int(cy + math.sin(angle) * r)
                points.append([px, py])
            return np.array(points, np.int32)
        
        # Layer 1 — Crown spikes: 6 elongated diamond points
        crown_points = []
        for i in range(6):
            angle = (math.pi / 3) * i + base_rotation
            tip_x = int(ex + math.cos(angle) * eye_r * 1.0)
            tip_y = int(ey + math.sin(angle) * eye_r * 1.0)
            base_x1 = int(ex + math.cos(angle - 0.1) * eye_r * 0.18)
            base_y1 = int(ey + math.sin(angle - 0.1) * eye_r * 0.18)
            base_x2 = int(ex + math.cos(angle + 0.1) * eye_r * 0.18)
            base_y2 = int(ey + math.sin(angle + 0.1) * eye_r * 0.18)
            crown_points.append([[base_x1, base_y1], [tip_x, tip_y], [base_x2, base_y2]])
        
        for crown_pt in crown_points:
            pts = np.array(crown_pt, np.int32)
            ov = frame.copy()
            cv2.fillPoly(ov, [pts.reshape(-1,1,2)], (240, 230, 180))
            cv2.addWeighted(ov, 0.50, frame, 0.50, 0, frame)
        
        # Layer 2 — Outer 8-gon
        oct_outer = _polygon_points(ex, ey, int(eye_r * 0.78), 8, base_rotation)
        ov = frame.copy()
        cv2.fillPoly(ov, [oct_outer.reshape(-1,1,2)], (210, 220, 50))
        cv2.addWeighted(ov, 0.58, frame, 0.42, 0, frame)
        # White outline
        ov2 = frame.copy()
        cv2.polylines(ov2, [oct_outer.reshape(-1,1,2)], True, (255,255,255), 1)
        cv2.addWeighted(ov2, 0.25, frame, 0.75, 0, frame)
        
        # Layer 3 — Second 8-gon rotated 22.5°
        oct_mid = _polygon_points(ex, ey, int(eye_r * 0.62), 8, base_rotation + math.pi/8)
        ov = frame.copy()
        cv2.fillPoly(ov, [oct_mid.reshape(-1,1,2)], (190, 205, 25))
        cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
        ov2 = frame.copy()
        cv2.polylines(ov2, [oct_mid.reshape(-1,1,2)], True, (255,255,255), 1)
        cv2.addWeighted(ov2, 0.25, frame, 0.75, 0, frame)
        
        # Layer 4 — Orange hexagon
        hex_outer = _polygon_points(ex, ey, int(eye_r * 0.46), 6, base_rotation)
        ov = frame.copy()
        cv2.fillPoly(ov, [hex_outer.reshape(-1,1,2)], (0, 155, 255))
        cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)
        ov2 = frame.copy()
        cv2.polylines(ov2, [hex_outer.reshape(-1,1,2)], True, (255,255,255), 1)
        cv2.addWeighted(ov2, 0.25, frame, 0.75, 0, frame)
        
        # Layer 5 — Cyan hexagon rotated 30°
        hex_inner = _polygon_points(ex, ey, int(eye_r * 0.30), 6, base_rotation + math.pi/6)
        ov = frame.copy()
        cv2.fillPoly(ov, [hex_inner.reshape(-1,1,2)], (220, 215, 15))
        cv2.addWeighted(ov, 0.82, frame, 0.18, 0, frame)
        ov2 = frame.copy()
        cv2.polylines(ov2, [hex_inner.reshape(-1,1,2)], True, (255,255,255), 1)
        cv2.addWeighted(ov2, 0.25, frame, 0.75, 0, frame)
        
        # Layer 6 — Deep blue diamond
        diamond = _polygon_points(ex, ey, int(eye_r * 0.17), 4, base_rotation + math.pi/4)
        ov = frame.copy()
        cv2.fillPoly(ov, [diamond.reshape(-1,1,2)], (170, 35, 15))
        cv2.addWeighted(ov, 0.92, frame, 0.08, 0, frame)
        ov2 = frame.copy()
        cv2.polylines(ov2, [diamond.reshape(-1,1,2)], True, (255,255,255), 1)
        cv2.addWeighted(ov2, 0.25, frame, 0.75, 0, frame)
        
        # Layer 7 — Bright white center dot
        ov = frame.copy()
        cv2.circle(ov, (ex, ey), int(eye_r * 0.07), (255,255,255), -1)
        cv2.addWeighted(ov, 0.95, frame, 0.05, 0, frame)

    # Feather rows configuration - 5 rows with specified parameters
    rows = [
        (15, int(radius * 3.0 * breathe), int(radius * 0.75), -175, -5),   # Row 1 (outermost)
        (13, int(radius * 2.4 * breathe), int(radius * 0.65), -168, -12),  # Row 2
        (11, int(radius * 1.85 * breathe), int(radius * 0.55), -160, -20), # Row 3
        (9,  int(radius * 1.30 * breathe), int(radius * 0.44), -150, -30), # Row 4
        (7,  int(radius * 0.85 * breathe), int(radius * 0.32), -140, -40), # Row 5 (innermost)
    ]

    # Draw feathers and connecting lines
    for row_idx, (n_feathers, distance, feather_length, angle_start, angle_end) in enumerate(rows):
        for i in range(n_feathers):
            t = i / max(n_feathers - 1, 1)
            sway = math.sin(now * 0.8 + i * 0.45 + row_idx * 1.1) * 0.028
            angle = math.radians(angle_start + t * (angle_end - angle_start)) + sway
            
            # Feather base position
            feather_x = int(ax + math.cos(angle) * distance)
            feather_y = int(ay + math.sin(angle) * distance)
            
            # Draw connecting line from anchor to feather base
            ov = frame.copy()
            cv2.line(ov, (ax, ay), (feather_x, feather_y), (25, 90, 25), 1)
            cv2.addWeighted(ov, 0.20, frame, 0.80, 0, frame)
            
            # Draw green leaves
            _draw_feather_leaves(frame, feather_x, feather_y, angle, feather_length, row_idx)
            
            # Draw mandala eyespot at feather tip
            eyespot_x = int(feather_x + math.cos(angle) * feather_length * 0.82)
            eyespot_y = int(feather_y + math.sin(angle) * feather_length * 0.82)
            eye_radius = int(feather_length * 0.30)
            _draw_mandala_eyespot(frame, eyespot_x, eyespot_y, eye_radius, now)

    # Body at anchor - elongated navy diamond with 6 points
    body_width = int(radius * 0.30)
    body_height = int(radius * 0.60)
    body_pts = np.array([
        [ax, ay - body_height],                    # Top
        [ax + body_width//2, ay - body_height//2],  # Upper-right
        [ax + body_width, ay],                     # Right
        [ax, ay + body_height],                    # Bottom
        [ax - body_width, ay],                     # Left
        [ax - body_width//2, ay - body_height//2],  # Upper-left
    ], np.int32)
    
    # Fill body
    ov = frame.copy()
    cv2.fillPoly(ov, [body_pts.reshape(-1,1,2)], (130, 60, 50))
    cv2.addWeighted(ov, 0.90, frame, 0.10, 0, frame)
    
    # Body outline
    cv2.polylines(frame, [body_pts.reshape(-1,1,2)], True, (170, 100, 70), 1)
    
    # Internal facet lines
    facet_color = (150, 80, 60)
    cv2.line(frame, (ax, ay - body_height), (ax + body_width//2, ay - body_height//2), facet_color, 1)
    cv2.line(frame, (ax, ay - body_height), (ax - body_width//2, ay - body_height//2), facet_color, 1)
    cv2.line(frame, (ax, ay + body_height), (ax + body_width, ay), facet_color, 1)
    cv2.line(frame, (ax, ay + body_height), (ax - body_width, ay), facet_color, 1)

    # Head above body - small diamond
    head_x = ax
    head_y = ay - body_height - int(radius * 0.25)
    head_radius = int(radius * 0.16)
    
    # Head diamond
    head_pts = np.array([
        [head_x, head_y - head_radius],      # Top
        [head_x + head_radius//2, head_y],  # Right
        [head_x, head_y + head_radius],      # Bottom
        [head_x - head_radius//2, head_y],  # Left
    ], np.int32)
    
    ov = frame.copy()
    cv2.fillPoly(ov, [head_pts.reshape(-1,1,2)], (155, 120, 90))
    cv2.addWeighted(ov, 0.85, frame, 0.15, 0, frame)
    cv2.polylines(frame, [head_pts.reshape(-1,1,2)], True, (175, 140, 110), 1)
    
    # 3 crest feathers above head
    crest_color = (50, 25, 10)
    crest_configs = [
        (-20, int(radius * 0.22)),   # Left crest
        (0, int(radius * 0.28)),     # Center crest
        (20, int(radius * 0.22)),    # Right crest
    ]
    
    for angle_offset, crest_length in crest_configs:
        crest_angle = math.radians(-90 + angle_offset)
        crest_tip_x = int(head_x + math.cos(crest_angle) * crest_length)
        crest_tip_y = int(head_y + math.sin(crest_angle) * crest_length)
        
        # Stem line
        cv2.line(frame, (head_x, head_y - head_radius), (crest_tip_x, crest_tip_y), crest_color, 1)
        
        # Tip circle
        cv2.circle(frame, (crest_tip_x, crest_tip_y), 2, crest_color, -1)
    
    # Two tiny eye dots on head
    eye_offset_x = int(radius * 0.08)
    eye_offset_y = int(radius * 0.04)
    cv2.circle(frame, (head_x - eye_offset_x, head_y - eye_offset_y), 2, (20, 10, 5), -1)
    cv2.circle(frame, (head_x + eye_offset_x, head_y - eye_offset_y), 2, (20, 10, 5), -1)

    # Soft outer glow
    ov = frame.copy()
    cv2.circle(ov, (ax, ay), int(radius * 4.0), color, -1)
    cv2.addWeighted(ov, 0.05, frame, 0.95, 0, frame)
    
    ov = frame.copy()
    cv2.circle(ov, (ax, ay), int(radius * 3.0), color, -1)
    cv2.addWeighted(ov, 0.04, frame, 0.96, 0, frame)


def draw_mrigashirsha_antlers(frame, cx, cy, radius, color, now,
                               index_tip_x=None, index_tip_y=None,
                               middle_tip_x=None, middle_tip_y=None):
    """
    Mrigashirsha (Deer) — geometric low-poly deer head with antlers.
    Full face with triangular facets, centered on palm center.
    """
    
    def _draw_triangle(frame, pts, fill_color, outline_color, fill_alpha=0.08, line_alpha=0.85, thickness=1):
        """Helper to draw filled triangle with outline"""
        arr = np.array(pts, np.int32).reshape(-1,1,2)
        # Fill
        ov = frame.copy()
        cv2.fillPoly(ov, [arr], fill_color)
        cv2.addWeighted(ov, fill_alpha, frame, 1-fill_alpha, 0, frame)
        # Outline
        ov2 = frame.copy()
        cv2.polylines(ov2, [arr], True, outline_color, thickness)
        cv2.addWeighted(ov2, line_alpha, frame, 1-line_alpha, 0, frame)
    
    def _draw_line(frame, p1, p2, line_color, alpha=0.85, thickness=1):
        """Helper to draw line with transparency"""
        ov = frame.copy()
        cv2.line(ov, p1, p2, line_color, thickness)
        cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)
    
    # Face structure points
    top_skull = (cx, cy - int(radius*1.4))
    upper_left_brow = (cx - int(radius*0.55), cy - int(radius*1.0))
    upper_right_brow = (cx + int(radius*0.55), cy - int(radius*1.0))
    left_cheek = (cx - int(radius*0.75), cy - int(radius*0.3))
    right_cheek = (cx + int(radius*0.75), cy - int(radius*0.3))
    left_jaw = (cx - int(radius*0.55), cy + int(radius*0.5))
    right_jaw = (cx + int(radius*0.55), cy + int(radius*0.5))
    chin = (cx, cy + int(radius*0.8))
    nose_bridge_top = (cx, cy - int(radius*0.5))
    nose_tip = (cx, cy + int(radius*0.1))
    left_nostril = (cx - int(radius*0.20), cy + int(radius*0.15))
    right_nostril = (cx + int(radius*0.20), cy + int(radius*0.15))
    
    fill_color = (0, 0, 0)
    outline_color = (0, 0, 0)
    
    # Draw face triangular facets
    face_triangles = [
        # Upper face triangles
        [top_skull, upper_left_brow, upper_right_brow],
        [upper_left_brow, left_cheek, nose_bridge_top],
        [upper_right_brow, right_cheek, nose_bridge_top],
        
        # Mid face triangles
        [left_cheek, left_jaw, nose_tip],
        [right_cheek, right_jaw, nose_tip],
        [left_jaw, chin, right_jaw],
        
        # Nose area triangles
        [nose_bridge_top, left_nostril, right_nostril],
        [left_nostril, right_nostril, nose_tip],
        
        # Additional inner facet lines
        [upper_left_brow, nose_bridge_top, left_cheek],
        [upper_right_brow, nose_bridge_top, right_cheek],
        [left_cheek, nose_tip, right_cheek],
    ]
    
    for triangle in face_triangles:
        _draw_triangle(frame, triangle, fill_color, outline_color)
    
    # Eyes (diamond shapes)
    eye_width = int(radius*0.18)
    eye_height = int(radius*0.12)
    left_eye_center = (cx - int(radius*0.38), cy - int(radius*0.72))
    right_eye_center = (cx + int(radius*0.38), cy - int(radius*0.72))
    
    for eye_center in [left_eye_center, right_eye_center]:
        ex, ey = eye_center
        eye_diamond = [
            (ex, ey - eye_height//2),
            (ex + eye_width//2, ey),
            (ex, ey + eye_height//2),
            (ex - eye_width//2, ey)
        ]
        _draw_triangle(frame, eye_diamond[:3], (0, 0, 0), outline_color, 0.15, 0.90)
        _draw_triangle(frame, [eye_diamond[0], eye_diamond[2], eye_diamond[3]], (0, 0, 0), outline_color, 0.15, 0.90)
    
    # Ears
    # Left ear
    left_ear_base_left = upper_left_brow
    left_ear_base_right = (cx - int(radius*0.30), cy - int(radius*1.1))
    left_ear_tip = (cx - int(radius*0.75), cy - int(radius*1.45))
    left_ear_triangle = [left_ear_base_left, left_ear_base_right, left_ear_tip]
    _draw_triangle(frame, left_ear_triangle, fill_color, outline_color, 0.10, 0.85)
    
    # Left inner ear
    left_inner_ear = [
        (left_ear_base_left[0] + int(radius*0.08), left_ear_base_left[1] + int(radius*0.08)),
        (left_ear_base_right[0] + int(radius*0.08), left_ear_base_right[1] + int(radius*0.08)),
        (left_ear_tip[0] + int(radius*0.08), left_ear_tip[1] + int(radius*0.08))
    ]
    _draw_triangle(frame, left_inner_ear, fill_color, outline_color, 0.10, 0.85)
    
    # Right ear
    right_ear_base_left = (cx + int(radius*0.30), cy - int(radius*1.1))
    right_ear_base_right = upper_right_brow
    right_ear_tip = (cx + int(radius*0.75), cy - int(radius*1.45))
    right_ear_triangle = [right_ear_base_left, right_ear_base_right, right_ear_tip]
    _draw_triangle(frame, right_ear_triangle, fill_color, outline_color, 0.10, 0.85)
    
    # Right inner ear
    right_inner_ear = [
        (right_ear_base_left[0] - int(radius*0.08), right_ear_base_left[1] + int(radius*0.08)),
        (right_ear_base_right[0] - int(radius*0.08), right_ear_base_right[1] + int(radius*0.08)),
        (right_ear_tip[0] - int(radius*0.08), right_ear_tip[1] + int(radius*0.08))
    ]
    _draw_triangle(frame, right_inner_ear, fill_color, outline_color, 0.10, 0.85)
    
    # Antlers with gentle sway animation
    sway_offset = int(math.sin(now * 0.9) * radius * 0.04)
    
    def _draw_antler(base_x, base_y, mirror):
        """Draw branching antler structure"""
        # Apply sway to all x positions
        sx = base_x + sway_offset
        
        # Main shaft
        shaft_top = (sx + mirror * int(-radius*0.15), base_y - int(radius*0.7))
        _draw_line(frame, (sx, base_y), shaft_top, outline_color, 0.85, 1)
        
        # Shaft midpoint
        shaft_mid = (sx + mirror * int(-radius*0.08), base_y - int(radius*0.35))
        
        # From shaft midpoint: branch left
        branch_left_end = (sx + mirror * int(-radius*0.45), base_y - int(radius*0.55))
        _draw_line(frame, shaft_mid, branch_left_end, outline_color, 0.85, 1)
        
        # From shaft midpoint: branch right
        branch_right_end = (sx + mirror * int(radius*0.20), base_y - int(radius*0.40))
        _draw_line(frame, shaft_mid, branch_right_end, outline_color, 0.85, 1)
        
        # From shaft top: branch left
        top_branch_left = (sx + mirror * int(-radius*0.35), base_y - int(radius*1.0))
        _draw_line(frame, shaft_top, top_branch_left, outline_color, 0.85, 1)
        
        # From shaft top: branch right
        top_branch_right = (sx + mirror * int(radius*0.10), base_y - int(radius*0.90))
        _draw_line(frame, shaft_top, top_branch_right, outline_color, 0.85, 1)
        
        # From far left branch tip: one more tine up-left
        far_left_tip = (sx + mirror * int(-radius*0.55), base_y - int(radius*0.85))
        _draw_line(frame, branch_left_end, far_left_tip, outline_color, 0.85, 1)
        
        # Add glow dots at branch tips
        for tip in [shaft_top, branch_left_end, branch_right_end, top_branch_left, top_branch_right, far_left_tip]:
            ov = frame.copy()
            cv2.circle(ov, tip, 3, (0, 0, 0), -1)
            cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
    
    # Draw antlers from ear tips
    _draw_antler(left_ear_tip[0], left_ear_tip[1], -1)  # Left antler (mirror = -1)
    _draw_antler(right_ear_tip[0], right_ear_tip[1], 1)  # Right antler (mirror = 1)
    
    # Neck below face
    left_neck_bottom = (cx - int(radius*0.45), cy + int(radius*1.1))
    right_neck_bottom = (cx + int(radius*0.45), cy + int(radius*1.1))
    
    # Neck lines
    _draw_line(frame, left_jaw, left_neck_bottom, outline_color, 0.85, 1)
    _draw_line(frame, right_jaw, right_neck_bottom, outline_color, 0.85, 1)
    _draw_line(frame, left_neck_bottom, right_neck_bottom, outline_color, 0.85, 1)
    
    # Additional neck facet lines for low-poly look
    neck_center = (cx, cy + int(radius*0.8))
    _draw_line(frame, neck_center, left_neck_bottom, outline_color, 0.85, 1)
    _draw_line(frame, neck_center, right_neck_bottom, outline_color, 0.85, 1)


def draw_simhamukha_mane(frame, cx, cy, radius, color, now,
                          tip_coords=None):
    """
    Simhamukha (Lion) — fully filled low-poly lion face with dense triangular facets.
    Warm amber/orange/gold colors with teal eyes and white muzzle.
    """
    # Color palette (BGR)
    dark_brown   = (20,  60, 120)   # dark shadow brown
    deep_amber   = (30,  90, 180)   # deep amber
    mid_amber    = (40, 130, 210)   # mid amber
    bright_amber = (60, 165, 230)   # bright amber orange
    light_gold   = (120, 200, 240)  # light gold
    pale_cream   = (180, 220, 245)  # pale cream/beige
    white_muzzle = (220, 230, 230)  # off-white muzzle
    teal_eye     = (160, 180,  60)  # teal eye color
    yellow_eye   = (40,  210, 220)  # yellow eye iris
    dark_eye     = (20,   20,  20)  # dark pupil
    dark_nose    = (30,   30,  60)  # dark nose
    
    pulse = (math.sin(now * 2.0) + 1) / 2
    
    # Helper function to draw filled triangle with outline
    def _tri(frame, p1, p2, p3, fill_color, fill_alpha, outline=True):
        pts = np.array([p1, p2, p3], np.int32).reshape((-1,1,2))
        ov = frame.copy()
        cv2.fillPoly(ov, [pts], fill_color)
        cv2.addWeighted(ov, fill_alpha, frame, 1-fill_alpha, 0, frame)
        
        if outline:
            outline_color = (15, 35, 80)
            ov2 = frame.copy()
            cv2.line(ov2, p1, p2, outline_color, 1)
            cv2.line(ov2, p2, p3, outline_color, 1)
            cv2.line(ov2, p3, p1, outline_color, 1)
            cv2.addWeighted(ov2, 0.60, frame, 0.40, 0, frame)
    
    # MANE - Top triangles (spiky crown)
    _tri(frame, (cx, cy-int(radius*1.85)), (cx-int(radius*0.30), cy-int(radius*1.40)), 
         (cx+int(radius*0.30), cy-int(radius*1.40)), bright_amber, 0.88 + 0.05*pulse)
    _tri(frame, (cx+int(radius*0.30), cy-int(radius*1.40)), (cx+int(radius*0.70), cy-int(radius*1.75)), 
         (cx+int(radius*0.90), cy-int(radius*1.30)), mid_amber, 0.85 + 0.05*pulse)
    _tri(frame, (cx-int(radius*0.30), cy-int(radius*1.40)), (cx-int(radius*0.70), cy-int(radius*1.75)), 
         (cx-int(radius*0.90), cy-int(radius*1.30)), mid_amber, 0.85 + 0.05*pulse)
    _tri(frame, (cx+int(radius*0.70), cy-int(radius*1.75)), (cx+int(radius*1.10), cy-int(radius*1.40)), 
         (cx+int(radius*0.90), cy-int(radius*1.30)), deep_amber, 0.82 + 0.05*pulse)
    _tri(frame, (cx-int(radius*0.70), cy-int(radius*1.75)), (cx-int(radius*1.10), cy-int(radius*1.40)), 
         (cx-int(radius*0.90), cy-int(radius*1.30)), deep_amber, 0.82 + 0.05*pulse)
    
    # Right mane triangles
    _tri(frame, (cx+int(radius*0.90), cy-int(radius*1.30)), (cx+int(radius*1.10), cy-int(radius*1.40)), 
         (cx+int(radius*1.45), cy-int(radius*0.80)), mid_amber, 0.80 + 0.05*pulse)
    _tri(frame, (cx+int(radius*0.90), cy-int(radius*1.30)), (cx+int(radius*1.45), cy-int(radius*0.80)), 
         (cx+int(radius*1.50), cy-int(radius*0.20)), bright_amber, 0.78 + 0.05*pulse)
    _tri(frame, (cx+int(radius*0.90), cy-int(radius*1.30)), (cx+int(radius*1.50), cy-int(radius*0.20)), 
         (cx+int(radius*1.35), cy+int(radius*0.40)), mid_amber, 0.80 + 0.05*pulse)
    _tri(frame, (cx+int(radius*0.90), cy-int(radius*1.30)), (cx+int(radius*1.35), cy+int(radius*0.40)), 
         (cx+int(radius*0.80), cy+int(radius*0.85)), deep_amber, 0.82 + 0.05*pulse)
    _tri(frame, (cx+int(radius*0.80), cy+int(radius*0.85)), (cx+int(radius*1.35), cy+int(radius*0.40)), 
         (cx+int(radius*0.55), cy+int(radius*1.20)), mid_amber, 0.78 + 0.05*pulse)
    _tri(frame, (cx+int(radius*0.55), cy+int(radius*1.20)), (cx+int(radius*1.35), cy+int(radius*0.40)), 
         (cx+int(radius*0.30), cy+int(radius*1.40)), light_gold, 0.75 + 0.05*pulse)
    
    # Left mane triangles (mirror of right)
    _tri(frame, (cx-int(radius*0.90), cy-int(radius*1.30)), (cx-int(radius*1.10), cy-int(radius*1.40)), 
         (cx-int(radius*1.45), cy-int(radius*0.80)), mid_amber, 0.80 + 0.05*pulse)
    _tri(frame, (cx-int(radius*0.90), cy-int(radius*1.30)), (cx-int(radius*1.45), cy-int(radius*0.80)), 
         (cx-int(radius*1.50), cy-int(radius*0.20)), bright_amber, 0.78 + 0.05*pulse)
    _tri(frame, (cx-int(radius*0.90), cy-int(radius*1.30)), (cx-int(radius*1.50), cy-int(radius*0.20)), 
         (cx-int(radius*1.35), cy+int(radius*0.40)), mid_amber, 0.80 + 0.05*pulse)
    _tri(frame, (cx-int(radius*0.90), cy-int(radius*1.30)), (cx-int(radius*1.35), cy+int(radius*0.40)), 
         (cx-int(radius*0.80), cy+int(radius*0.85)), deep_amber, 0.82 + 0.05*pulse)
    _tri(frame, (cx-int(radius*0.80), cy+int(radius*0.85)), (cx-int(radius*1.35), cy+int(radius*0.40)), 
         (cx-int(radius*0.55), cy+int(radius*1.20)), mid_amber, 0.78 + 0.05*pulse)
    _tri(frame, (cx-int(radius*0.55), cy+int(radius*1.20)), (cx-int(radius*1.35), cy+int(radius*0.40)), 
         (cx-int(radius*0.30), cy+int(radius*1.40)), light_gold, 0.75 + 0.05*pulse)
    
    # Dark shadow triangles in mane
    _tri(frame, (cx, cy-int(radius*1.85)), (cx+int(radius*0.30), cy-int(radius*1.40)), 
         (cx+int(radius*0.70), cy-int(radius*1.75)), dark_brown, 0.70)
    _tri(frame, (cx, cy-int(radius*1.85)), (cx-int(radius*0.30), cy-int(radius*1.40)), 
         (cx-int(radius*0.70), cy-int(radius*1.75)), dark_brown, 0.70)
    _tri(frame, (cx+int(radius*1.10), cy-int(radius*1.40)), (cx+int(radius*1.45), cy-int(radius*0.80)), 
         (cx+int(radius*0.90), cy-int(radius*1.30)), dark_brown, 0.55)
    _tri(frame, (cx-int(radius*1.10), cy-int(radius*1.40)), (cx-int(radius*1.45), cy-int(radius*0.80)), 
         (cx-int(radius*0.90), cy-int(radius*1.30)), dark_brown, 0.55)
    
    # FACE - Forehead triangles
    _tri(frame, (cx-int(radius*0.30), cy-int(radius*1.40)), (cx+int(radius*0.30), cy-int(radius*1.40)), 
         (cx, cy-int(radius*0.90)), bright_amber, 0.85)
    _tri(frame, (cx-int(radius*0.30), cy-int(radius*1.40)), (cx-int(radius*0.90), cy-int(radius*1.30)), 
         (cx-int(radius*0.55), cy-int(radius*0.80)), mid_amber, 0.82)
    _tri(frame, (cx+int(radius*0.30), cy-int(radius*1.40)), (cx+int(radius*0.90), cy-int(radius*1.30)), 
         (cx+int(radius*0.55), cy-int(radius*0.80)), mid_amber, 0.82)
    _tri(frame, (cx-int(radius*0.55), cy-int(radius*0.80)), (cx-int(radius*0.90), cy-int(radius*1.30)), 
         (cx-int(radius*0.85), cy-int(radius*0.55)), deep_amber, 0.80)
    _tri(frame, (cx+int(radius*0.55), cy-int(radius*0.80)), (cx+int(radius*0.90), cy-int(radius*1.30)), 
         (cx+int(radius*0.85), cy-int(radius*0.55)), deep_amber, 0.80)
    
    # Mid face triangles
    _tri(frame, (cx-int(radius*0.55), cy-int(radius*0.80)), (cx, cy-int(radius*0.90)), 
         (cx-int(radius*0.40), cy-int(radius*0.40)), light_gold, 0.80)
    _tri(frame, (cx+int(radius*0.55), cy-int(radius*0.80)), (cx, cy-int(radius*0.90)), 
         (cx+int(radius*0.40), cy-int(radius*0.40)), light_gold, 0.80)
    _tri(frame, (cx, cy-int(radius*0.90)), (cx-int(radius*0.40), cy-int(radius*0.40)), 
         (cx+int(radius*0.40), cy-int(radius*0.40)), pale_cream, 0.75)
    _tri(frame, (cx-int(radius*0.85), cy-int(radius*0.55)), (cx-int(radius*0.55), cy-int(radius*0.80)), 
         (cx-int(radius*0.80), cy-int(radius*0.10)), mid_amber, 0.80)
    _tri(frame, (cx+int(radius*0.85), cy-int(radius*0.55)), (cx+int(radius*0.55), cy-int(radius*0.80)), 
         (cx+int(radius*0.80), cy-int(radius*0.10)), mid_amber, 0.80)
    _tri(frame, (cx-int(radius*0.80), cy-int(radius*0.10)), (cx-int(radius*0.55), cy-int(radius*0.80)), 
         (cx-int(radius*0.40), cy-int(radius*0.40)), bright_amber, 0.78)
    _tri(frame, (cx+int(radius*0.80), cy-int(radius*0.10)), (cx+int(radius*0.55), cy-int(radius*0.80)), 
         (cx+int(radius*0.40), cy-int(radius*0.40)), bright_amber, 0.78)
    
    # Lower face triangles
    _tri(frame, (cx-int(radius*0.40), cy-int(radius*0.40)), (cx+int(radius*0.40), cy-int(radius*0.40)), 
         (cx, cy+int(radius*0.10)), pale_cream, 0.80)
    _tri(frame, (cx-int(radius*0.80), cy-int(radius*0.10)), (cx-int(radius*0.40), cy-int(radius*0.40)), 
         (cx-int(radius*0.65), cy+int(radius*0.40)), mid_amber, 0.78)
    _tri(frame, (cx+int(radius*0.80), cy-int(radius*0.10)), (cx+int(radius*0.40), cy-int(radius*0.40)), 
         (cx+int(radius*0.65), cy+int(radius*0.40)), mid_amber, 0.78)
    _tri(frame, (cx-int(radius*0.65), cy+int(radius*0.40)), (cx-int(radius*0.40), cy-int(radius*0.40)), 
         (cx, cy+int(radius*0.10)), light_gold, 0.75)
    _tri(frame, (cx+int(radius*0.65), cy+int(radius*0.40)), (cx+int(radius*0.40), cy-int(radius*0.40)), 
         (cx, cy+int(radius*0.10)), light_gold, 0.75)
    _tri(frame, (cx-int(radius*0.65), cy+int(radius*0.40)), (cx, cy+int(radius*0.10)), 
         (cx, cy+int(radius*0.65)), pale_cream, 0.78)
    _tri(frame, (cx+int(radius*0.65), cy+int(radius*0.40)), (cx, cy+int(radius*0.10)), 
         (cx, cy+int(radius*0.65)), pale_cream, 0.78)
    _tri(frame, (cx-int(radius*0.65), cy+int(radius*0.40)), (cx-int(radius*0.80), cy-int(radius*0.10)), 
         (cx-int(radius*0.80), cy+int(radius*0.75)), mid_amber, 0.80)
    _tri(frame, (cx+int(radius*0.65), cy+int(radius*0.40)), (cx+int(radius*0.80), cy-int(radius*0.10)), 
         (cx+int(radius*0.80), cy+int(radius*0.75)), mid_amber, 0.80)
    _tri(frame, (cx-int(radius*0.80), cy+int(radius*0.75)), (cx-int(radius*0.65), cy+int(radius*0.40)), 
         (cx, cy+int(radius*0.65)), light_gold, 0.75)
    _tri(frame, (cx+int(radius*0.80), cy+int(radius*0.75)), (cx+int(radius*0.65), cy+int(radius*0.40)), 
         (cx, cy+int(radius*0.65)), light_gold, 0.75)
    
    # MUZZLE - white/cream area
    _tri(frame, (cx-int(radius*0.38), cy+int(radius*0.10)), (cx+int(radius*0.38), cy+int(radius*0.10)), 
         (cx, cy+int(radius*0.65)), white_muzzle, 0.85)
    _tri(frame, (cx-int(radius*0.38), cy+int(radius*0.10)), (cx-int(radius*0.55), cy+int(radius*0.50)), 
         (cx, cy+int(radius*0.65)), white_muzzle, 0.80)
    _tri(frame, (cx+int(radius*0.38), cy+int(radius*0.10)), (cx+int(radius*0.55), cy+int(radius*0.50)), 
         (cx, cy+int(radius*0.65)), white_muzzle, 0.80)
    
    # NOSE
    _tri(frame, (cx-int(radius*0.15), cy+int(radius*0.08)), (cx+int(radius*0.15), cy+int(radius*0.08)), 
         (cx, cy+int(radius*0.25)), dark_nose, 0.90)
    _tri(frame, (cx-int(radius*0.06), cy+int(radius*0.10)), (cx+int(radius*0.06), cy+int(radius*0.10)), 
         (cx, cy+int(radius*0.16)), (100, 100, 150), 0.60)
    
    # EYES - function to draw eye at position
    def draw_eye(eye_cx, eye_cy):
        # Outer eye diamond
        outer_pts = [
            (eye_cx, eye_cy - int(radius*0.10)),
            (eye_cx + int(radius*0.14), eye_cy),
            (eye_cx, eye_cy + int(radius*0.10)),
            (eye_cx - int(radius*0.14), eye_cy)
        ]
        _tri(frame, outer_pts[0], outer_pts[1], outer_pts[2], teal_eye, 0.90, False)
        _tri(frame, outer_pts[0], outer_pts[2], outer_pts[3], teal_eye, 0.90, False)
        
        # Inner iris diamond
        inner_pts = [
            (eye_cx, eye_cy - int(radius*0.07)),
            (eye_cx + int(radius*0.09), eye_cy),
            (eye_cx, eye_cy + int(radius*0.07)),
            (eye_cx - int(radius*0.09), eye_cy)
        ]
        _tri(frame, inner_pts[0], inner_pts[1], inner_pts[2], yellow_eye, 0.88, False)
        _tri(frame, inner_pts[0], inner_pts[2], inner_pts[3], yellow_eye, 0.88, False)
        
        # Pupil diamond
        pupil_pts = [
            (eye_cx, eye_cy - int(radius*0.05)),
            (eye_cx + int(radius*0.04), eye_cy),
            (eye_cx, eye_cy + int(radius*0.05)),
            (eye_cx - int(radius*0.04), eye_cy)
        ]
        _tri(frame, pupil_pts[0], pupil_pts[1], pupil_pts[2], dark_eye, 0.95, False)
        _tri(frame, pupil_pts[0], pupil_pts[2], pupil_pts[3], dark_eye, 0.95, False)
        
        # White outline on outer diamond
        for i in range(4):
            cv2.line(frame, outer_pts[i], outer_pts[(i+1)%4], (255, 255, 255), 1)
    
    # Draw left and right eyes
    draw_eye(cx - int(radius*0.32), cy - int(radius*0.48))
    draw_eye(cx + int(radius*0.32), cy - int(radius*0.48))
    
    # Bottom chin/beard area
    _tri(frame, (cx-int(radius*0.80), cy+int(radius*0.75)), (cx+int(radius*0.80), cy+int(radius*0.75)), 
         (cx, cy+int(radius*1.30)), light_gold, 0.72)
    _tri(frame, (cx-int(radius*0.80), cy+int(radius*0.75)), (cx-int(radius*0.30), cy+int(radius*1.40)), 
         (cx, cy+int(radius*1.30)), pale_cream, 0.70)
    _tri(frame, (cx+int(radius*0.80), cy+int(radius*0.75)), (cx+int(radius*0.30), cy+int(radius*1.40)), 
         (cx, cy+int(radius*1.30)), pale_cream, 0.70)
    
    # Connect bottom of mane to bottom of face
    _tri(frame, (cx+int(radius*0.30), cy+int(radius*1.40)), (cx+int(radius*0.80), cy+int(radius*0.75)), 
         (cx, cy+int(radius*1.30)), mid_amber, 0.75)
    _tri(frame, (cx-int(radius*0.30), cy+int(radius*1.40)), (cx-int(radius*0.80), cy+int(radius*0.75)), 
         (cx, cy+int(radius*1.30)), mid_amber, 0.75)


def draw_sarpashirsha_snake(frame, cx, cy, radius, color, now,
                             middle_tip_x=None, middle_tip_y=None):
    """
    Sarpashirsha (Serpent) — geometric cobra hood above MIDDLE FINGERTIP.
    Low-poly cobra head with scale facets + forked tongue flick.
    """
    if middle_tip_x is None:
        middle_tip_x = cx
        middle_tip_y = cy - int(radius*1.2)
    mx, my = middle_tip_x, middle_tip_y

    # All hardcoded colors
    outline_c  = (0, 0, 0)          # black
    fill_dark  = (20, 20, 20)        # near black fill for shadow
    fill_mid   = (60, 60, 60)        # dark grey
    fill_light = (120, 120, 120)     # mid grey
    fill_pale  = (200, 200, 200)     # pale grey
    tongue_c   = (0, 50, 200)        # red tongue BGR
    eye_c      = (200, 200, 200)     # light eye

    # Helper function for triangle
    def _tri(frame, p1, p2, p3, fill_c, fill_alpha):
        pts = np.array([p1, p2, p3], np.int32).reshape((-1,1,2))
        ov = frame.copy()
        cv2.fillPoly(ov, [pts], fill_c)
        cv2.addWeighted(ov, fill_alpha, frame, 1-fill_alpha, 0, frame)
        cv2.polylines(frame, [pts], True, outline_c, 1)

    # Pulse animation
    pulse = (math.sin(now * 2.5) + 1) / 2
    line_alpha_mult = 0.80 + pulse * 0.20

    # Soft jade glow at anchor
    ov = frame.copy()
    cv2.circle(ov, (mx, my), int(radius*2.5), color, -1)
    cv2.addWeighted(ov, 0.05, frame, 0.95, 0, frame)

    # OUTER HOOD POLYGON
    hood_points = [
        (int(mx - radius*1.10), int(my - radius*0.80)),
        (int(mx), int(my - radius*1.40)),
        (int(mx + radius*1.10), int(my - radius*0.80)),
        (int(mx + radius*1.0), int(my - radius*0.10)),
        (int(mx + radius*0.55), int(my + radius*0.60)),
        (int(mx), int(my + radius*1.10)),
        (int(mx - radius*0.55), int(my + radius*0.60)),
        (int(mx - radius*1.0), int(my - radius*0.10)),
    ]
    hood_arr = np.array(hood_points, np.int32).reshape(-1,1,2)
    ov = frame.copy()
    cv2.polylines(ov, [hood_arr], True, outline_c, 2)
    cv2.addWeighted(ov, 0.90, frame, 0.10, 0, frame)

    # HOOD INNER STRUCTURE — facet lines creating scale panels
    facet_lines = [
        # Center vertical line
        ((int(mx), int(my-radius*1.40)), (int(mx), int(my+radius*1.10))),
        # Left hood diagonal
        ((int(mx-radius*1.10), int(my-radius*0.80)), (int(mx-radius*0.35), int(my-radius*0.20))),
        # Right hood diagonal
        ((int(mx+radius*1.10), int(my-radius*0.80)), (int(mx+radius*0.35), int(my-radius*0.20))),
        # Left mid diagonal
        ((int(mx-radius*1.0), int(my-radius*0.10)), (int(mx-radius*0.35), int(my-radius*0.20))),
        # Right mid diagonal
        ((int(mx+radius*1.0), int(my-radius*0.10)), (int(mx+radius*0.35), int(my-radius*0.20))),
        # Inner diamond top lines
        ((int(mx), int(my-radius*1.40)), (int(mx-radius*0.35), int(my-radius*0.20))),
        ((int(mx), int(my-radius*1.40)), (int(mx+radius*0.35), int(my-radius*0.20))),
        # Horizontal neck line
        ((int(mx-radius*0.35), int(my-radius*0.20)), (int(mx+radius*0.35), int(my-radius*0.20))),
        # Left jaw lines
        ((int(mx-radius*0.35), int(my-radius*0.20)), (int(mx-radius*0.55), int(my+radius*0.60))),
        ((int(mx-radius*0.55), int(my+radius*0.60)), (int(mx), int(my+radius*1.10))),
        # Right jaw lines
        ((int(mx+radius*0.35), int(my-radius*0.20)), (int(mx+radius*0.55), int(my+radius*0.60))),
        ((int(mx+radius*0.55), int(my+radius*0.60)), (int(mx), int(my+radius*1.10))),
        # Hood scale horizontal lines at 3 levels
        ((int(mx-radius*0.80), int(my-radius*0.55)), (int(mx+radius*0.80), int(my-radius*0.55))),
        ((int(mx-radius*0.65), int(my-radius*0.10)), (int(mx+radius*0.65), int(my-radius*0.10))),
        ((int(mx-radius*0.45), int(my+radius*0.28)), (int(mx+radius*0.45), int(my+radius*0.28))),
    ]

    for p1, p2 in facet_lines:
        ov = frame.copy()
        cv2.line(ov, p1, p2, outline_c, 1)
        cv2.addWeighted(ov, 0.75 * line_alpha_mult, frame, 1-0.75*line_alpha_mult, 0, frame)

    # FILLED SHADOW TRIANGLES
    # Top center triangle
    _tri(frame, (int(mx), int(my-radius*1.40)), 
         (int(mx-radius*0.35), int(my-radius*0.20)), 
         (int(mx+radius*0.35), int(my-radius*0.20)), fill_mid, 0.25)
    
    # Left hood shadow
    _tri(frame, (int(mx-radius*1.10), int(my-radius*0.80)), 
         (int(mx-radius*0.35), int(my-radius*0.20)), 
         (int(mx-radius*1.0), int(my-radius*0.10)), fill_dark, 0.30)
    
    # Right hood shadow
    _tri(frame, (int(mx+radius*1.10), int(my-radius*0.80)), 
         (int(mx+radius*0.35), int(my-radius*0.20)), 
         (int(mx+radius*1.0), int(my-radius*0.10)), fill_dark, 0.30)
    
    # Lower jaw shadow
    _tri(frame, (int(mx-radius*0.35), int(my-radius*0.20)), 
         (int(mx+radius*0.35), int(my-radius*0.20)), 
         (int(mx), int(my+radius*1.10)), fill_mid, 0.20)

    # EYES — two oval diamond shapes
    for eye_x, eye_y in [(int(mx - radius*0.38), int(my - radius*0.62)), 
                         (int(mx + radius*0.38), int(my - radius*0.62))]:
        # Outer diamond
        eye_w = int(radius*0.28)
        eye_h = int(radius*0.20)
        eye_pts = np.array([
            [eye_x, eye_y - eye_h],
            [eye_x + eye_w, eye_y],
            [eye_x, eye_y + eye_h],
            [eye_x - eye_w, eye_y],
        ], np.int32).reshape((-1,1,2))
        ov = frame.copy()
        cv2.fillPoly(ov, [eye_pts], fill_pale)
        cv2.addWeighted(ov, 0.80, frame, 0.20, 0, frame)
        cv2.polylines(frame, [eye_pts], True, outline_c, 1)
        
        # Inner oval
        inner_w = int(radius*0.14)
        inner_h = int(radius*0.10)
        inner_pts = np.array([
            [eye_x, eye_y - inner_h],
            [eye_x + inner_w, eye_y],
            [eye_x, eye_y + inner_h],
            [eye_x - inner_w, eye_y],
        ], np.int32).reshape((-1,1,2))
        ov = frame.copy()
        cv2.fillPoly(ov, [inner_pts], fill_dark)
        cv2.addWeighted(ov, 0.90, frame, 0.10, 0, frame)
        
        # Small white highlight dot top-right of inner oval
        highlight_x = eye_x + int(inner_w * 0.5)
        highlight_y = eye_y - int(inner_h * 0.5)
        cv2.circle(frame, (highlight_x, highlight_y), 2, eye_c, -1)

    # OPEN JAW — mouth cavity below eye level
    # Upper jaw line
    ov = frame.copy()
    cv2.line(ov, (int(mx-radius*0.35), int(my+radius*0.28)), 
             (int(mx+radius*0.35), int(my+radius*0.28)), outline_c, 1)
    cv2.addWeighted(ov, 0.85 * line_alpha_mult, frame, 1-0.85*line_alpha_mult, 0, frame)
    
    # Mouth cavity polygon
    mouth_pts = np.array([
        [int(mx-radius*0.32), int(my+radius*0.28)],
        [int(mx+radius*0.32), int(my+radius*0.28)],
        [int(mx+radius*0.20), int(my+radius*0.65)],
        [int(mx-radius*0.20), int(my+radius*0.65)],
    ], np.int32).reshape((-1,1,2))
    ov = frame.copy()
    cv2.fillPoly(ov, [mouth_pts], fill_dark)
    cv2.addWeighted(ov, 0.70, frame, 0.30, 0, frame)
    cv2.polylines(frame, [mouth_pts], True, outline_c, 1)
    
    # Two fangs hanging from upper jaw
    # Left fang
    left_fang = np.array([
        [int(mx-radius*0.18), int(my+radius*0.28)],
        [int(mx-radius*0.10), int(my+radius*0.28)],
        [int(mx-radius*0.14), int(my+radius*0.52)],
    ], np.int32).reshape((-1,1,2))
    ov = frame.copy()
    cv2.fillPoly(ov, [left_fang], fill_pale)
    cv2.addWeighted(ov, 0.90, frame, 0.10, 0, frame)
    cv2.polylines(frame, [left_fang], True, outline_c, 1)
    
    # Right fang
    right_fang = np.array([
        [int(mx+radius*0.18), int(my+radius*0.28)],
        [int(mx+radius*0.10), int(my+radius*0.28)],
        [int(mx+radius*0.14), int(my+radius*0.52)],
    ], np.int32).reshape((-1,1,2))
    ov = frame.copy()
    cv2.fillPoly(ov, [right_fang], fill_pale)
    cv2.addWeighted(ov, 0.90, frame, 0.10, 0, frame)
    cv2.polylines(frame, [right_fang], True, outline_c, 1)

    # FORKED TONGUE — flicks out from mouth
    flick = math.sin(now * 8.0) * 0.3
    tongue_base = (int(mx), int(my+radius*0.65))
    tongue_tip = (int(mx + int(flick*radius*0.3)), int(my+radius*0.90))
    fork_left = (int(mx - radius*0.12 + int(flick*radius*0.3)), int(my+radius*1.05))
    fork_right = (int(mx + radius*0.12 + int(flick*radius*0.3)), int(my+radius*1.05))
    
    # Draw tongue base to tip
    ov = frame.copy()
    cv2.line(ov, tongue_base, tongue_tip, tongue_c, 2)
    cv2.addWeighted(ov, 0.85, frame, 0.15, 0, frame)
    
    # Draw two fork lines
    ov = frame.copy()
    cv2.line(ov, tongue_tip, fork_left, tongue_c, 1)
    cv2.line(ov, tongue_tip, fork_right, tongue_c, 1)
    cv2.addWeighted(ov, 0.80, frame, 0.20, 0, frame)


def draw_hamsapaksha_wing(frame, cx, cy, radius, color, now,
                           pinky_tip_x=None, pinky_tip_y=None):
    """
    Hamsapaksha (Swan Wing) — geometric low-poly swan facing left.
    Body centered at base of pinky finger, tail extends along pinky direction.
    Neck curves up and away from pinky, wing raises upward.
    Built from straight-edged triangular facets.
    """
    if pinky_tip_x is None:
        bx, by = cx, cy
    else:
        # Body center sits at base of pinky finger (offset from pinky tip)
        bx = pinky_tip_x + int(radius * 0.6)
        by = pinky_tip_y + int(radius * 0.4)

    # Hardcoded colors
    outline_c  = (0, 0, 0)        # black
    fill_white = (245, 245, 245)  # near white
    fill_grey  = (200, 200, 200)  # light grey
    fill_dark  = (150, 150, 150)  # shadow grey
    beak_c     = (0, 140, 220)    # orange beak BGR

    # Wing beat animation
    beat = math.sin(now * 1.6) * 0.05

    # Helper function for triangle
    def _tri(frame, p1, p2, p3, fill_c, fill_alpha):
        pts = np.array([p1, p2, p3], np.int32).reshape((-1,1,2))
        ov = frame.copy()
        cv2.fillPoly(ov, [pts], fill_c)
        cv2.addWeighted(ov, fill_alpha, frame, 1-fill_alpha, 0, frame)
        cv2.polylines(frame, [pts], True, outline_c, 1)

    # Tail direction: down and slightly right from body center (+0.5, +0.8) normalized
    tail_dir_x = 0.5
    tail_dir_y = 0.8
    tail_len = math.sqrt(tail_dir_x**2 + tail_dir_y**2)
    tail_dir_x /= tail_len
    tail_dir_y /= tail_len

    # Tail tip point following pinky finger direction
    tail_tip_x = bx + int(radius * 1.0 * tail_dir_x)
    tail_tip_y = by + int(radius * 1.3 * tail_dir_y)

    # BODY polygon centered at (bx, by) with tail extending in tail direction
    body_points = [
        (int(bx - radius*0.825), int(by)),                    # left middle
        (int(bx - radius*0.6), int(by - radius*0.6)),          # left upper
        (int(bx),             int(by - radius*0.75)),        # top
        (int(bx + radius*0.75), int(by - radius*0.45)),          # right upper
        (int(bx + radius*0.975), int(by + radius*0.15)),          # right middle
        (tail_tip_x, tail_tip_y),                              # tail tip (replaces right lower)
        (int(bx),             int(by + radius*0.825)),        # bottom
        (int(bx - radius*0.6), int(by + radius*0.525)),          # left lower
    ]

    # Fill triangles connecting all points to center
    center = (int(bx), int(by))
    for i, point in enumerate(body_points):
        fill_c = fill_white if i % 2 == 0 else fill_grey
        _tri(frame, center, point, body_points[(i+1)%len(body_points)], fill_c, 0.50)

    # Draw body outline
    cv2.polylines(frame, [np.array(body_points, np.int32).reshape((-1,1,2))], True, outline_c, 1)

    # Internal facet lines from center to each point
    for point in body_points:
        ov = frame.copy()
        cv2.line(ov, center, point, outline_c, 1)
        cv2.addWeighted(ov, 0.40, frame, 0.60, 0, frame)

    # WING polygon raised above body on right side
    wing_root = (int(bx + radius*0.15), int(by - radius*0.6))
    wing_points = [
        (int(bx + radius*0.75), int(by - radius*1.35 + beat*radius)),    # upper mid
        (int(bx + radius*1.2), int(by - radius*1.95 + beat*radius)),    # upper tip
        (int(bx + radius*1.65), int(by - radius*2.25 + beat*radius)),    # top tip
        (int(bx + radius*2.025), int(by - radius*1.95 + beat*radius)),    # upper right
        (int(bx + radius*2.1), int(by - radius*1.35 + beat*radius)),    # right tip
        (int(bx + radius*1.65), int(by - radius*0.825)),             # lower right
        (int(bx + radius*0.9), int(by - radius*0.45)),             # lower mid
    ]

    # Fill wing triangles fan-style from root point
    for i, point in enumerate(wing_points):
        fill_c = fill_white if i % 2 == 0 else fill_grey
        next_point = wing_points[(i+1)%len(wing_points)]
        _tri(frame, wing_root, point, next_point, fill_c, 0.50)

    # Dark shadow triangle at wing base
    _tri(frame, wing_root, wing_points[6], wing_points[0], fill_dark, 0.55)

    # Draw wing outline
    all_wing_points = [wing_root] + wing_points
    cv2.polylines(frame, [np.array(all_wing_points, np.int32).reshape((-1,1,2))], True, outline_c, 1)

    # Internal facet lines from root to each tip
    for point in wing_points:
        ov = frame.copy()
        cv2.line(ov, wing_root, point, outline_c, 1)
        cv2.addWeighted(ov, 0.45, frame, 0.55, 0, frame)

    # NECK — 4-segment polygon forming S-curve going up-left from body
    neck_segments = [
        # Base segment
        [(int(bx - radius*0.825), int(by - radius*0.15)), (int(bx - radius*0.6), int(by - radius*0.6))],
        [(int(bx - radius*1.05), int(by - radius*0.975)), (int(bx - radius*0.825), int(by - radius*1.125))],
        # Lower neck segment
        [(int(bx - radius*1.05), int(by - radius*0.975)), (int(bx - radius*0.825), int(by - radius*1.125))],
        [(int(bx - radius*0.825), int(by - radius*1.575)), (int(bx - radius*0.525), int(by - radius*1.5))],
        # Upper neck segment
        [(int(bx - radius*0.825), int(by - radius*1.575)), (int(bx - radius*0.525), int(by - radius*1.5))],
        [(int(bx - radius*0.45), int(by - radius*1.875)), (int(bx - radius*0.75), int(by - radius*1.92))],
        # Head base segment
        [(int(bx - radius*0.45), int(by - radius*1.875)), (int(bx - radius*0.75), int(by - radius*1.92))],
        [(int(bx - radius*0.57), int(by - radius*2.13)), (int(bx - radius*0.27), int(by - radius*2.13))],
    ]

    # Draw neck as connected quad segments
    for i in range(0, len(neck_segments), 2):
        quad_points = neck_segments[i] + neck_segments[i+1]
        pts = np.array(quad_points, np.int32).reshape((-1,1,2))
        ov = frame.copy()
        cv2.fillPoly(ov, [pts], fill_white)
        cv2.addWeighted(ov, 0.50, frame, 0.50, 0, frame)
        cv2.polylines(frame, [pts], True, outline_c, 1)

        # Internal diagonal facet lines
        ov = frame.copy()
        cv2.line(ov, quad_points[0], quad_points[3], outline_c, 1)
        cv2.addWeighted(ov, 0.40, frame, 0.60, 0, frame)

    # HEAD — small polygon at top of neck
    head_points = [
        (int(bx - radius*0.57), int(by - radius*2.37)),  # top
        (int(bx - radius*0.27), int(by - radius*2.13)),  # right
        (int(bx - radius*0.42), int(by - radius*1.92)),  # bottom right
        (int(bx - radius*0.78), int(by - radius*1.95)),  # bottom left
        (int(bx - radius*0.87), int(by - radius*2.16)),  # left
    ]

    # 3 internal facet triangles
    _tri(frame, head_points[0], head_points[1], head_points[2], fill_white, 0.52)
    _tri(frame, head_points[0], head_points[2], head_points[3], fill_white, 0.52)
    _tri(frame, head_points[0], head_points[3], head_points[4], fill_white, 0.52)

    # Draw head outline
    cv2.polylines(frame, [np.array(head_points, np.int32).reshape((-1,1,2))], True, outline_c, 1)

    # BEAK — small triangle pointing right
    beak_points = np.array([
        [int(bx - radius*0.27), int(by - radius*2.25)],  # top
        [int(bx - radius*0.27), int(by - radius*2.07)],  # bottom
        [int(bx + radius*0.03), int(by - radius*2.16)],  # tip
    ], np.int32)
    ov = frame.copy()
    cv2.fillPoly(ov, [beak_points.reshape(-1,1,2)], beak_c)
    cv2.addWeighted(ov, 0.90, frame, 0.10, 0, frame)
    cv2.polylines(frame, [beak_points.reshape(-1,1,2)], True, outline_c, 1)

    # EYE — tiny filled circle
    eye_center = (int(bx - radius*0.45), int(by - radius*2.22))
    cv2.circle(frame, eye_center, 3, outline_c, -1)

    # Soft glow at anchor
    ov = frame.copy()
    cv2.circle(ov, (bx, by), int(radius * 3.0), color, -1)
    cv2.addWeighted(ov, 0.05, frame, 0.95, 0, frame)


def draw_suchi_needle(frame, cx, cy, radius, color, now,
                       index_tip_x=None, index_tip_y=None):
    # Default anchor position if not provided
    if index_tip_x is None:
        index_tip_x = cx
        index_tip_y = cy - int(radius * 1.5)
    
    ix, iy = index_tip_x, index_tip_y
    
    # NEEDLE BEAM — sharp bright shaft shooting upward from fingertip
    beam_end_y = iy - int(radius * 8.0)
    beam_layers = [
        (12, (200, 200, 255), 0.06),
        (6, (220, 220, 255), 0.12),
        (2, (240, 240, 255), 0.55),
        (1, (255, 255, 255), 0.95)
    ]
    
    for width, beam_color, alpha in beam_layers:
        ov = frame.copy()
        cv2.line(ov, (ix, iy), (ix, beam_end_y), beam_color, width)
        cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)
    
    # NEEDLE TIP GLOW — bright point at fingertip
    glow_circles = [
        (int(radius * 0.8), (200, 200, 255), 0.08),
        (int(radius * 0.45), (220, 220, 255), 0.18),
        (int(radius * 0.20), (255, 255, 255), 0.70)
    ]
    
    for glow_radius, glow_color, alpha in glow_circles:
        ov = frame.copy()
        cv2.circle(ov, (ix, iy), glow_radius, glow_color, -1)
        cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)
    
    # Bright white dot at center
    cv2.circle(frame, (ix, iy), 3, (255, 255, 255), -1)
    
    # THREAD SPIRAL — thin spiral thread wrapping around the needle
    spiral_phase = now * 2.5
    spiral_points = []
    n_points = 80
    
    for t in range(n_points):
        t_norm = t / (n_points - 1)
        angle = t_norm * 3 * 2 * math.pi + spiral_phase
        sx = int(ix + math.cos(angle) * radius * 0.12)
        sy = int(iy - t_norm * radius * 4.0)
        spiral_points.append((sx, sy))
    
    if len(spiral_points) > 1:
        ov = frame.copy()
        for i in range(1, len(spiral_points)):
            cv2.line(ov, spiral_points[i-1], spiral_points[i], (180, 220, 255), 1)
        cv2.addWeighted(ov, 0.45, frame, 0.55, 0, frame)
    
    # PARTICLE SPARKS — tiny sparks shooting off from tip
    for i in range(8):
        spark_angle = i * math.pi / 4 + now * 3
        spark_length = int(radius * 0.35)
        spark_end_x = int(ix + math.cos(spark_angle) * spark_length)
        spark_end_y = int(iy + math.sin(spark_angle) * spark_length)
        spark_alpha = 0.30 + 0.25 * math.sin(now * 4 + i)
        
        ov = frame.copy()
        cv2.line(ov, (ix, iy), (spark_end_x, spark_end_y), (255, 255, 255), 1)
        cv2.addWeighted(ov, spark_alpha, frame, 1 - spark_alpha, 0, frame)
    
    # UNIVERSE RINGS — concentric rings expanding outward from tip
    for i in range(3):
        ring_phase = (now * 0.8 + i * 0.5) % 1.0
        ring_r = int(radius * (0.5 + ring_phase * 2.5))
        ring_alpha = (1.0 - ring_phase) * 0.25
        
        ov = frame.copy()
        cv2.circle(ov, (ix, iy), ring_r, (200, 200, 255), 1)
        cv2.addWeighted(ov, ring_alpha, frame, 1 - ring_alpha, 0, frame)
    
    # ONE SYMBOL — small "1" indicator near tip
    symbol_x = ix + int(radius * 0.35)
    symbol_y = iy - int(radius * 0.5)
    
    # Vertical line for "1"
    ov = frame.copy()
    cv2.line(ov, (symbol_x, symbol_y), (symbol_x, symbol_y + int(radius * 0.30)), (255, 255, 255), 1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
    
    # Small dot below it
    ov = frame.copy()
    cv2.circle(ov, (symbol_x, symbol_y + int(radius * 0.35)), 2, (255, 255, 255), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)


# ─────────────────────────────────────────────────────
# MUDRA THEMES
# ─────────────────────────────────────────────────────

MUDRA_THEMES = {
    'Pataka': {
        'color':       (215, 245, 255),
        'glow_color':  (200, 235, 250),
        'p_color':     (220, 248, 255),
        'geometry_fn': draw_pataka_rays,
        'p_behavior':  'drift_up',
        'p_count':     4,
        'trail_color': (200, 235, 245),
    },
    'Tripataka': {
        'color':       (30, 130, 255),
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
        'color':       (255, 230, 190),
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
        'color':       (160, 130, 230),
        'glow_color':  (150, 120, 220),
        'p_color':     (170, 140, 235),
        'geometry_fn': draw_katakamukha_petals,
        'p_behavior':  'none',
        'p_count':     0,
        'trail_color': (155, 125, 225),
    },
    'Mushti': {
        'color':       (30,  20, 180),
        'glow_color':  (20,  10, 160),
        'p_color':     (40,  30, 195),
        'geometry_fn': draw_mushti_core,
        'p_behavior':  'none',
        'p_count':     0,
        'trail_color': (25,  15, 170),
    },
    'Shikhara': {
        'color':       (0,  140, 255),
        'glow_color':  (0,  120, 240),
        'p_color':     (20, 160, 255),
        'geometry_fn': draw_shikhara_pillar,
        'p_behavior':  'rise',
        'p_count':     3,
        'trail_color': (0,  130, 245),
    },
    'Trishula': {
        'color':       (30, 180, 220),
        'glow_color':  (20, 150, 200),
        'p_color':     (40, 190, 230),
        'geometry_fn': draw_trishula_flames,
        'p_behavior':  'none',
        'p_count':     0,
        'trail_color': (25, 160, 210),
    },
    'Tamrachuda': {
        'color':       (0,  100, 220),
        'glow_color':  (0,   80, 200),
        'p_color':     (20, 120, 235),
        'geometry_fn': draw_tamrachuda_crest,
        'p_behavior':  'none',
        'p_count':     0,
        'trail_color': (0,   90, 210),
    },
    'Kartarimukha': {
        'color':       (255, 200, 100),
        'glow_color':  (240, 185,  90),
        'p_color':     (255, 210, 120),
        'geometry_fn': draw_kartarimukha_lightning,
        'p_behavior':  'scatter',
        'p_count':     4,
        'trail_color': (245, 190,  95),
    },
    'Arala': {
        'color':       (230, 240, 220),
        'glow_color':  (220, 230, 210),
        'p_color':     (235, 245, 225),
        'geometry_fn': draw_arala_swirl,
        'p_behavior':  'none',
        'p_count':     0,
        'trail_color': (220, 230, 210),
    },
    'Shukatunda': {
        'color':       (80,  220,  60),
        'glow_color':  (60,  200,  40),
        'p_color':     (100, 230,  80),
        'geometry_fn': draw_shukatunda_arrow,
        'p_behavior':  'none',
        'p_count':     0,
        'trail_color': (70,  210,  50),
    },
    'Chandrakala': {
        'color':       (235, 235, 255),
        'glow_color':  (220, 220, 245),
        'p_color':     (240, 240, 255),
        'geometry_fn': draw_chandrakala_moon,
        'p_behavior':  'orbit',
        'p_count':     2,
        'trail_color': (225, 225, 248),
    },
    'Hamsasya': {
        'color':       (20,  190, 255),
        'glow_color':  (10,  170, 240),
        'p_color':     (30,  200, 255),
        'geometry_fn': draw_hamsasya_flame,
        'p_behavior':  'none',
        'p_count':     0,
        'trail_color': (15,  180, 248),
    },
    'Kapittha': {
        'color':       (0,   200, 255),
        'glow_color':  (0,   180, 240),
        'p_color':     (20,  210, 255),
        'geometry_fn': draw_kapittha_coins,
        'p_behavior':  'none',
        'p_count':     0,
        'trail_color': (0,   190, 248),
    },
    'Mayura': {
        'color':       (170, 160,  40),
        'glow_color':  (150, 140,  30),
        'p_color':     (180, 170,  50),
        'geometry_fn': draw_mayura_peacock,
        'p_behavior':  'float',
        'p_count':     2,
        'trail_color': (160, 150,  35),
    },
    'Mrigashirsha': {
        'color':       (40, 100, 140),
        'glow_color':  (30,  80, 120),
        'p_color':     (50, 110, 150),
        'geometry_fn': draw_mrigashirsha_antlers,
        'p_behavior':  'drift_up',
        'p_count':     2,
        'trail_color': (35,  90, 130),
    },
    'Simhamukha': {
        'color':       (30,  160, 220),
        'glow_color':  (20,  140, 200),
        'p_color':     (40,  170, 230),
        'geometry_fn': draw_simhamukha_mane,
        'p_behavior':  'scatter',
        'p_count':     3,
        'trail_color': (25,  150, 210),
    },
    'Sarpashirsha': {
        'color':       (100, 220, 120),
        'glow_color':  ( 80, 200, 100),
        'p_color':     (110, 230, 130),
        'geometry_fn': draw_sarpashirsha_snake,
        'p_behavior':  'none',
        'p_count':     0,
        'trail_color': ( 90, 210, 110),
    },
    'Hamsapaksha': {
        'color':       (230, 220, 200),
        'glow_color':  (220, 210, 190),
        'p_color':     (235, 225, 210),
        'geometry_fn': draw_hamsapaksha_wing,
        'p_behavior':  'float',
        'p_count':     2,
        'trail_color': (225, 215, 195),
    },
    'Suchi': {
        'color':       (255, 255, 220),
        'glow_color':  (240, 240, 200),
        'p_color':     (255, 255, 235),
        'geometry_fn': draw_suchi_needle,
        'p_behavior':  'rise',
        'p_count':     3,
        'trail_color': (245, 245, 210),
    },
}


# ─────────────────────────────────────────────────────
# MAIN RENDERER
# ─────────────────────────────────────────────────────



def compute_speed_scale(hand_speed):
    """Convert hand speed in pixels/frame to
    a scale multiplier for effects.
    Slow/still = 0.6, normal = 1.0, fast = 1.8"""
    if hand_speed < 2.0:
        return 0.6
    elif hand_speed < 8.0:
        return 0.6 + (hand_speed - 2.0) / 6.0 * 0.4
    elif hand_speed < 20.0:
        return 1.0 + (hand_speed - 8.0) / 12.0 * 0.8
    else:
        return 1.8


class MudraRenderer:
    def __init__(self, width=640, height=480):
        self.w = width
        self.h = height
        self.particles  = ParticleSystem(max_particles=200)
        self.trail_right = TrailSystem(maxlen=45)
        self.trail_left  = TrailSystem(maxlen=45)
        self.trail_left_arm  = TrailSystem(maxlen=20)
        self.trail_right_arm = TrailSystem(maxlen=20)
        self.frame_count = 0
        self.second_hand_pos = None
        self.pose_state  = None
        self.hand_speed  = 0.0

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
               second_landmarks=None,
               pose_state=None,
               hand_speed=0.0):
        if not landmarks:
            self.particles.update_draw(frame)
            return frame

        self.frame_count += 1
        self.pose_state  = pose_state
        self.hand_speed  = hand_speed
        speed_scale = compute_speed_scale(hand_speed)
        now = time.time()

        cx, cy  = self._palm_center(landmarks)
        radius  = self._hand_radius(landmarks)

        if second_landmarks:
            s_cx = int(second_landmarks[9][0] * self.w)
            s_cy = int(second_landmarks[9][1] * self.h)
            self.second_hand_pos = (s_cx, s_cy)
        else:
            self.second_hand_pos = None

        _trail = self.trail_right if handedness == 'Right' else self.trail_left
        _trail.update(cx, cy)
        
        # Get theme for arm trail color
        theme = MUDRA_THEMES.get(mudra)
        
        # Update arm trails from pose
        if self.pose_state:
            trail_thickness = max(1, int(2 * speed_scale))
            _trail_c = theme['trail_color'] if theme else (80, 80, 80)

        if mudra == 'Unknown' or score < 0.55:
            _trail = self.trail_right if handedness == 'Right' else self.trail_left
            _trail.draw(frame, (100,100,100), 1)
            self.particles.update_draw(frame)
            return frame

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
        _trail.draw(frame, t_color, thickness=3)

        # LAYER 2 — Glow
        glow_radius = int(radius * speed_scale)
        draw_glow(frame, cx, cy, glow_radius, g_color)

        # LAYER 3 — Geometry dispatch
        if mudra == 'Ardhapataka' and \
                hasattr(self, 'second_hand_pos') and \
                self.second_hand_pos is not None:
            tx, ty = self.second_hand_pos
            geo_fn(frame, cx, cy, radius, color, now, tx, ty)

        elif mudra == 'Trishula':
            t1 = (int(landmarks[8][0]*self.w),  int(landmarks[8][1]*self.h))
            t2 = (int(landmarks[12][0]*self.w), int(landmarks[12][1]*self.h))
            t3 = (int(landmarks[16][0]*self.w), int(landmarks[16][1]*self.h))
            geo_fn(frame, cx, cy, radius, color, now, t1, t2, t3)

        elif mudra == 'Arala':
            itx = int(landmarks[8][0]*self.w)
            ity = int(landmarks[8][1]*self.h)
            geo_fn(frame, cx, cy, radius, color, now, itx, ity)

        elif mudra == 'Shukatunda':
            itx  = int(landmarks[8][0]*self.w)
            ity  = int(landmarks[8][1]*self.h)
            ibx  = int(landmarks[5][0]*self.w)
            iby  = int(landmarks[5][1]*self.h)
            geo_fn(frame, cx, cy, radius, color, now, itx, ity, ibx, iby)

        elif mudra == 'Chandrakala':
            thx = int(landmarks[4][0]*self.w)
            thy = int(landmarks[4][1]*self.h)
            idx = int(landmarks[8][0]*self.w)
            idy = int(landmarks[8][1]*self.h)
            geo_fn(frame, cx, cy, radius, color, now, thx, thy, idx, idy)

        elif mudra == 'Hamsasya':
            pinch_x = int((landmarks[4][0]+landmarks[8][0])*0.5*self.w)
            pinch_y = int((landmarks[4][1]+landmarks[8][1])*0.5*self.h)
            spd = self.hand_speed
            geo_fn(frame, cx, cy, radius, color, now, pinch_x, pinch_y, spd)

        # ── NATURE GROUP — fingertip-anchored dispatch ──
        elif mudra == 'Mayura':
            # Anchor: thumb tip (lm 4)
            ttx = int(landmarks[4][0] * self.w)
            tty = int(landmarks[4][1] * self.h)
            geo_fn(frame, cx, cy, radius, color, now, ttx, tty)

        elif mudra == 'Mrigashirsha':
            # Anchor: index tip (lm 8) + middle tip (lm 12)
            itx = int(landmarks[8][0] * self.w)
            ity = int(landmarks[8][1] * self.h)
            mtx = int(landmarks[12][0] * self.w)
            mty = int(landmarks[12][1] * self.h)
            geo_fn(frame, cx, cy, radius, color, now, itx, ity, mtx, mty)

        elif mudra == 'Simhamukha':
            # Anchor: convex hull of all 5 fingertips
            tips = [
                (int(landmarks[4][0]*self.w),  int(landmarks[4][1]*self.h)),
                (int(landmarks[8][0]*self.w),  int(landmarks[8][1]*self.h)),
                (int(landmarks[12][0]*self.w), int(landmarks[12][1]*self.h)),
                (int(landmarks[16][0]*self.w), int(landmarks[16][1]*self.h)),
                (int(landmarks[20][0]*self.w), int(landmarks[20][1]*self.h)),
            ]
            geo_fn(frame, cx, cy, radius, color, now, tips)

        elif mudra == 'Sarpashirsha':
            # Anchor: middle finger tip (lm 12)
            mtx = int(landmarks[12][0] * self.w)
            mty = int(landmarks[12][1] * self.h)
            geo_fn(frame, cx, cy, radius, color, now, mtx, mty)

        elif mudra == 'Hamsapaksha':
            # Anchor: pinky tip (lm 20)
            ptx = int(landmarks[20][0] * self.w)
            pty = int(landmarks[20][1] * self.h)
            geo_fn(frame, cx, cy, radius, color, now, ptx, pty)

        elif mudra == 'Suchi':
            # Anchor: index fingertip (lm 8)
            itx = int(landmarks[8][0] * self.w)
            ity = int(landmarks[8][1] * self.h)
            geo_fn(frame, cx, cy, radius, color, now, itx, ity)

        else:
            geo_fn(frame, cx, cy, radius, color, now)

        # LAYER 4 — Particles
        if self.frame_count % 2 == 0 and theme.get('p_count', 0) > 0:
            scaled_count = max(1, int(p_count * speed_scale))
            self.particles.spawn(cx, cy, p_color, behavior,
                                 speed_factor=0.8 * speed_scale,
                                 count=scaled_count)
        self.particles.update_draw(frame)

        # LAYER 5 — Fingertip dots
        draw_fingertip_dots(frame, landmarks, color, self.w, self.h)

        return frame
