from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import cv2
import numpy as np
import math
import time
from collections import deque


@dataclass
class BodyState:
    # HAND DATA (populated now)
    left_hand:       Optional[List] = None
    right_hand:      Optional[List] = None
    active_hand:     Optional[List] = None
    active_side:     str = "Right"
    hand_speed:      float = 0.0
    hand_velocity:   Tuple = (0.0, 0.0)

    # BODY DATA (None until MediaPipe Pose added later)
    # Pose landmark indices for reference:
    # 11=left_shoulder  12=right_shoulder
    # 13=left_elbow     14=right_elbow
    # 23=left_hip       24=right_hip
    # 0=nose
    left_shoulder:   Optional[Tuple] = None
    right_shoulder:  Optional[Tuple] = None
    left_elbow:      Optional[Tuple] = None
    right_elbow:     Optional[Tuple] = None
    left_hip:        Optional[Tuple] = None
    right_hip:       Optional[Tuple] = None
    nose:            Optional[Tuple] = None
    spine_center:    Optional[Tuple] = None
    body_width:      float = 0.4
    torso_height:    float = 0.5

    # DERIVED PIXEL POSITIONS
    wrist_pos_px:    Tuple = (320, 240)
    fingertip_px:    Tuple = (320, 200)
    elbow_pos_px:    Optional[Tuple] = None
    shoulder_pos_px: Optional[Tuple] = None

    # MOTION HISTORY
    speed_history:   List = field(default_factory=list)
    avg_speed:       float = 0.0


class VisualEngine:

    MUDRA_TO_MODE = {
        "Pataka":       "FLOW",
        "Tripataka":    "ENERGY",
        "Ardhachandra": "ORBIT",
        "Mushti":       "COMPRESS",
        "Shikhara":     "BEAM",
        "Kapittha":     "ORB",
        "Katakamukha":  "RIBBON",
        "Suchi":        "RAY",
    }

    def __init__(self, width=640, height=480):
        self.w = width
        self.h = height

        self.mode           = "NONE"
        self.prev_mode      = "NONE"
        self.mode_start     = time.time()
        self.TRANSITION_DUR = 0.35

        self.particles      = []
        self.MAX_PARTICLES  = 300
        self.flow_particles = []

        self.ribbon_points  = deque(maxlen=80)
        self.ribbon_widths  = deque(maxlen=80)
        self.hand_trail     = deque(maxlen=40)

        self.body           = BodyState()

        self.fps_history    = deque(maxlen=30)
        self.last_time      = time.time()
        self.frame_count    = 0

    def update_hands(self, hand_landmarks, handedness="Right"):
        if not hand_landmarks:
            return
        b = self.body
        if handedness == "Right":
            b.right_hand = hand_landmarks
        else:
            b.left_hand = hand_landmarks
        b.active_hand = hand_landmarks
        b.active_side = handedness

        wrist = hand_landmarks[0]
        tip   = hand_landmarks[8]
        prev  = b.wrist_pos_px

        b.wrist_pos_px = (int(wrist[0]*self.w), int(wrist[1]*self.h))
        b.fingertip_px = (int(tip[0]*self.w),   int(tip[1]*self.h))

        dx = b.wrist_pos_px[0] - prev[0]
        dy = b.wrist_pos_px[1] - prev[1]
        b.hand_velocity = (dx, dy)
        b.hand_speed    = math.sqrt(dx*dx + dy*dy)

        b.speed_history.append(b.hand_speed)
        if len(b.speed_history) > 10:
            b.speed_history.pop(0)
        b.avg_speed = sum(b.speed_history) / len(b.speed_history)

        self.hand_trail.append((
            b.wrist_pos_px[0],
            b.wrist_pos_px[1],
            time.time()
        ))

    def update_body(self, pose_landmarks):
        # READY FOR FUTURE — call this when MediaPipe Pose is added
        # pose_landmarks = list of 33 (x,y,z) tuples
        if not pose_landmarks:
            return
        b = self.body
        b.left_shoulder  = pose_landmarks[11]
        b.right_shoulder = pose_landmarks[12]
        b.left_elbow     = pose_landmarks[13]
        b.right_elbow    = pose_landmarks[14]
        b.left_hip       = pose_landmarks[23]
        b.right_hip      = pose_landmarks[24]
        b.nose           = pose_landmarks[0]

        ls = pose_landmarks[11]
        rs = pose_landmarks[12]
        lh = pose_landmarks[23]
        rh = pose_landmarks[24]
        b.spine_center = (
            (ls[0]+rs[0]+lh[0]+rh[0])/4,
            (ls[1]+rs[1]+lh[1]+rh[1])/4,
            0
        )
        b.body_width   = abs(ls[0]-rs[0])
        b.torso_height = abs(((ls[1]+rs[1])/2)-((lh[1]+rh[1])/2))

        ae = pose_landmarks[14] if b.active_side=="Right" else pose_landmarks[13]
        as_ = pose_landmarks[12] if b.active_side=="Right" else pose_landmarks[11]
        b.elbow_pos_px    = (int(ae[0]*self.w),  int(ae[1]*self.h))
        b.shoulder_pos_px = (int(as_[0]*self.w), int(as_[1]*self.h))

    def set_mudra(self, mudra_name):
        new_mode = self.MUDRA_TO_MODE.get(mudra_name, "NONE")
        if new_mode != self.mode:
            self.prev_mode  = self.mode
            self.mode       = new_mode
            self.mode_start = time.time()
            self._on_mode_enter()

    def _on_mode_enter(self):
        hx, hy = self.body.wrist_pos_px
        for _ in range(25):
            angle = np.random.uniform(0, 2*math.pi)
            spd   = np.random.uniform(4, 14)
            self.particles.append({
                'x': float(hx), 'y': float(hy),
                'vx': math.cos(angle)*spd,
                'vy': math.sin(angle)*spd,
                'life': 1.0, 'decay': 0.055,
                'size': np.random.randint(2,5),
                'color': (255, 215, 80),
            })

    def render(self, frame):
        now = time.time()
        dt  = now - self.last_time
        self.last_time = now
        self.frame_count += 1

        fps = 1.0 / max(dt, 0.001)
        self.fps_history.append(fps)
        avg_fps = sum(self.fps_history)/len(self.fps_history)
        self.MAX_PARTICLES = 150 if avg_fps < 22 else 300

        if self.mode == "NONE":
            return frame

        overlay = frame.copy()

        dispatch = {
            "FLOW":     self._render_flow,
            "ENERGY":   self._render_energy,
            "ORBIT":    self._render_orbit,
            "COMPRESS": self._render_compress,
            "BEAM":     self._render_beam,
            "ORB":      self._render_orb,
            "RIBBON":   self._render_ribbon,
            "RAY":      self._render_ray,
        }
        fn = dispatch.get(self.mode)
        if fn:
            fn(overlay, dt)

        elapsed = now - self.mode_start
        if elapsed < self.TRANSITION_DUR:
            alpha = elapsed / self.TRANSITION_DUR
            cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
        else:
            frame[:] = overlay

        return frame

    def _update_draw_particles(self, frame):
        alive = []
        for p in self.particles:
            p['x']  += p['vx']
            p['y']  += p['vy']
            p['vx'] *= 0.95
            p['vy'] *= 0.95
            p['life'] -= p.get('decay', 0.03)
            if p['life'] > 0:
                alpha = p['life']
                color = tuple(int(c*alpha) for c in p['color'])
                cx,cy = int(p['x']), int(p['y'])
                if 0 <= cx < self.w and 0 <= cy < self.h:
                    cv2.circle(frame, (cx,cy),
                               p.get('size',3), color, -1)
                alive.append(p)
        self.particles = alive

    def _draw_trail(self, frame, color=(255,255,255), thickness=2):
        pts = list(self.hand_trail)
        if len(pts) < 2:
            return
        now = time.time()
        for i in range(1, len(pts)):
            age   = now - pts[i][2]
            alpha = max(0, 1.0 - age*3.0)
            if alpha <= 0:
                continue
            c = tuple(int(ch*alpha) for ch in color)
            cv2.line(frame,
                     (pts[i-1][0], pts[i-1][1]),
                     (pts[i][0],   pts[i][1]),
                     c, thickness)

    def _jagged_line(self, x1,y1,x2,y2, jag=8, segs=8):
        pts = [(x1,y1)]
        for i in range(1, segs):
            t  = i/segs
            mx = int(x1+(x2-x1)*t + np.random.randint(-jag,jag))
            my = int(y1+(y2-y1)*t + np.random.randint(-jag,jag))
            pts.append((mx,my))
        pts.append((x2,y2))
        return pts

    def _dist_px(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

    def clear(self):
        self.particles.clear()
        self.flow_particles = []
        self.ribbon_points.clear()
        self.ribbon_widths.clear()
        self.hand_trail.clear()
        self.mode = "NONE"

    def _render_flow(self, frame, dt):
        """Pataka — wind particles from palm.
        Body upgrade: elbow spawn point activates automatically."""
        b  = self.body
        hx,hy = b.wrist_pos_px
        sf = min(b.hand_speed/10.0, 3.0) + 0.5

        # Spawn from wrist
        for _ in range(int(3*sf)):
            if len(self.flow_particles) < self.MAX_PARTICLES:
                angle = np.random.uniform(0, 2*math.pi)
                spd   = np.random.uniform(1.5, 4.0)*sf
                self.flow_particles.append({
                    'x': hx + np.random.randint(-15,15),
                    'y': hy + np.random.randint(-15,15),
                    'vx': math.cos(angle)*spd,
                    'vy': math.sin(angle)*spd,
                    'life': 1.0,
                    'decay': np.random.uniform(0.012,0.030),
                    'size': np.random.randint(2,5),
                    'color': (
                        np.random.randint(150,255),
                        np.random.randint(200,255),
                        np.random.randint(220,255),
                    )
                })

        # BODY UPGRADE HOOK — activates when body tracked
        if b.elbow_pos_px:
            ex,ey = b.elbow_pos_px
            for _ in range(2):
                angle = np.random.uniform(0, 2*math.pi)
                self.flow_particles.append({
                    'x': float(ex), 'y': float(ey),
                    'vx': math.cos(angle)*2,
                    'vy': math.sin(angle)*2,
                    'life': 0.7, 'decay': 0.02,
                    'size': 2, 'color': (100,180,255)
                })

        # Attract toward hand + update
        alive = []
        for p in self.flow_particles:
            dx = hx - p['x']
            dy = hy - p['y']
            dist = math.sqrt(dx*dx+dy*dy)+1
            p['vx'] += (dx/dist)*0.3
            p['vy'] += (dy/dist)*0.3
            p['vx'] *= 0.97
            p['vy'] *= 0.97
            p['x']  += p['vx']
            p['y']  += p['vy']
            p['life'] -= p['decay']
            if p['life'] > 0:
                alpha = p['life']
                color = tuple(int(c*alpha) for c in p['color'])
                cx,cy = int(p['x']), int(p['y'])
                if 0<=cx<self.w and 0<=cy<self.h:
                    cv2.circle(frame,(cx,cy),p['size'],color,-1)
                alive.append(p)
        self.flow_particles = alive
        self._draw_trail(frame, color=(180,230,255), thickness=2)
    def _render_energy(self, frame, dt):
        """Tripataka — electric sparks and radial bursts.
        Body upgrade: elbow arc and shoulder halo."""
        b   = self.body
        hx,hy = b.wrist_pos_px
        sf  = min(b.hand_speed/8.0, 4.0) + 0.3
        now = time.time()

        for _ in range(int(8*sf)):
            if len(self.particles) < self.MAX_PARTICLES:
                angle = np.random.uniform(0, 2*math.pi)
                spd   = np.random.uniform(3.0,10.0)*sf
                self.particles.append({
                    'x': float(hx), 'y': float(hy),
                    'vx': math.cos(angle)*spd,
                    'vy': math.sin(angle)*spd,
                    'life': 1.0,
                    'decay': np.random.uniform(0.04,0.09),
                    'size': np.random.randint(2,5),
                    'color': (
                        np.random.randint(0,80),
                        np.random.randint(100,200),
                        np.random.randint(200,255),
                    )
                })

        if b.hand_speed < 2 and int(now*4)%2==0:
            pr = int(30+15*math.sin(now*4))
            cv2.circle(frame,(hx,hy),pr,(0,180,255),2)
            cv2.circle(frame,(hx,hy),pr+8,(0,100,200),1)

        for i in range(4):
            angle = now*3 + i*(math.pi/2)
            ex = int(hx+math.cos(angle)*45)
            ey = int(hy+math.sin(angle)*45)
            pts = self._jagged_line(hx,hy,ex,ey)
            for j in range(len(pts)-1):
                cv2.line(frame,pts[j],pts[j+1],(0,200,255),1)

        # BODY UPGRADE HOOKS
        if b.elbow_pos_px:
            ex2,ey2 = b.elbow_pos_px
            pts = self._jagged_line(ex2,ey2,hx,hy,jag=12,segs=12)
            for j in range(len(pts)-1):
                cv2.line(frame,pts[j],pts[j+1],(0,160,255),1)
        if b.shoulder_pos_px:
            sx,sy = b.shoulder_pos_px
            cv2.circle(frame,(sx,sy),25,(0,80,180),1)

        self._update_draw_particles(frame)
        self._draw_trail(frame,color=(0,200,255),thickness=3)
    def _render_orbit(self, frame, dt):
        """Ardhachandra — crescent arcs orbiting hand.
        Body upgrade: larger arc anchored to elbow."""
        b   = self.body
        hx,hy = b.wrist_pos_px
        now = time.time()

        rings = [
            (40, 1.2, (180,180,255), 1),
            (65, 0.7, (140,200,255), 2),
            (90, 0.4, (100,160,230), 1),
        ]

        if b.elbow_pos_px:
            ex,ey = b.elbow_pos_px
            cv2.ellipse(frame,(ex,ey),(80,40),
                        int(math.degrees(now*0.3)),
                        0,180,(80,100,200),1)

        for r,spd,color,w in rings:
            cv2.ellipse(frame,(hx,hy),(r,r//2),
                        int(math.degrees(now*spd)),
                        0,200,color,w)

        if self.frame_count % 3 == 0:
            angle = now*0.8
            px = int(hx+math.cos(angle)*55)
            py = int(hy+math.sin(angle)*28)
            self.particles.append({
                'x':float(px),'y':float(py),
                'vx':np.random.uniform(-0.5,0.5),
                'vy':np.random.uniform(-1.5,-0.3),
                'life':1.0,'decay':0.012,
                'size':np.random.randint(2,4),
                'color':(200,220,255)
            })

        self._update_draw_particles(frame)
        self._draw_trail(frame,color=(180,200,255),thickness=2)
    def _render_compress(self, frame, dt):
        """Mushti — gravity collapse toward fist.
        Body upgrade: shoulder as extra spawn point."""
        b  = self.body
        hx,hy = b.wrist_pos_px

        spawn = [
            (np.random.randint(0,self.w), 0),
            (np.random.randint(0,self.w), self.h),
            (0, np.random.randint(0,self.h)),
            (self.w, np.random.randint(0,self.h)),
        ]
        if b.shoulder_pos_px:
            spawn.append(b.shoulder_pos_px)

        if len(self.particles) < self.MAX_PARTICLES:
            for _ in range(4):
                px,py = spawn[np.random.randint(0,len(spawn))]
                self.particles.append({
                    'x':float(px),'y':float(py),
                    'vx':0.0,'vy':0.0,
                    'life':1.0,'decay':0.007,
                    'size':np.random.randint(2,5),
                    'color':(
                        np.random.randint(180,255),
                        np.random.randint(50,120),
                        np.random.randint(0,60),
                    )
                })

        alive = []
        for p in self.particles:
            dx = hx-p['x']; dy = hy-p['y']
            dist = math.sqrt(dx*dx+dy*dy)+1
            force = min(800/(dist*dist), 5.0)
            p['vx'] += (dx/dist)*force
            p['vy'] += (dy/dist)*force
            p['vx'] *= 0.92; p['vy'] *= 0.92
            p['x']  += p['vx']; p['y'] += p['vy']
            p['life'] -= p['decay']
            if dist < 15:
                p['life'] = 0
                for _ in range(3):
                    a2 = np.random.uniform(0,2*math.pi)
                    self.particles.append({
                        'x':p['x'],'y':p['y'],
                        'vx':math.cos(a2)*5,
                        'vy':math.sin(a2)*5,
                        'life':0.5,'decay':0.08,
                        'size':2,'color':(255,150,50)
                    })
            if p['life']>0:
                alpha=p['life']
                color=tuple(int(c*alpha) for c in p['color'])
                cx,cy=int(p['x']),int(p['y'])
                if 0<=cx<self.w and 0<=cy<self.h:
                    cv2.circle(frame,(cx,cy),p['size'],color,-1)
                alive.append(p)
        self.particles = alive

        ring_r = int(25+10*math.sin(time.time()*4))
        cv2.circle(frame,(hx,hy),ring_r,(255,100,30),2)
    def _render_beam(self, frame, dt):
        """Shikhara — vertical light pillar from thumb tip.
        Body upgrade: beam extends from elbow, shoulder glow ring."""
        b   = self.body
        hx,hy = b.wrist_pos_px
        now   = time.time()
        twist = int(math.sin(now*2)*8)

        beam_bottom = b.elbow_pos_px[1] if b.elbow_pos_px else hy
        beam_top    = 0

        for gw,af in [(22,0.12),(15,0.22),(9,0.40),(4,0.75)]:
            ov = frame.copy()
            cv2.rectangle(ov,
                (hx-gw//2+twist, beam_top),
                (hx+gw//2+twist, beam_bottom),
                (200,220,255),-1)
            cv2.addWeighted(ov,af,frame,1-af,0,frame)

        cv2.line(frame,
                 (hx+twist,beam_top),
                 (hx+twist,beam_bottom),
                 (255,255,255),2)

        if int(now*30)%2==0:
            sy2 = np.random.randint(beam_top, max(beam_top+1,beam_bottom))
            sx2 = hx+twist+np.random.randint(-10,10)
            cv2.circle(frame,(sx2,sy2),2,(200,230,255),-1)

        # BODY UPGRADE HOOK
        if b.shoulder_pos_px:
            sx3,sy3 = b.shoulder_pos_px
            cv2.circle(frame,(sx3,sy3),18,(180,200,255),1)
    def _render_orb(self, frame, dt):
        """Kapittha — glowing orb follows hand.
        Body upgrade: smaller secondary orbs at elbow and shoulder."""
        b   = self.body
        hx,hy = b.wrist_pos_px
        now   = time.time()
        orb_r = int(28+6*math.sin(now*3))

        for r,af in [(orb_r+20,0.10),(orb_r+12,0.20),
                     (orb_r+6, 0.35),(orb_r,    0.65)]:
            ov = frame.copy()
            cv2.circle(ov,(hx,hy),r,(150,255,200),-1)
            cv2.addWeighted(ov,af,frame,1-af,0,frame)
        cv2.circle(frame,(hx,hy),orb_r//3,(220,255,240),-1)

        for i in range(5):
            angle = now*1.5 + i*(2*math.pi/5)
            sx = int(hx+math.cos(angle)*(orb_r+12))
            sy = int(hy+math.sin(angle)*(orb_r+12))
            cv2.circle(frame,(sx,sy),3,(180,255,210),-1)

        # BODY UPGRADE HOOKS
        if b.elbow_pos_px:
            ex,ey = b.elbow_pos_px
            r2 = int(12+3*math.sin(now*3+1))
            cv2.circle(frame,(ex,ey),r2,(100,200,150),2)
        if b.shoulder_pos_px:
            sx4,sy4 = b.shoulder_pos_px
            r3 = int(8+2*math.sin(now*3+2))
            cv2.circle(frame,(sx4,sy4),r3,(80,160,120),1)

        if b.hand_speed > 12:
            for _ in range(8):
                angle = np.random.uniform(0,2*math.pi)
                self.particles.append({
                    'x':float(hx),'y':float(hy),
                    'vx':math.cos(angle)*np.random.uniform(3,8),
                    'vy':math.sin(angle)*np.random.uniform(3,8),
                    'life':1.0,'decay':0.05,
                    'size':3,'color':(100,255,180)
                })
        self._update_draw_particles(frame)
    def _render_ribbon(self, frame, dt):
        """Katakamukha — elegant color-shifting ribbon trail.
        Body upgrade: ribbon originates from elbow for longer curves."""
        b  = self.body
        hx,hy = b.wrist_pos_px
        sf = min(b.hand_speed/5.0, 3.0)

        self.ribbon_points.append((hx,hy))
        self.ribbon_widths.append(max(2,int(4+sf*4)))

        pts = list(self.ribbon_points)
        wds = list(self.ribbon_widths)

        # BODY UPGRADE HOOK
        if b.elbow_pos_px and len(pts) > 0:
            pts = [b.elbow_pos_px] + pts
            wds = [3] + wds

        if len(pts) < 4:
            return

        for i in range(1, len(pts)):
            t     = i/len(pts)
            alpha = t
            r  = int(255*(1-t))
            g  = int(150*math.sin(t*math.pi))
            bl = int(255*t)
            color     = (bl, g, r)
            thickness = max(1, int(wds[i]*alpha))
            cv2.line(frame, pts[i-1], pts[i], color, thickness)

        cv2.circle(frame,(hx,hy),5,(255,220,180),-1)
    def _render_ray(self, frame, dt):
        """Suchi — laser beam from index fingertip in pointing direction.
        Body upgrade: beam intensity scales with arm extension."""
        b  = self.body
        if not b.active_hand:
            return

        lm     = b.active_hand
        tip_x  = int(lm[8][0]*self.w)
        tip_y  = int(lm[8][1]*self.h)
        base_x = int(lm[5][0]*self.w)
        base_y = int(lm[5][1]*self.h)

        dx = tip_x - base_x
        dy = tip_y - base_y
        length = math.sqrt(dx*dx+dy*dy)+1
        scale  = max(self.w, self.h)*1.5
        end_x  = int(tip_x+(dx/length)*scale)
        end_y  = int(tip_y+(dy/length)*scale)

        # BODY UPGRADE HOOK — scales beam power
        beam_power = 1.0
        if b.elbow_pos_px and b.shoulder_pos_px:
            arm_ext    = self._dist_px(b.elbow_pos_px, b.shoulder_pos_px)
            beam_power = min(arm_ext/150.0, 2.0)

        intensity = min(int(beam_power*255), 255)

        cv2.line(frame,(tip_x,tip_y),(end_x,end_y),
                 (80,80,intensity),10)
        cv2.line(frame,(tip_x,tip_y),(end_x,end_y),
                 (160,160,255),5)
        cv2.line(frame,(tip_x,tip_y),(end_x,end_y),
                 (255,255,255),1)

        cv2.circle(frame,(tip_x,tip_y),8,(200,200,255),-1)
        cv2.circle(frame,(tip_x,tip_y),14,(150,150,255),2)
