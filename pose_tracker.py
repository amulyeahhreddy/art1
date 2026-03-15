import cv2
import mediapipe as mp


class PoseTracker:
    def __init__(self, use_lite=False):
        self.mp_pose = mp.solutions.pose
        model_complexity = 0 if use_lite else 1
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.KEY_LANDMARKS = {
            'left_shoulder':  11,
            'right_shoulder': 12,
            'left_elbow':     13,
            'right_elbow':    14,
            'left_wrist':     15,
            'right_wrist':    16,
            'left_hip':       23,
            'right_hip':      24,
        }

    def process(self, frame_rgb):
        return self.pose.process(frame_rgb)

    def get_pose_state(self, results, frame_w, frame_h):
        """Returns dict of key landmark pixel positions.
        Returns None if no pose detected."""
        if not results or not results.pose_landmarks:
            return None
        lm = results.pose_landmarks.landmark
        state = {}
        for name, idx in self.KEY_LANDMARKS.items():
            state[name] = (
                int(lm[idx].x * frame_w),
                int(lm[idx].y * frame_h),
                lm[idx].visibility,
            )
        return state

    def get_hand_speed(self, pose_state, prev_pose_state, frame_w, frame_h):
        """Compute wrist movement speed in pixels per frame."""
        if not pose_state or not prev_pose_state:
            return 0.0
        import math
        speeds = []
        for wrist in ['left_wrist', 'right_wrist']:
            if wrist in pose_state and wrist in prev_pose_state:
                x1, y1, _ = prev_pose_state[wrist]
                x2, y2, _ = pose_state[wrist]
                speeds.append(math.sqrt((x2-x1)**2 + (y2-y1)**2))
        return max(speeds) if speeds else 0.0
