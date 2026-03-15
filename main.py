import cv2
import math
import time
import numpy as np
import mediapipe as mp
from hand_tracker import HandTracker
from mudra_recognizer import MudraRecognizer
from visual_effects import VisualEffects
from visual_engine import VisualEngine
from mandala_renderer import MandalaRenderer
from renderer import MudraRenderer
from pose_tracker import PoseTracker
from renderer import MUDRA_THEMES

# Renderer handles visual effects for mudras
DEFAULT_GLOW_COLOR = (180, 180, 180)




def main():
    # Initialize webcam with fallback logic
    cap = None
    camera_indices = [0, 1, 2]  # Try different camera indices
    
    for idx in camera_indices:
        print(f"Trying camera index {idx}...")
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            # Test if we can actually read a frame
            ret, test_frame = cap.read()
            if ret:
                print(f"Successfully opened camera {idx}")
                break
            else:
                cap.release()
                cap = None
        else:
            if cap:
                cap.release()
                cap = None
    
    if cap is None:
        print("Error: Could not open any webcam. Please check:")
        print("1. Camera is connected and not in use by another application")
        print("2. Camera drivers are properly installed")
        print("3. Camera permissions are granted")
        print("4. Try running: python -c \"import cv2; print('OpenCV version:', cv2.__version__)\"")
        return
    
    # Set webcam properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Initialize hand tracker, mudra recognizer, and visual effects
    hand_tracker = HandTracker()
    mudra_recognizer = MudraRecognizer()
    visual_effects = VisualEffects()
    engine = VisualEngine(width=640, height=480)
    renderer = MudraRenderer(width=640, height=480)
    pose_tracker = PoseTracker(use_lite=True)
    pose_state = None
    prev_pose_state = None
    hand_speed = 0.0
    mandala = MandalaRenderer(width=640, height=480)
    last_mudra = None
    prev_left_shoulder_x  = None
    spin_cooldown          = 0
    mandala_visible   = False
    mandala_alpha     = 0.0
    mandala_scale     = 0.8   # starts small, grows while spinning
    mandala_expanding = False
    # Face mesh for skeleton mode
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mp_selfie = mp.solutions.selfie_segmentation
    selfie_seg = mp_selfie.SelfieSegmentation(model_selection=1)
    face_results = None
    # Visualization mode: 0=camera, 1=skeleton, 2=silhouette
    viz_mode = 0
    
    # Debug mode and freeze frame state
    debug_mode = False
    freeze_frame = False
    last_debug_info = None
    debug_info = None
    
    # FPS calculation
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
    
    print("Indian Classical Mudra Detection System Started")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Could not read frame from camera")
            # Try to reinitialize camera
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(0)  # Try to reconnect
            if not cap.isOpened():
                print("Error: Lost camera connection")
                break
            continue
        
        # Check if frame is valid
        if frame is None or frame.size == 0:
            print("Warning: Invalid frame received")
            continue
        
        # Flip frame horizontally for mirror view
        frame = cv2.flip(frame, 1)

        # Blend mandala into background BEFORE drawing anything on frame
        if viz_mode == 0 and mandala_visible and mandala_alpha > 0:
            mandala_layer = np.zeros_like(frame)
            # Scale the mandala renderer radius by mandala_scale
            orig_w = mandala.w
            orig_h = mandala.h
            mandala.w = int(640 * mandala_scale)
            mandala.h = int(480 * mandala_scale)
            mandala.render(mandala_layer)
            mandala.w = orig_w
            mandala.h = orig_h
            cv2.addWeighted(mandala_layer, mandala_alpha,
                            frame, 1.0 - mandala_alpha, 0, frame)
        # Update mandala every frame
        # Normalize hand speed to 0-1 for breathing
        movement_energy = min(hand_speed / 25.0, 1.0)
        mandala.update(movement_energy=movement_energy)
        # Store clean frame for skeleton/silhouette modes
        clean_frame = frame.copy()
        
        # Run pose detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose_tracker.process(frame_rgb)
        # Run face mesh in skeleton mode only (save CPU in other modes)
        if viz_mode == 1:
            face_results = face_mesh.process(frame_rgb)
        else:
            face_results = None
        # Run selfie segmentation in silhouette mode only
        if viz_mode == 2:
            seg_results = selfie_seg.process(frame_rgb)
        else:
            seg_results = None
        prev_pose_state = pose_state
        pose_state = pose_tracker.get_pose_state(
            pose_results, frame.shape[1], frame.shape[0])
        hand_speed = pose_tracker.get_hand_speed(
            pose_state, prev_pose_state,
            frame.shape[1], frame.shape[0])
        # Spin detection — left shoulder crossing right shoulder
        if pose_state and prev_pose_state:
            ls_x = pose_state.get('left_shoulder', (0,0,0))[0]
            rs_x = pose_state.get('right_shoulder', (0,0,0))[0]
            prev_ls_x = prev_pose_state.get('left_shoulder', (0,0,0))[0]
            if spin_cooldown <= 0:
                if prev_ls_x > rs_x and ls_x <= rs_x:
                    mandala.add_spin(0.25)
                    mandala_visible   = True
                    mandala_expanding = True
                    mandala_alpha     = 0.0
                    mandala_scale     = 0.8
                    spin_cooldown     = 15
                    print("Spin detected!")
            else:
                spin_cooldown -= 1

        # Animate mandala alpha
        if mandala_visible:
            if mandala_expanding:
                mandala_alpha = min(mandala_alpha + 0.03, 0.75)
                mandala_scale = min(mandala_scale + 0.020, 1.6)
                if mandala_alpha >= 0.45:
                    mandala_expanding = False
            else:
                mandala_alpha = max(mandala_alpha - 0.008, 0.0)
                mandala_scale = max(mandala_scale - 0.005, 0.0)
                if mandala_alpha <= 0.0:
                    mandala_visible = False
                    mandala_scale   = 0.8
        
        # Find hands
        results = hand_tracker.find_hands(frame)
        
        # Get hand landmarks and handedness
        hand_landmarks_list = hand_tracker.get_hand_landmarks(results)
        handedness_list = hand_tracker.get_handedness(results)
        
        # Draw landmarks on frame
        frame = hand_tracker.draw_landmarks(frame, results)
        # Apply visualization mode
        if viz_mode == 1:
            # Skeleton mode — black background, glowing skeletons
            frame = np.zeros_like(frame)
            # Draw pose skeleton on black background
            if pose_results and pose_results.pose_landmarks:
                mudra_color = MUDRA_THEMES.get(mudra, {}).get('color', (100, 255, 180)) if mudra != 'Unknown' else (80, 80, 80)
                connections = [
                    ('left_shoulder', 'right_shoulder'),
                    ('left_shoulder', 'left_elbow'),
                    ('left_elbow', 'left_wrist'),
                    ('right_shoulder', 'right_elbow'),
                    ('right_elbow', 'right_wrist'),
                    ('left_shoulder', 'left_hip'),
                    ('right_shoulder', 'right_hip'),
                    ('left_hip', 'right_hip'),
                ]
                key_idx = {
                    'left_shoulder': 11, 'right_shoulder': 12,
                    'left_elbow': 13, 'right_elbow': 14,
                    'left_wrist': 15, 'right_wrist': 16,
                    'left_hip': 23, 'right_hip': 24,
                }
                lm_pose = pose_results.pose_landmarks.landmark
                h2, w2 = frame.shape[:2]
                pts_pose = {name: (int(lm_pose[idx].x * w2), int(lm_pose[idx].y * h2))
                            for name, idx in key_idx.items()}
                for a, b in connections:
                    if a in pts_pose and b in pts_pose:
                        ov = frame.copy()
                        cv2.line(ov, pts_pose[a], pts_pose[b], mudra_color, 4)
                        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
                        cv2.line(frame, pts_pose[a], pts_pose[b], mudra_color, 1)
                for name, pt in pts_pose.items():
                    ov = frame.copy()
                    cv2.circle(ov, pt, 6, mudra_color, -1)
                    cv2.addWeighted(ov, 0.70, frame, 0.30, 0, frame)
            # Draw hand skeleton on black background
            if results.multi_hand_landmarks:
                for hand_lm in results.multi_hand_landmarks:
                    mudra_color = MUDRA_THEMES.get(mudra, {}).get('color', (100, 255, 180)) if mudra != 'Unknown' else (80, 80, 80)
                    h2, w2 = frame.shape[:2]
                    hand_connections = [
                        (0,1),(1,2),(2,3),(3,4),
                        (0,5),(5,6),(6,7),(7,8),
                        (0,9),(9,10),(10,11),(11,12),
                        (0,13),(13,14),(14,15),(15,16),
                        (0,17),(17,18),(18,19),(19,20),
                        (5,9),(9,13),(13,17),
                    ]
                    pts_hand = [(int(lm.x * w2), int(lm.y * h2))
                                for lm in hand_lm.landmark]
                    for a, b in hand_connections:
                        ov = frame.copy()
                        cv2.line(ov, pts_hand[a], pts_hand[b], mudra_color, 3)
                        cv2.addWeighted(ov, 0.60, frame, 0.40, 0, frame)
                        cv2.line(frame, pts_hand[a], pts_hand[b], mudra_color, 1)
                    for pt in pts_hand:
                        cv2.circle(frame, pt, 3, mudra_color, -1)
            # Draw face as glowing constellation dots in skeleton mode
            if face_results and face_results.multi_face_landmarks:
                h2, w2 = frame.shape[:2]
                face_lm = face_results.multi_face_landmarks[0].landmark
                # Key sparse landmark indices — eyes, nose, lips, jaw, brows
                sparse_idx = [
                    # Jaw outline
                    10, 338, 297, 332, 284, 251, 389, 356,
                    454, 323, 361, 288, 397, 365, 379, 378,
                    400, 377, 152, 148, 176, 149, 150, 136,
                    172, 58, 132, 93, 234, 127, 162, 21,
                    # Left eye
                    33, 160, 158, 133, 153, 144,
                    # Right eye
                    362, 385, 387, 263, 373, 380,
                    # Nose tip + bridge
                    1, 2, 98, 327,
                    # Lips
                    61, 291, 0, 17, 269, 39,
                    # Brows
                    70, 63, 105, 66, 107,
                    336, 296, 334, 293, 300,
                ]
                for idx in sparse_idx:
                    lx = int(face_lm[idx].x * w2)
                    ly = int(face_lm[idx].y * h2)
                    # Outer glow
                    ov = frame.copy()
                    cv2.circle(ov, (lx, ly), 5, mudra_color, -1)
                    cv2.addWeighted(ov, 0.20, frame, 0.80, 0, frame)
                    # Mid glow
                    ov2 = frame.copy()
                    cv2.circle(ov2, (lx, ly), 3, mudra_color, -1)
                    cv2.addWeighted(ov2, 0.45, frame, 0.55, 0, frame)
                    # Bright core dot
                    cv2.circle(frame, (lx, ly), 1,
                               (255, 255, 255), -1)

        elif viz_mode == 2:
            # Silhouette mode using MediaPipe Selfie Segmentation
            sil_frame = np.zeros_like(frame)
            mudra_color = MUDRA_THEMES.get(mudra, {}).get('color', (100, 255, 180)) if mudra != 'Unknown' else (60, 60, 60)

            if seg_results is not None and seg_results.segmentation_mask is not None:
                # Get clean binary mask
                mask = seg_results.segmentation_mask
                condition = mask > 0.55
                mask_uint8 = (condition * 255).astype(np.uint8)

                # Soften mask edges
                mask_blur = cv2.GaussianBlur(mask_uint8, (21, 21), 0)
                mask_norm = mask_blur.astype(np.float32) / 255.0

                # Fill silhouette with mudra color
                colored = np.zeros_like(frame, dtype=np.float32)
                for c in range(3):
                    colored[:, :, c] = mudra_color[c] * mask_norm

                # Outer glow — slightly expanded mask
                mask_dilated = cv2.dilate(mask_uint8,
                                          np.ones((25, 25), np.uint8),
                                          iterations=2)
                mask_glow = cv2.GaussianBlur(mask_dilated, (35, 35), 0)
                mask_glow_norm = mask_glow.astype(np.float32) / 255.0
                glow = np.zeros_like(frame, dtype=np.float32)
                for c in range(3):
                    glow[:, :, c] = mudra_color[c] * mask_glow_norm * 0.35

                sil_frame = np.clip(glow + colored, 0, 255).astype(np.uint8)

                # Bright edge outline
                edges = cv2.Canny(mask_uint8, 50, 150)
                edges_dilated = cv2.dilate(edges,
                                           np.ones((2, 2), np.uint8),
                                           iterations=1)
                sil_frame[edges_dilated > 0] = tuple(
                    min(255, int(c * 1.4)) for c in mudra_color)

            frame = sil_frame
        
        # Recognize mudras for each detected hand with proper handedness
        hand_data = []
        if results.multi_hand_landmarks:
            for i, hand_lm in enumerate(results.multi_hand_landmarks):
                lm_list = [(lm.x, lm.y, lm.z) for lm in hand_lm.landmark]
                handedness = results.multi_handedness[i].classification[0].label
                hand_data.append((lm_list, handedness))

        if len(hand_data) == 0:
            mudra, score = "Unknown", 0.0
        elif len(hand_data) == 1:
            if debug_mode:
                mudra, score, debug_info = mudra_recognizer.recognize_single(
                    hand_data[0][0], hand_data[0][1], debug=True)
                last_debug_info = debug_info  # cache it
            else:
                mudra, score = mudra_recognizer.recognize_single(
                    hand_data[0][0], hand_data[0][1])
        else:  # 2 hands
            mudra, score = mudra_recognizer.recognize_two_hand(hand_data[0], hand_data[1])
        
        # Update mandala pattern when mudra changes
        if mudra != last_mudra and mudra != 'Unknown':
            mandala.set_mudra(mudra)
            last_mudra = mudra
        
        # Draw mudra visual effects using new renderer
        if hand_data:
            second_lm = hand_data[1][0] \
                        if len(hand_data) > 1 else None
            frame = renderer.render(
                frame,
                mudra,
                score,
                hand_data[0][0],
                hand_data[0][1],
                second_landmarks=second_lm,
                pose_state=pose_state,
                hand_speed=hand_speed,
            )
        else:
            renderer.particles.update_draw(frame)
        
        # ── Visual Engine ──────────────────────────────
        # VISUALS TEMPORARILY DISABLED — re-enable after detection fixed
        # if hand_data:
        #     engine.update_hands(hand_data[0][0], hand_data[0][1])

        # FUTURE: uncomment when MediaPipe Pose is added
        # if pose_results and pose_results.pose_landmarks:
        #     pose_lm = [(lm.x, lm.y, lm.z)
        #                for lm in pose_results.pose_landmarks.landmark]
        #     engine.update_body(pose_lm)

        # if mudra != "Unknown":
        #     engine.set_mudra(mudra)

        # frame = engine.render(frame)
        # ── End Visual Engine ───────────────────────────
            
        # Display mudra name and confidence
        if mudra != "Unknown":
            label = f"{mudra}  {int(score*100)}%"
            cv2.putText(frame, label, (20, 60),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,255,180), 2)
        else:
            cv2.putText(frame, "Unknown", (20, 60),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,0,255), 2)
            
            # Draw visual effects for detected mudras
            if len(hand_data) == 1:  # Only draw effects for single hand
                lm, handedness = hand_data[0]
                wrist_x = int(lm[0][0] * frame.shape[1])
                wrist_y = int(lm[0][1] * frame.shape[0])
                
                if mudra == "Pataka":
                    # Draw special yellow circle with rays effect above the hand
                    effect_y = wrist_y - 80
                    frame = visual_effects.draw_pataka_effect(frame, wrist_x, effect_y, 40)
                elif mudra != "Unknown":
                    # Draw simple badge for other mudras
                    effect_y = wrist_y - 80
                    frame = visual_effects.draw_mudra_badge(frame, mudra, wrist_x, effect_y, 25)
            
            if debug_mode and last_debug_info:
                di = last_debug_info
                overlay = frame.copy()
                cv2.rectangle(overlay, (5, 85), (420, 340),
                              (0,0,0), -1)

                # Finger angles
                angles = di.get('finger_angles', {})
                a_line = (
                    f"T:{angles.get('thumb',0):.0f} "
                    f"I:{angles.get('index',0):.0f} "
                    f"M:{angles.get('middle',0):.0f} "
                    f"R:{angles.get('ring',0):.0f} "
                    f"P:{angles.get('pinky',0):.0f}"
                )

                # Build lines using CORRECT dict keys
                lines = [
                    a_line,
                    f"Thumb: {di.get('thumb_state','?').upper()}",
                    f"Groups: {di.get('stage1_groups',[])}",
                    f"HandSize: {di.get('hand_size',0):.3f}",
                    f"Detected: {mudra} ({score:.2f})",
                    "─── Top 5 ───────────────",
                ]

                top5 = di.get('top5', [])
                if top5:
                    for i, item in enumerate(top5[:5]):
                        # handle both (name, score) tuple and other formats
                        if isinstance(item, (list, tuple)) and len(item) == 2:
                            name, scr = item
                            bar = int(scr * 12)
                            lines.append(
                                f"{i+1}. {name}: {scr:.2f} "
                                f"{'|' * bar}")
                        else:
                            lines.append(f"{i+1}. {str(item)}")
                else:
                    lines.append("  (no scores yet)")

                for i, line in enumerate(lines):
                    cv2.putText(overlay, line,
                                (10, 105 + i * 22),
                                cv2.FONT_HERSHEY_PLAIN,
                                1.0,
                                (100, 255, 100), 1)

                cv2.addWeighted(overlay, 0.65,
                                frame, 0.35, 0, frame)
        
        # Calculate and display FPS
        fps_counter += 1
        if fps_counter >= 10:
            current_fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()
        
        # Display FPS
        fps_text = f"FPS: {current_fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display supported mudras
        mudras_text = "30+ Mudras: Pataka, Tripataka, Ardhapataka, Kartarimukha, Mayura, Ardhachandra, Arala, Shukatunda, Mushti, Shikhara, Kapitta, Katakamukha, Suchi, Chandrakala, Padmakosha, Sarpashirsha, Mrigashirsha, Simhamukha, Kangula, Alapadma, Chatura, Bhramara, Hamsasya, Hamsapaksha, Sandamsha, Mukula, Tamrachuda, Trishula"
        cv2.putText(frame, mudras_text[:120] + "...", (10, frame.shape[0] - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        # Display instructions
        modes = ['Camera', 'Skeleton', 'Silhouette']
        cv2.putText(frame, f"Press 'q' to quit | 'm' = mode: {modes[viz_mode]}", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display the frame
        cv2.imshow('Indian Classical Mudra Detection', frame)
        
        # Check for quit keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:  # q, Q, or ESC
            print("Quit key pressed - closing application...")
            break
        elif key == ord('d') or key == ord('D'):  # Toggle debug mode
            debug_mode = not debug_mode
            mudra_recognizer.debug_mode = debug_mode
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
        elif key == ord('f') or key == ord('F'):  # Freeze frame
            freeze_frame = not freeze_frame
            print(f"Frame frozen: {'YES' if freeze_frame else 'NO'}")
        elif key == ord('s') or key == ord('S'):  # Save debug snapshot
            if debug_info:
                with open('debug_log.txt', 'a') as f:
                    f.write(f"Debug Snapshot - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Mudra: {mudra}, Score: {score:.2f}\n")
                    f.write(f"Hand Size: {debug_info.get('hand_size', 0):.3f}\n")
                    f.write(f"Groups: {debug_info.get('stage1_groups', [])}\n")
                    f.write(f"Thumb State: {debug_info.get('thumb_state', 'unknown')}\n")
                    angles = debug_info.get('finger_angles', {})
                    f.write(f"Angles: T:{angles.get('thumb', 0):.0f} I:{angles.get('index', 0):.0f} M:{angles.get('middle', 0):.0f} R:{angles.get('ring', 0):.0f} P:{angles.get('pinky', 0):.0f}\n")
                    if 'top5' in debug_info:
                        f.write("Top 5 Scores:\n")
                        for i, (name, scr) in enumerate(debug_info['top5']):
                            f.write(f"  {i+1}. {name}: {scr:.2f}\n")
                print("Debug snapshot saved to debug_log.txt")
            else:
                print("Debug mode not enabled - press 'd' to enable")
        elif key == ord('t') or key == ord('T'):  # Toggle Samyuktha Hastas
            mudra_recognizer.samyuktha_enabled = not mudra_recognizer.samyuktha_enabled
            print(f"Samyuktha Hastas: {'ON' if mudra_recognizer.samyuktha_enabled else 'OFF'}")
        elif key == ord('m') or key == ord('M'):
            viz_mode = (viz_mode + 1) % 3
            modes = ['Camera', 'Skeleton', 'Silhouette']
            print(f"Visualization mode: {modes[viz_mode]}")
    
    # Cleanup
    print("Cleaning up resources...")
    engine.clear()
    
    # Release camera
    if cap is not None and cap.isOpened():
        cap.release()
        print("Camera released")
    
    # Close all OpenCV windows
    cv2.destroyAllWindows()
    
    # Force close any remaining windows
    for i in range(4):
        cv2.waitKey(1)
    
    print("Mudra Detection System Stopped")


if __name__ == "__main__":
    main()
