import cv2
import math
import time
from hand_tracker import HandTracker
from mudra_recognizer import MudraRecognizer
from visual_effects import VisualEffects
from visual_engine import VisualEngine
from renderer import MudraRenderer

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
        
        # Find hands
        results = hand_tracker.find_hands(frame)
        
        # Get hand landmarks and handedness
        hand_landmarks_list = hand_tracker.get_hand_landmarks(results)
        handedness_list = hand_tracker.get_handedness(results)
        
        # Draw landmarks on frame
        frame = hand_tracker.draw_landmarks(frame, results)
        
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
        
        # Draw mudra visual effects using new renderer
        if hand_data:
            frame = renderer.render(
                frame,
                mudra,
                score,
                hand_data[0][0],
                hand_data[0][1]
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
        cv2.putText(frame, "Press 'q' or ESC to quit | Debug mode ON", (10, frame.shape[0] - 20),
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
