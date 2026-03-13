import cv2
import numpy as np
import math


class VisualEffects:
    def __init__(self):
        self.pataka_animation_frame = 0
        self.pataka_animation_active = False
        
    def draw_pataka_effect(self, frame, center_x, center_y, radius=50):
        """Draw yellow circle with rays for Pataka mudra detection"""
        # Animation frame counter
        self.pataka_animation_frame += 1
        if self.pataka_animation_frame > 20:
            self.pataka_animation_frame = 0
        
        # Draw main yellow circle
        cv2.circle(frame, (center_x, center_y), radius, (0, 255, 255), -1)
        
        # Draw rays emanating from the circle
        num_rays = 12
        for i in range(num_rays):
            angle = (2 * math.pi * i) / num_rays
            
            # Animated ray length
            base_length = radius + 20
            animation_offset = 10 * math.sin(self.pataka_animation_frame * 0.3 + i * 0.5)
            ray_length = base_length + animation_offset
            
            # Calculate ray end point
            end_x = int(center_x + ray_length * math.cos(angle))
            end_y = int(center_y + ray_length * math.sin(angle))
            
            # Draw ray as a line
            cv2.line(frame, (center_x, center_y), (end_x, end_y), (0, 255, 255), 3)
            
            # Draw ray tip as a small circle
            cv2.circle(frame, (end_x, end_y), 4, (0, 200, 200), -1)
        
        # Draw inner bright circle
        cv2.circle(frame, (center_x, center_y), radius // 3, (255, 255, 0), -1)
        
        # Draw "PATAKA" text
        font_scale = 0.6
        text = "PATAKA"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        text_x = center_x - text_size[0] // 2
        text_y = center_y + radius + 30
        
        # Draw text background
        cv2.rectangle(frame, 
                     (text_x - 5, text_y - text_size[1] - 5),
                     (text_x + text_size[0] + 5, text_y + 5),
                     (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(frame, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 2)
        
        return frame
    
    def draw_mudra_badge(self, frame, mudra_name, center_x, center_y, size=40):
        """Draw a small badge for any detected mudra"""
        if mudra_name == "Pataka":
            return self.draw_pataka_effect(frame, center_x, center_y, size)
        elif mudra_name != "Unknown":
            # Draw a simple colored circle for other mudras
            color_map = {
                "Tripataka": (255, 100, 100),
                "Alapadma": (100, 255, 100),
                "Ardhachandra": (100, 100, 255),
                "Shikhara": (255, 255, 100),
                "Hamsasya": (255, 100, 255),
                "Mukula": (100, 255, 255),
                "Mushti": (200, 200, 200)
            }
            
            color = color_map.get(mudra_name, (128, 128, 128))
            
            # Draw circle
            cv2.circle(frame, (center_x, center_y), size, color, -1)
            cv2.circle(frame, (center_x, center_y), size, (255, 255, 255), 2)
            
            # Draw mudra name
            font_scale = 0.4
            text_size = cv2.getTextSize(mudra_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            text_x = center_x - text_size[0] // 2
            text_y = center_y + size + 15
            
            cv2.putText(frame, mudra_name, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
        
        return frame
