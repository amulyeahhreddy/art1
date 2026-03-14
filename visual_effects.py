import cv2
import numpy as np
import math


# Color-coded mudra categories (BGR tuples)
MUDRA_COLORS = {
    # Gold/Yellow — auspicious
    'Pataka': (0, 215, 255), 'Alapadma': (0, 200, 255),
    'Mukula': (0, 210, 240), 'Padmakosha': (0, 195, 255),
    
    # Green — nature/animals
    'Sarpashirsha': (0, 180, 60), 'Mrigashirsha': (0, 160, 50),
    'Mayura': (0, 200, 80), 'Bhramara': (40, 170, 40),
    'Simhamukha': (20, 150, 30),
    
    # Red/Orange — powerful
    'Mushti': (0, 60, 220), 'Shikhara': (0, 40, 200),
    'Trishula': (0, 80, 210), 'Suchi': (0, 100, 230),
    'Kartarimukha': (20, 70, 200), 'Tamrachuda': (0, 120, 240),
    
    # Blue/Purple — divine
    'Abhaya': (200, 100, 0), 'Varada': (180, 80, 20),
    'Dhyana': (210, 120, 0), 'Anjali': (190, 90, 10),
    'Shivalinga': (170, 60, 30),
    
    # Pink/Lavender — graceful
    'Hamsasya': (180, 100, 200), 'Hamsapaksha': (190, 120, 210),
    'Arala': (170, 90, 190), 'Chandrakala': (200, 130, 220),
    'Sandamsha': (160, 80, 180),
    
    # Teal — general
    'DEFAULT': (140, 140, 140),
}


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
        """Draw colored badge for any detected mudra with confidence bar"""
        # Get color for this mudra
        color = MUDRA_COLORS.get(mudra_name, MUDRA_COLORS['DEFAULT'])
        
        # Draw filled rounded rectangle background
        cv2.rectangle(frame, 
                     (center_x - size - 5, center_y - size - 25),
                     (center_x + size + 5, center_y + 5), color, -1)
        
        # Draw white text mudra name
        cv2.putText(frame, mudra_name, 
                    (center_x, center_y - 8), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw confidence bar (if score provided)
        if len(mudra_name.split()) > 1:  # Check if score is included in name
            parts = mudra_name.split()
            if len(parts) == 3 and parts[1].isdigit():  # Extract score from name like "Pataka 85%"
                try:
                    score = float(parts[1].replace('%', '')) / 100.0
                    # Draw confidence bar below name
                    bar_width = int((center_x + size + 5) - (center_x - size - 5))
                    bar_fill_width = int(bar_width * score)
                    
                    cv2.rectangle(frame, 
                                 (center_x - size - 5, center_y + 12),
                                 (center_x - size - 5 + bar_fill_width, center_y + 15), color, -1)
                    cv2.rectangle(frame, 
                                 (center_x - size - 5 + bar_fill_width, center_y + 12),
                                 (center_x + size + 5, center_y + 15), (255, 255, 255), 1)
                except:
                    pass
        
        return frame
