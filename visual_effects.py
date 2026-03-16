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
        return frame
    
    def draw_mudra_badge(self, frame, mudra_name, center_x, center_y, size=40):
        return frame
