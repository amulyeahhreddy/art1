import math
from collections import deque


class MudraRecognizer:
    def __init__(self):
        self.finger_tip_indices = {
            'thumb': 4,
            'index': 8,
            'middle': 12,
            'ring': 16,
            'pinky': 20
        }
        
        self.finger_pip_indices = {
            'thumb': 3,
            'index': 6,
            'middle': 10,
            'ring': 14,
            'pinky': 18
        }
        
        self.finger_mcp_indices = {
            'thumb': 2,
            'index': 5,
            'middle': 9,
            'ring': 13,
            'pinky': 17
        }
        
        # Temporal smoothing buffers for each hand (increased to 7 frames)
        self.left_hand_history = deque(maxlen=7)
        self.right_hand_history = deque(maxlen=7)
        self.debug_mode = True
        
        # Initialize mudra rules dictionary
        self._init_mudra_rules()
        
        # Store last features for debugging
        self.last_features = {}
        self.last_confidence = 0.0
    
    def calculate_distance(self, point1, point2):
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def normalize_hand_orientation(self, landmarks, hand_type):
        """Normalize landmarks so both hands behave like right hand"""
        if hand_type == "Left":
            # Mirror x coordinates around palm center
            palm_center_x = landmarks[0][0]
            normalized_landmarks = []
            for landmark in landmarks:
                normalized_x = palm_center_x - (landmark[0] - palm_center_x)
                normalized_landmarks.append([normalized_x, landmark[1], landmark[2]])
            return normalized_landmarks
        return landmarks
    
    def calculate_angle(self, p1, p2, p3):
        """Calculate angle at p2 between p1-p2-p3 using cosine law"""
        # Vector from p2 to p1
        v1 = [p1[0] - p2[0], p1[1] - p2[1]]
        # Vector from p2 to p3
        v2 = [p3[0] - p2[0], p3[1] - p2[1]]
        
        # Calculate dot product
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        
        # Calculate magnitudes
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag1 == 0 or mag2 == 0:
            return 180  # Default to extended
        
        # Calculate angle in degrees
        cos_angle = dot_product / (mag1 * mag2)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp to [-1, 1]
        angle = math.degrees(math.acos(cos_angle))
        
        return angle
    
    def get_finger_angle(self, landmarks, finger_name):
        """Calculate angle at PIP joint for finger extension detection"""
        mcp_idx = self.finger_mcp_indices[finger_name]
        pip_idx = self.finger_pip_indices[finger_name]
        tip_idx = self.finger_tip_indices[finger_name]
        
        mcp = landmarks[mcp_idx]
        pip = landmarks[pip_idx]
        tip = landmarks[tip_idx]
        
        return self.calculate_angle(mcp, pip, tip)
    
    def get_finger_state_from_angle(self, angle):
        """Determine finger state from angle"""
        if angle > 160:
            return "extended"
        elif angle < 140:
            return "bent"
        else:
            return "half-bent"
    
    def extract_hand_features(self, landmarks):
        """Extract comprehensive hand features for mudra recognition"""
        features = {
            'finger_states': {},
            'finger_angles': {},
            'finger_spread': 0,
            'thumb_index_distance': 0,
            'palm_orientation': 0
        }
        
        # Extract finger states and angles
        for finger_name in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            angle = self.get_finger_angle(landmarks, finger_name)
            state = self.get_finger_state_from_angle(angle)
            
            features['finger_angles'][finger_name] = angle
            features['finger_states'][finger_name] = state
        
        # Calculate finger spread
        features['finger_spread'] = self.get_finger_spread(landmarks)
        
        # Calculate thumb-index distance
        thumb_tip = landmarks[self.finger_tip_indices['thumb']]
        index_tip = landmarks[self.finger_tip_indices['index']]
        features['thumb_index_distance'] = self.calculate_distance(thumb_tip, index_tip)
        
        # Calculate palm orientation (simplified)
        wrist = landmarks[0]
        middle_mcp = landmarks[self.finger_mcp_indices['middle']]
        features['palm_orientation'] = math.atan2(middle_mcp[1] - wrist[1], middle_mcp[0] - wrist[0])
        
        return features
    
    def _init_mudra_rules(self):
        """Initialize comprehensive mudra rules dictionary"""
        self.mudra_rules = {
            "Pataka": {
                "extended": ["thumb", "index", "middle", "ring", "pinky"]
            },
            "Tripataka": {
                "extended": ["thumb", "index", "middle", "pinky"],
                "bent": ["ring"]
            },
            "Ardhapataka": {
                "extended": ["thumb", "index", "middle", "ring"],
                "bent": ["pinky"]
            },
            "Kartarimukha": {
                "extended": ["thumb", "index", "middle"],
                "bent": ["ring", "pinky"],
                "thumb_index_touch": True
            },
            "Mayura": {
                "extended": ["thumb", "pinky"],
                "bent": ["index", "middle", "ring"],
                "thumb_pinky_touch": True
            },
            "Ardhachandra": {
                "extended": ["thumb", "index", "middle", "ring", "pinky"],
                "thumb_extended_outward": True
            },
            "Arala": {
                "extended": ["thumb", "index", "middle", "ring"],
                "half-bent": ["pinky"]
            },
            "Shukatunda": {
                "extended": ["thumb", "index"],
                "bent": ["middle", "ring", "pinky"]
            },
            "Mushti": {
                "bent": ["thumb", "index", "middle", "ring", "pinky"]
            },
            "Shikhara": {
                "extended": ["thumb"],
                "bent": ["index", "middle", "ring", "pinky"]
            },
            "Kapitta": {
                "extended": ["thumb"],
                "half-bent": ["index", "middle", "ring", "pinky"],
                "thumb_index_touch": True
            },
            "Katakamukha": {
                "extended": ["index", "middle"],
                "bent": ["thumb", "ring", "pinky"],
                "thumb_middle_touch": True
            },
            "Suchi": {
                "extended": ["thumb", "index"],
                "bent": ["middle", "ring", "pinky"]
            },
            "Chandrakala": {
                "extended": ["thumb", "index"],
                "bent": ["middle", "ring", "pinky"],
                "thumb_index_circle": True
            },
            "Padmakosha": {
                "fingertips_touch": True
            },
            "Sarpashirsha": {
                "extended": ["index", "middle", "ring", "pinky"],
                "bent": ["thumb"],
                "fingers_curved": True
            },
            "Mrigashirsha": {
                "extended": ["thumb", "index", "middle"],
                "bent": ["ring", "pinky"],
                "thumb_middle_touch": True
            },
            "Simhamukha": {
                "extended": ["thumb", "index", "middle"],
                "bent": ["ring", "pinky"],
                "thumb_index_spread": True
            },
            "Kangula": {
                "extended": ["index"],
                "bent": ["thumb", "middle", "ring", "pinky"],
                "thumb_curved": True
            },
            "Alapadma": {
                "extended": ["thumb", "index", "middle", "ring", "pinky"],
                "finger_spread_high": True
            },
            "Chatura": {
                "extended": ["thumb", "index", "middle", "pinky"],
                "bent": ["ring"]
            },
            "Bhramara": {
                "extended": ["thumb", "index", "middle", "ring"],
                "bent": ["pinky"],
                "thumb_middle_touch": True,
                "fingers_curved": True
            },
            "Hamsasya": {
                "thumb_index_touch": True,
                "extended": ["middle", "ring", "pinky"]
            },
            "Hamsapaksha": {
                "extended": ["thumb", "index", "middle"],
                "bent": ["ring", "pinky"],
                "thumb_index_angle": 45
            },
            "Sandamsha": {
                "extended": ["thumb", "index"],
                "bent": ["middle", "ring", "pinky"],
                "thumb_index_opposite": True
            },
            "Mukula": {
                "fingertips_touch": True
            },
            "Tamrachuda": {
                "extended": ["thumb", "index", "middle"],
                "bent": ["ring", "pinky"],
                "thumb_curved": True
            },
            "Trishula": {
                "extended": ["thumb", "index", "middle"],
                "bent": ["ring", "pinky"],
                "thumb_spread": True
            }
        }
    
    def check_mudra_rule(self, mudra_name, features, landmarks):
        """Check if features match a specific mudra rule"""
        rule = self.mudra_rules[mudra_name]
        finger_states = features['finger_states']
        
        # Check finger state requirements
        if 'extended' in rule:
            for finger in rule['extended']:
                if finger_states.get(finger) != 'extended':
                    return False
        
        if 'bent' in rule:
            for finger in rule['bent']:
                if finger_states.get(finger) != 'bent':
                    return False
        
        if 'half-bent' in rule:
            for finger in rule['half-bent']:
                if finger_states.get(finger) != 'half-bent':
                    return False
        
        # Check special conditions
        if rule.get('thumb_index_touch'):
            if not self.is_thumb_touching_index(landmarks):
                return False
        
        if rule.get('thumb_pinky_touch'):
            if not self.is_thumb_touching_pinky(landmarks):
                return False
        
        if rule.get('thumb_middle_touch'):
            if not self.is_thumb_touching_middle(landmarks):
                return False
        
        if rule.get('fingertips_touch'):
            if not self.are_fingertips_touching(landmarks):
                return False
        
        if rule.get('thumb_extended_outward'):
            if not self.is_thumb_extended_outward(landmarks):
                return False
        
        if rule.get('finger_spread_high'):
            if features['finger_spread'] <= 0.12:
                return False
        
        if rule.get('fingers_curved'):
            # Check if fingers are in curved position
            if not self.are_fingers_curved(landmarks):
                return False
        
        return True
    
    def is_thumb_touching_pinky(self, landmarks):
        """Check if thumb tip is touching pinky tip"""
        thumb_tip = landmarks[self.finger_tip_indices['thumb']]
        pinky_tip = landmarks[self.finger_tip_indices['pinky']]
        distance = self.calculate_distance(thumb_tip, pinky_tip)
        return distance < 0.05
    
    def is_thumb_touching_middle(self, landmarks):
        """Check if thumb tip is touching middle tip"""
        thumb_tip = landmarks[self.finger_tip_indices['thumb']]
        middle_tip = landmarks[self.finger_tip_indices['middle']]
        distance = self.calculate_distance(thumb_tip, middle_tip)
        return distance < 0.05
    
    def are_fingers_curved(self, landmarks):
        """Check if fingers are in curved position"""
        curved_count = 0
        for finger in ['index', 'middle', 'ring', 'pinky']:
            angle = self.get_finger_angle(landmarks, finger)
            if 120 < angle < 160:  # Curved but not fully bent
                curved_count += 1
        return curved_count >= 2
    
    def get_finger_spread(self, landmarks):
        index_tip = landmarks[self.finger_tip_indices['index']]
        middle_tip = landmarks[self.finger_tip_indices['middle']]
        ring_tip = landmarks[self.finger_tip_indices['ring']]
        pinky_tip = landmarks[self.finger_tip_indices['pinky']]
        
        spread_1 = self.calculate_distance(index_tip, middle_tip)
        spread_2 = self.calculate_distance(middle_tip, ring_tip)
        spread_3 = self.calculate_distance(ring_tip, pinky_tip)
        
        return (spread_1 + spread_2 + spread_3) / 3
    
    def is_thumb_touching_index(self, landmarks):
        """Check if thumb tip is touching index tip (for Hamsasya)"""
        thumb_tip = landmarks[self.finger_tip_indices['thumb']]
        index_tip = landmarks[self.finger_tip_indices['index']]
        distance = self.calculate_distance(thumb_tip, index_tip)
        return distance < 0.05
    
    def are_fingertips_touching(self, landmarks):
        """Check if all fingertips are close together (for Mukula)"""
        thumb_tip = landmarks[self.finger_tip_indices['thumb']]
        index_tip = landmarks[self.finger_tip_indices['index']]
        middle_tip = landmarks[self.finger_tip_indices['middle']]
        ring_tip = landmarks[self.finger_tip_indices['ring']]
        pinky_tip = landmarks[self.finger_tip_indices['pinky']]
        
        # Calculate distances between all fingertip pairs
        distances = [
            self.calculate_distance(thumb_tip, index_tip),
            self.calculate_distance(index_tip, middle_tip),
            self.calculate_distance(middle_tip, ring_tip),
            self.calculate_distance(ring_tip, pinky_tip)
        ]
        
        # All distances should be small for fingertips to be touching
        return all(dist < 0.08 for dist in distances)
    
    def is_thumb_extended_outward(self, landmarks):
        """Check if thumb is extended outward from hand"""
        thumb_tip = landmarks[self.finger_tip_indices['thumb']]
        index_tip = landmarks[self.finger_tip_indices['index']]
        
        thumb_index_distance = self.calculate_distance(thumb_tip, index_tip)
        return thumb_index_distance > 0.15
    
    def apply_temporal_smoothing(self, hand_type, current_mudra):
        """Apply temporal smoothing with 7-frame window and majority voting"""
        if hand_type == 'Left':
            self.left_hand_history.append(current_mudra)
            history = self.left_hand_history
        else:
            self.right_hand_history.append(current_mudra)
            history = self.right_hand_history
        
        if len(history) < 4:
            return current_mudra
        
        # Use majority voting for stability
        mudra_counts = {}
        for mudra in history:
            mudra_counts[mudra] = mudra_counts.get(mudra, 0) + 1
        
        # Return the most frequent mudra
        most_frequent = max(mudra_counts.items(), key=lambda x: x[1])[0]
        confidence = mudra_counts[most_frequent] / len(history)
        
        # Store confidence for debugging
        self.last_confidence = confidence
        
        return most_frequent
    
    def recognize_mudra(self, landmarks, hand_type='Right'):
        """Main mudra recognition function with improved accuracy"""
        if not landmarks or len(landmarks) != 21:
            return "Unknown"
        
        # Step 1: Normalize hand orientation for consistent behavior
        normalized_landmarks = self.normalize_hand_orientation(landmarks, hand_type)
        
        # Step 2: Extract comprehensive features
        features = self.extract_hand_features(normalized_landmarks)
        
        # Store features for debugging
        self.last_features = features
        
        # Step 3: Check mudra rules
        for mudra_name in self.mudra_rules.keys():
            if self.check_mudra_rule(mudra_name, features, normalized_landmarks):
                # Apply temporal smoothing
                return self.apply_temporal_smoothing(hand_type, mudra_name)
        
        # No mudra matched
        return self.apply_temporal_smoothing(hand_type, "Unknown")
    
    def get_debug_info(self):
        """Get comprehensive debugging information"""
        debug_info = {
            'finger_states': self.last_features.get('finger_states', {}),
            'finger_angles': self.last_features.get('finger_angles', {}),
            'confidence': self.last_confidence,
            'finger_spread': self.last_features.get('finger_spread', 0),
            'thumb_index_distance': self.last_features.get('thumb_index_distance', 0)
        }
        return debug_info
