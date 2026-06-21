"""Model Loading Manager with Lazy Loading"""

from ultralytics import YOLO
from .config import Config
import gc


class ModelManager:
    """Lazy-loads and caches all models to avoid redundant loading"""
    
    def __init__(self):
        self.bdd100k_model = None
        self.uvh_model = None
        self.license_plate_model = None
        self.helmet_model = None
        self.seatbelt_model = None
    
    def load_bdd100k(self):
        """Load BDD100K object detection model (lazy)"""
        if self.bdd100k_model is None:
            self.bdd100k_model = YOLO(Config.MODEL_PATHS["bdd100k"])
        return self.bdd100k_model
    
    def load_uvh(self):
        """Load UVH vehicle detection model (lazy) - for 3-wheelers"""
        if self.uvh_model is None:
            self.uvh_model = YOLO(Config.MODEL_PATHS["uvh"])
        return self.uvh_model
    
    def load_license_plate(self):
        """Load license plate detection model (lazy)"""
        if self.license_plate_model is None:
            self.license_plate_model = YOLO(Config.MODEL_PATHS["license_plate"])
        return self.license_plate_model
    
    def load_helmet(self):
        """Load helmet detection model (lazy)"""
        if self.helmet_model is None:
            self.helmet_model = YOLO(Config.MODEL_PATHS["helmet"])
        return self.helmet_model
    
    def load_seatbelt(self):
        """Load seatbelt detection model (lazy)"""
        if self.seatbelt_model is None:
            self.seatbelt_model = YOLO(Config.MODEL_PATHS["seatbelt"])
        return self.seatbelt_model
