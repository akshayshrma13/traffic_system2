"""OCR Processing Module - License Plate Text Extraction with Strict Validation (v3)

Version 3 improvements:
- Enable RapidOCR detection mode for robustness (handles rotated/skewed plates)
- Add comprehensive logging for debugging validation failures
- Safer config handling with fallback defaults
- Enhanced OCR output parsing with debug info
- Better error handling and recovery
"""

import cv2
import numpy as np
import re
import logging
from rapidocr_onnxruntime import RapidOCR
from pydantic import BaseModel, field_validator, ValidationError
from typing import Optional
from .config import Config

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Valid Indian state codes (29 states + 8 UTs + special categories)
INDIAN_STATE_CODES = {
    "AP", "AR", "AS", "BR", "CT", "GA", "GJ", "HR", "HP", "JH", "JK", "KA", "KL", "LD", "MH",
    "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TG", "TN", "TR", "UP", "WB",
    "AN", "CH", "DD", "DN", "DL", "LA", "TS"
}

# Bidirectional maps for OCR misreads context-aware substitution
ALPHA_TO_DIGIT = {"O": "0", "I": "1", "S": "5", "B": "8", "Z": "2", "G": "6", "T": "1"}
DIGIT_TO_ALPHA = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G"}


class IndianLicensePlate(BaseModel):
    """Pydantic model for Indian License Plate validation and fuzzy correction"""
    plate_number: str

    @field_validator('plate_number')
    @classmethod
    def validate_plate_format(cls, v: str) -> str:
        """
        Validate and correct Indian license plate format.
        
        Format: State(2) + District(2) + Optional Vehicle Class(1-2 Alphas) + Registration Number(1-4 Digits) + Optional(Alphas)
        
        Features:
        - Sliding window detection for state codes (handles leading noise)
        - OCR misread correction for district digits (O->0, I->1, etc.)
        - Structural regex validation
        """
        # 1. Base Cleanup
        v = "".join([c for c in v if c.isalnum()]).upper()
        
        # 2. Sliding Window Matching for Leading Noise (e.g., 'RKA02...' -> 'KA02...')
        matched_state = None
        start_idx = 0
        
        for i in range(min(3, len(v) - 1)):
            candidate = v[i:i+2]
            # Try applying correction mapping to the candidate characters
            corrected_candidate = "".join([DIGIT_TO_ALPHA.get(c, c) for c in candidate])
            
            if corrected_candidate in INDIAN_STATE_CODES:
                matched_state = corrected_candidate
                start_idx = i
                break
            elif candidate in INDIAN_STATE_CODES:
                matched_state = candidate
                start_idx = i
                break
        
        if matched_state:
            v = v[start_idx:]
            v = matched_state + v[2:]
        else:
            raise ValueError(f"No valid Indian state code found in sequence: {v}")

        if len(v) < 6:
            raise ValueError("License plate is too short. Expected at least 6 characters.")

        # 3. Contextual Correction for District Code (Characters 3 and 4 must be digits)
        district_part = list(v[2:4])
        for idx, char in enumerate(district_part):
            if not char.isdigit() and char in ALPHA_TO_DIGIT:
                district_part[idx] = ALPHA_TO_DIGIT[char]
        v = v[:2] + "".join(district_part) + v[4:]

        if not v[2:4].isdigit():
            raise ValueError(f"Characters 3-4 must be numeric district codes. Got: {v[2:4]}")

        # 4. Final Format Check using Regex
        pattern = r"^[A-Z]{2}[0-9]{2}[A-Z]{0,2}[0-9]{1,4}[A-Z]{0,2}$"
        if not re.match(pattern, v):
            raise ValueError(f"Plate failed structural pattern validation: {v}")

        return v


class OCRProcessor:
    """Handles license plate text extraction using RapidOCR with strict validation (v3)"""
    
    def __init__(self):
        """Initialize RapidOCR engine with optimal settings"""
        self.ocr_engine = RapidOCR()
        
        # Safe config retrieval with sensible defaults
        self.border_px = self._get_config("white_border_px", 12)
        self.use_detection = self._get_config("use_ocr_detection", True)
        self.use_classification = self._get_config("use_ocr_classification", False)
        
        logger.info(
            f"OCRProcessor initialized: border_px={self.border_px}, "
            f"use_detection={self.use_detection}, use_classification={self.use_classification}"
        )
    
    def _get_config(self, key: str, default=None):
        """Safely retrieve config with fallback to default"""
        try:
            return Config.OCR_PARAMS.get(key, default)
        except (AttributeError, KeyError):
            logger.warning(f"Config key '{key}' not found, using default: {default}")
            return default
    
    def extract_plate_text(self, image, plate_boxes):
        """
        Extract text from detected license plates with STRICT validation.
        
        ONLY valid plates are added to results.
        Invalid plates are NOT stored (not added to results list).
        
        Features (v3):
        - Enabled RapidOCR detection mode for rotated/skewed plate robustness
        - Comprehensive debug logging for validation failures
        - Safe config handling with fallback defaults
        - Enhanced error recovery
        
        Args:
            image: Original image (BGR, not preprocessed)
            plate_boxes: List of plate bounding boxes [{"box": [x1,y1,x2,y2], "confidence": 0.88}, ...]
        
        Returns:
            List of VALIDATED extracted texts only
            [{"box": [x1,y1,x2,y2], "text": "KA02ME3547", "confidence": 0.92, "validated": True}, ...]
            
            Invalid detections are silently dropped (not returned).
        """
        results = []
        plate_count = 0
        valid_count = 0
        
        for plate_idx, plate_box in enumerate(plate_boxes):
            try:
                plate_count += 1
                box = plate_box["box"]
                detection_confidence = plate_box.get("confidence", 0.0)
                
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                
                logger.debug(f"[Plate {plate_idx}] Box: ({x1},{y1}) -> ({x2},{y2}), "
                           f"Detection conf: {detection_confidence:.3f}")
                
                # Crop plate region from image
                plate_crop = image[y1:y2, x1:x2]
                
                if plate_crop.size == 0:
                    logger.warning(f"[Plate {plate_idx}] Empty crop (size=0), skipping")
                    continue
                
                # Log crop dimensions
                crop_h, crop_w = plate_crop.shape[:2]
                logger.debug(f"[Plate {plate_idx}] Crop size: {crop_w}x{crop_h}")
                
                # Add white border padding (prevents edge characters from being cut off)
                plate_with_border = cv2.copyMakeBorder(
                    plate_crop,
                    self.border_px, self.border_px, self.border_px, self.border_px,
                    cv2.BORDER_CONSTANT,
                    value=(255, 255, 255)  # White border for all channels (BGR)
                )
                
                # Run RapidOCR on padded crop
                # v3: ENABLE DETECTION MODE for robustness against rotated/skewed plates
                ocr_output, _ = self.ocr_engine(
                    plate_with_border,
                    use_det=self.use_detection,        # v3: True (was False in v2)
                    use_cls=self.use_classification,   # Keep False for speed
                    use_rec=True                        # Always True
                )
                
                if ocr_output:
                    logger.debug(f"[Plate {plate_idx}] RapidOCR output structures: {len(ocr_output)} detections")
                    
                    extracted_texts = []

                    for det_idx, detection in enumerate(ocr_output):
                        try:
                            text = None
                            confidence = 0.0
                            
                            # v3: Enhanced parsing with better debugging
                            if len(detection) >= 2:
                                # Try Structure 1: [text, confidence]
                                if isinstance(detection[0], str):
                                    text = detection[0]
                                    confidence = float(detection[1]) if len(detection) > 1 else 0.0
                                    logger.debug(f"[Plate {plate_idx}][Det {det_idx}] Structure 1: text='{text}', conf={confidence:.3f}")
                                
                                # Try Structure 2: [box, text, confidence]
                                elif len(detection) >= 3:
                                    text = detection[1]
                                    confidence = float(detection[2]) if isinstance(detection[2], (int, float)) else 0.0
                                    logger.debug(f"[Plate {plate_idx}][Det {det_idx}] Structure 2: text='{text}', conf={confidence:.3f}")
                                
                                # Fallback: Try to extract any string
                                elif isinstance(detection[1], str):
                                    text = detection[1]
                                    confidence = float(detection[2]) if len(detection) > 2 else 0.0
                                    logger.debug(f"[Plate {plate_idx}][Det {det_idx}] Fallback: text='{text}', conf={confidence:.3f}")
                            
                            if text:
                                extracted_texts.append((text, confidence))
                        
                        except Exception as e:
                            logger.warning(f"[Plate {plate_idx}][Det {det_idx}] Parse error: {e}, detection={detection}")
                            continue
                    
                    if extracted_texts:
                        # Combine all detected texts
                        combined_text = "".join([text for text, _ in extracted_texts])
                        avg_confidence = np.mean([conf for _, conf in extracted_texts])
                        
                        logger.debug(f"[Plate {plate_idx}] Raw OCR: '{combined_text}', avg_conf={avg_confidence:.3f}")
                        
                        # Clean text: uppercase, alphanumeric only
                        cleaned_text = self._clean_plate_text(combined_text)
                        
                        logger.debug(f"[Plate {plate_idx}] Cleaned: '{cleaned_text}'")
                        
                        if cleaned_text:
                            # STRICT VALIDATION: Only store if validation succeeds
                            validated_plate = self._validate_indian_license_plate(cleaned_text)
                            
                            # ✓ ONLY add to results if validation PASSED
                            # ✗ If validation FAILED, do NOT add to results (flagged as NOT_DETECTED implicitly)
                            if validated_plate is not None:
                                logger.info(f"[Plate {plate_idx}] ✓ VALID: '{cleaned_text}' → '{validated_plate}'")
                                valid_count += 1
                                results.append({
                                    "box": box,
                                    "text": validated_plate,
                                    "confidence": avg_confidence,
                                    "validated": True,
                                    "validation_status": "VALID",
                                    "detection_confidence": detection_confidence
                                })
                            else:
                                logger.warning(f"[Plate {plate_idx}] ✗ INVALID: '{cleaned_text}' (validation failed)")
                        else:
                            logger.warning(f"[Plate {plate_idx}] ✗ EMPTY after cleaning: '{combined_text}'")
                else:
                    logger.warning(f"[Plate {plate_idx}] No OCR output (RapidOCR returned empty)")
            
            except Exception as e:
                logger.error(f"[Plate {plate_idx}] Unexpected error: {e}", exc_info=True)
                continue
        
        logger.info(f"OCR Extraction complete: {plate_count} plates detected, {valid_count} validated successfully")
        return results
    
    def _clean_plate_text(self, text):
        """
        Clean extracted license plate text.
        
        Args:
            text: Raw extracted text
        
        Returns:
            Cleaned text (uppercase, alphanumeric only)
        """
        if not text:
            return ""
        
        # Convert to uppercase
        text = text.upper()
        
        # Keep only alphanumeric
        cleaned = "".join(c for c in text if c.isalnum())
        
        return cleaned.strip()
    
    def _validate_indian_license_plate(self, plate_text: str) -> Optional[str]:
        """
        Cross-validate Indian license plate format and state codes.
        
        Args:
            plate_text: Extracted plate text to validate
        
        Returns:
            Validated plate number if VALID, None if INVALID.
            When None is returned, the plate is NOT stored in results.
        """
        try:
            plate = IndianLicensePlate(plate_number=plate_text)
            return plate.plate_number
        except ValidationError as e:
            # v3: Log detailed validation error
            logger.debug(f"Validation error for '{plate_text}': {e.errors()}")
            return None
        except Exception as e:
            logger.error(f"Unexpected validation error for '{plate_text}': {e}")
            return None