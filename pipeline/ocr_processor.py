"""OCR Processing Module - License Plate Text Extraction"""

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR
from .config import Config


class OCRProcessor:
    """Handles license plate text extraction using RapidOCR"""
    
    def __init__(self):
        """Initialize RapidOCR engine"""
        self.ocr_engine = RapidOCR()
    
    def extract_plate_text(self, image, plate_boxes):
        """
        Extract text from detected license plates.
        
        Args:
            image: Original image (BGR, not preprocessed)
            plate_boxes: List of plate bounding boxes [{"box": [x1,y1,x2,y2], "confidence": 0.88}, ...]
        
        Returns:
            List of extracted texts with confidence scores
            [{"box": [x1,y1,x2,y2], "text": "DL1AB1234", "confidence": 0.92}, ...]
        """
        results = []
        
        for plate_box in plate_boxes:
            try:
                box = plate_box["box"]
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                
                # Crop plate region from image
                plate_crop = image[y1:y2, x1:x2]
                
                if plate_crop.size == 0:
                    continue
                
                # Add white border padding (prevents edge characters from being cut off)
                border_px = Config.OCR_PARAMS["white_border_px"]
                plate_with_border = cv2.copyMakeBorder(
                    plate_crop,
                    border_px, border_px, border_px, border_px,
                    cv2.BORDER_CONSTANT,
                    value=(255, 255, 255)
                )
                
                # Run RapidOCR on padded crop (recognition-only; plate region already cropped)
                ocr_output, _ = self.ocr_engine(
                    plate_with_border,
                    use_det=False,
                    use_cls=False,
                    use_rec=True,
                )

                if ocr_output:
                    extracted_texts = []

                    for detection in ocr_output:
                        if len(detection) >= 2:
                            if isinstance(detection[0], str):
                                text = detection[0]
                                confidence = float(detection[1])
                            else:
                                text = detection[1]
                                confidence = float(detection[2]) if len(detection) > 2 else 0.0
                            extracted_texts.append((text, confidence))
                    
                    if extracted_texts:
                        # Combine all detected texts (usually one license plate text)
                        combined_text = "".join([text for text, _ in extracted_texts])
                        avg_confidence = np.mean([conf for _, conf in extracted_texts])
                        
                        # Clean text: uppercase, alphanumeric + hyphens only
                        cleaned_text = self._clean_plate_text(combined_text)
                        
                        if cleaned_text:  # Only add if text is not empty after cleaning
                            results.append({
                                "box": box,
                                "text": cleaned_text,
                                "confidence": avg_confidence,
                                "raw_text": combined_text  # Store original for debugging
                            })
            
            except Exception as e:
                # Lenient error handling: skip failed plates, continue
                continue
        
        return results
    
    def _clean_plate_text(self, text):
        """
        Clean extracted license plate text.
        
        Args:
            text: Raw extracted text
        
        Returns:
            Cleaned text (uppercase, alphanumeric + hyphens only)
        """
        if not text:
            return ""
        
        # Convert to uppercase
        text = text.upper()
        
        # Keep only alphanumeric and hyphens
        cleaned = "".join(c for c in text if c.isalnum() or c == "-")
        
        return cleaned.strip()
