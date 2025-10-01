# ALPR Module

This folder will contain the detection (YOLO) and OCR pipeline.
Initial plan:
- Detector: YOLOv8n/s (export to ONNX for inference)
- OCR: EasyOCR or Tesseract with character whitelist.
- Post-processing: regex by plate pattern + edit-distance correction.
