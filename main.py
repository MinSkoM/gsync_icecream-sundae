
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import cv2
import uuid
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import Dict, Any, List

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- App Initialization ---
app = FastAPI(
    title="Face Liveness Detection API",
    description="Processes video and motion data to determine face liveness.",
    version="1.0.0"
)

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Motion Model (LivenessPredictor Class) ---
class LivenessPredictor:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Motion model not found at {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        self.MAX_SEQ_LENGTH = 80
        self.NUM_LANDMARKS = 28 * 3
        self.NUM_SENSORS = 6
        self.NUM_BG = 6
        self.ENABLE_SMOOTHING = True
        self.SMOOTH_SPAN = 3

    def _safe_float(self, val: Any) -> float:
        try:
            return float(val) if val is not None else 0.0
        except (ValueError, TypeError):
            return 0.0

    def _perform_temporal_alignment(self, seq: List[List[float]], feature_dim: int) -> np.ndarray:
        seq_len = len(seq)
        np_seq = np.array(seq, dtype=np.float32)
        
        if seq_len > self.MAX_SEQ_LENGTH:
            start = (seq_len - self.MAX_SEQ_LENGTH) // 2
            return np_seq[start:start + self.MAX_SEQ_LENGTH]
        elif seq_len < self.MAX_SEQ_LENGTH:
            padding = np.zeros((self.MAX_SEQ_LENGTH - seq_len, feature_dim), dtype=np.float32)
            return np.vstack([np_seq, padding])
        return np_seq

    def _process_signal_data(self, data_seq: np.ndarray) -> np.ndarray:
        if data_seq.size == 0:
            return np.zeros((self.MAX_SEQ_LENGTH, data_seq.shape[1] if data_seq.ndim > 1 else 1))
        
        df = pd.DataFrame(data_seq)
        if self.ENABLE_SMOOTHING:
            df = df.ewm(span=self.SMOOTH_SPAN, adjust=False).mean()
        
        vals = df.astype(float).fillna(0.0).values
        median = np.median(vals, axis=0)
        q75, q25 = np.percentile(vals, [75, 25], axis=0)
        iqr = q75 - q25
        iqr[iqr == 0] = 1.0
        normalized = (vals - median) / iqr
        return np.clip(normalized, -4.0, 4.0)

    def _extract_features(self, json_data: Dict[str, Any]) -> (np.ndarray, np.ndarray, np.ndarray):
        frames = json_data.get('data', [])
        if not frames:
            raise ValueError("No frames found in JSON data")

        lm_seq, sn_seq, bg_seq = [], [], []
        for frame in frames:
            lm_seq.append(frame.get('faceMesh', [])[:self.NUM_LANDMARKS])
            
            sensors = frame.get('sensors', {})
            accel = sensors.get('accel', {})
            gyro = sensors.get('gyro', {})
            sn_seq.append([
                self._safe_float(accel.get('x')), self._safe_float(accel.get('y')), self._safe_float(accel.get('z')),
                self._safe_float(gyro.get('x')), self._safe_float(gyro.get('y')), self._safe_float(gyro.get('z'))
            ])
            
            motion = frame.get('motion_analysis', {})
            bg_seq.append([
                self._safe_float(motion.get('face_dx')), self._safe_float(motion.get('face_dy')),
                self._safe_float(motion.get('bg_dx')), self._safe_float(motion.get('bg_dy')),
                self._safe_float(motion.get('relative_magnitude')),
                self._safe_float(frame.get('bg_variance'))
            ])
        
        aligned_lm = self._perform_temporal_alignment(lm_seq, self.NUM_LANDMARKS)
        aligned_sn = self._perform_temporal_alignment(sn_seq, self.NUM_SENSORS)
        aligned_bg = self._perform_temporal_alignment(bg_seq, self.NUM_BG)
        
        processed_sn = self._process_signal_data(aligned_sn)
        processed_bg = self._process_signal_data(aligned_bg)

        # Normalize landmarks
        lm_df = pd.DataFrame(aligned_lm).astype(float).fillna(0.0)
        processed_lm = ((lm_df - lm_df.mean()) / (lm_df.std().replace(0, 1))).fillna(0.0).values
        
        return processed_lm, processed_sn, processed_bg

    def predict(self, json_data: Dict[str, Any], threshold: float = 0.5) -> Dict[str, Any]:
        try:
            lm, sn, bg = self._extract_features(json_data)
        except ValueError as e:
            return {"status": "error", "message": str(e)}

        input_lm = np.expand_dims(lm, axis=0)
        input_sn = np.expand_dims(sn, axis=0)
        input_bg = np.expand_dims(bg, axis=0)
        
        score = self.model.predict([input_lm, input_sn, input_bg], verbose=0)[0][0]
        label = "REAL" if score > threshold else "SPOOF"
        
        return {"status": "success", "label": label, "score": float(score), "confidence": f"{score:.2%}"}

# --- Vision Model (predict_video function) ---
IMG_SIZE = (224, 224)
try:
    vision_model = tf.keras.models.load_model("vision.keras")
except IOError:
    vision_model = None
    logger.error("Vision model 'vision.keras' not found. Vision predictions will be disabled.")

def predict_video(video_path: str, threshold: float = 0.5) -> Dict[str, Any]:
    if vision_model is None:
        return {"status": "error", "message": "Vision model not loaded"}
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"status": "error", "message": "Could not open video file."}

    frame_scores = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = tf.image.resize(frame_rgb, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        img_array = tf.expand_dims(img, axis=0)
        
        score = vision_model.predict(img_array, verbose=0)
        frame_scores.append(score[0][0])
    
    cap.release()
    
    if not frame_scores:
        return {"status": "error", "message": "No frames processed from video."}
    
    final_score = np.mean(frame_scores)
    label = "LIVE" if final_score >= threshold else "SPOOF"
    
    return {"status": "success", "label": label, "score": float(final_score)}

# --- Load Models ---
try:
    motion_predictor = LivenessPredictor(model_path='motion.keras')
except FileNotFoundError as e:
    motion_predictor = None
    logger.error(f"Error loading motion model: {e}. Motion predictions will be disabled.")


# --- API Endpoint ---
@app.post("/api/predict/liveness")
async def predict_liveness_endpoint(video_file: UploadFile = File(...), json_file: UploadFile = File(...)):
    if motion_predictor is None or vision_model is None:
        raise HTTPException(status_code=503, detail="One or more AI models are not available.")
        
    # Process JSON file
    try:
        json_content = await json_file.read()
        motion_data = json.loads(json_content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file.")
    finally:
        await json_file.close()

    motion_result = motion_predictor.predict(motion_data)
    
    # Process video file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
        content = await video_file.read()
        tmp_video.write(content)
        tmp_video_path = tmp_video.name
    
    await video_file.close()
    
    vision_result = predict_video(tmp_video_path)
    
    # Clean up temporary video file
    os.unlink(tmp_video_path)
    
    # Determine final verdict
    is_motion_real = motion_result.get("label") == "REAL"
    is_vision_live = vision_result.get("label") == "LIVE"
    
    if is_motion_real and is_vision_live:
        final_verdict = "LIVENESS CONFIRMED"
    else:
        final_verdict = "LIVENESS DENIED"
        
    return {
        "motion_model": motion_result,
        "vision_model": vision_result,
        "final_verdict": final_verdict
    }

@app.get("/")
def read_root():
    return {"message": "Liveness Detection API is running."}
