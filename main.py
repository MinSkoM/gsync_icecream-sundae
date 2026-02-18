import os

# --- 1. สั่งใบ้หวย System Variable ก่อนทำอย่างอื่น ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # ยอมให้ Library ซ้ำกันได้ (สำคัญมาก!)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# --- 2. Import OpenCV เป็นตัวแรก (ต้องมาก่อน TensorFlow!) ---
import cv2
# สั่งปิด Multi-threading ของ OpenCV ทันที
cv2.setNumThreads(0) 

# --- 3. จากนั้นค่อย Import TensorFlow ---
import tensorflow as tf

# 🔥 Hackathon Hotfix: Patch Keras (ใส่ตรงนี้เหมือนเดิม)
original_layer_init = tf.keras.layers.Layer.__init__

def patched_layer_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    original_layer_init(self, *args, **kwargs)

tf.keras.layers.Layer.__init__ = patched_layer_init

# --- 4. Import Library อื่นๆ ---
import json
import numpy as np
import pandas as pd
import tempfile
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIG ---
MAX_SEQ_LENGTH = 80
NUM_LANDMARKS = 9
# ตามไฟล์ preall.py และ training script
SMOOTH_SPAN = 3 
DECISION_THRESHOLD = 0.5 # ปรับได้ตามความเหมาะสม

# --- LOAD MODELS ---
try:
    logger.info("⏳ Loading Motion Model...")
    motion_model = tf.keras.models.load_model("Model/Motion.keras")
    logger.info("✅ Motion Model Loaded.")
    
    logger.info("⏳ Loading Vision Model...")
    vision_model = tf.keras.models.load_model("Model/Vision.keras")
    logger.info("✅ Vision Model Loaded.")
except Exception as e:
    logger.error(f"❌ Critical Error Loading Models: {e}")
    motion_model = None
    vision_model = None

def preprocess_motion(json_data):
    """
    Logic เดียวกับ preall.py:
    1. Extract faceMesh (9 points) & sensors (6 values)
    2. Exponential Smoothing (span=3)
    3. Z-Score Normalization (per sequence)
    4. Padding/Truncate to 80 frames
    """
    frames = json_data.get('data', [])
    if not frames:
        raise ValueError("No frames in JSON")

    lm_seq = []
    sn_seq = []

    for frame in frames:
        # Extract Landmarks (Flat list)
        mesh = frame.get('faceMesh', [])
        if len(mesh) < NUM_LANDMARKS:
            mesh += [0.0] * (NUM_LANDMARKS - len(mesh))
        lm_seq.append(mesh[:NUM_LANDMARKS])

        # Extract Sensors
        s = frame.get('sensors', {})
        acc = s.get('accel', {'x':0, 'y':0, 'z':0})
        gyr = s.get('gyro', {'x':0, 'y':0, 'z':0})
        # เช็ค null
        ax = acc['x'] if acc['x'] is not None else 0
        ay = acc['y'] if acc['y'] is not None else 0
        az = acc['z'] if acc['z'] is not None else 0
        gx = gyr['x'] if gyr['x'] is not None else 0
        gy = gyr['y'] if gyr['y'] is not None else 0
        gz = gyr['z'] if gyr['z'] is not None else 0
        
        sn_seq.append([ax, ay, az, gx, gy, gz])

    # Convert to DF for Smoothing
    lm_df = pd.DataFrame(lm_seq)
    sn_df = pd.DataFrame(sn_seq)

    # 1. Smoothing
    lm_smooth = lm_df.ewm(span=SMOOTH_SPAN, adjust=False).mean().values
    sn_smooth = sn_df.ewm(span=SMOOTH_SPAN, adjust=False).mean().values

    # 2. Normalization (Z-Score per sample)
    lm_mean = np.mean(lm_smooth, axis=0)
    lm_std = np.std(lm_smooth, axis=0) + 1e-6
    lm_norm = (lm_smooth - lm_mean) / lm_std

    sn_mean = np.mean(sn_smooth, axis=0)
    sn_std = np.std(sn_smooth, axis=0) + 1e-6
    sn_norm = (sn_smooth - sn_mean) / sn_std

    # 3. Padding / Truncating
    curr_len = len(lm_norm)
    if curr_len < MAX_SEQ_LENGTH:
        pad_len = MAX_SEQ_LENGTH - curr_len
        lm_final = np.pad(lm_norm, ((0, pad_len), (0, 0)), mode='constant')
        sn_final = np.pad(sn_norm, ((0, pad_len), (0, 0)), mode='constant')
    else:
        lm_final = lm_norm[:MAX_SEQ_LENGTH]
        sn_final = sn_norm[:MAX_SEQ_LENGTH]

    return {
        "lm": np.expand_dims(lm_final, axis=0), 
        "sn": np.expand_dims(sn_final, axis=0)
    }

def process_video_frames(video_path):
    """
    ดึงเฟรมจากวิดีโอมา 5 เฟรม (กระจายทั่วคลิป) เพื่อส่งเข้า Vision Model
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < 1:
        return np.array([])

    # เลือก 5 เฟรม กระจายๆ กัน
    indices = np.linspace(0, total_frames - 1, 5, dtype=int)
    batch_images = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Resize 224x224
            frame = cv2.resize(frame, (224, 224))
            # Convert BGR to RGB (เพราะ OpenCV อ่านเป็น BGR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch_images.append(frame)
    
    cap.release()
    return np.array(batch_images) # (N, 224, 224, 3)

@app.post("/api/predict/liveness")
async def predict(
    video_file: UploadFile = File(...), 
    json_file: UploadFile = File(...)
):
    if not motion_model or not vision_model:
        raise HTTPException(status_code=503, detail="Models not loaded")

    # --- 1. MOTION PREDICTION ---
    motion_score = 0.0
    try:
        content = await json_file.read()
        data = json.loads(content)
        
        inputs = preprocess_motion(data)
        preds = motion_model.predict(inputs)
        motion_score = float(preds[0][0])
        logger.info(f"🧠 Motion Score: {motion_score}")
        
    except Exception as e:
        logger.error(f"Motion Error: {e}")
        # ถ้าพัง ให้คะแนนต่ำสุด
        motion_score = 0.0

    # --- 2. VISION PREDICTION ---
    vision_score = 0.0
    try:
        # Save video temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await video_file.read())
            tmp_path = tmp.name
        
        frames = process_video_frames(tmp_path)
        os.unlink(tmp_path) # ลบไฟล์ทิ้ง

        if len(frames) > 0:
            # Model Vision มี Rescaling layer อยู่ข้างในแล้ว (scale=1/127.5)
            # ดังนั้นส่งค่า 0-255 ไปได้เลย ไม่ต้องหารเอง
            v_preds = vision_model.predict(frames)
            vision_score = float(np.mean(v_preds))
            logger.info(f"👁️ Vision Score: {vision_score}")
        
    except Exception as e:
        logger.error(f"Vision Error: {e}")
        vision_score = 0.0

    # --- 3. FUSION LOGIC ---
    # ใช้ Weighted Average แบบเดียวกับ preall.py
    # (vision * 0.4) + (motion * 0.6)
    
    final_score = (vision_score * 0.4) + (motion_score * 0.6)
    is_live = final_score >= DECISION_THRESHOLD

    return {
        "final_verdict": "LIVENESS CONFIRMED" if is_live else "LIVENESS DENIED",
        "score": final_score,
        "details": {
            "motion": {
                "score": motion_score,
                "label": "REAL" if motion_score >= DECISION_THRESHOLD else "SPOOF"
            },
            "vision": {
                "score": vision_score,
                "label": "LIVE" if vision_score >= DECISION_THRESHOLD else "SPOOF"
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)