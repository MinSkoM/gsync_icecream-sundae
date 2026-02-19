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

# ==========================================
# ⚙️ CONFIGURATION (นำมาจาก preall.py)
# ==========================================
MAX_SEQ_LENGTH = 80
NUM_LANDMARKS = 9
NUM_SENSORS = 6
NUM_BG = 6

GATE_THRESHOLD = 3.0
ENABLE_SMOOTHING = True
SMOOTH_SPAN = 3
DECISION_THRESHOLD = 0.5

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


# ==========================================
# 🛠️ PRE-PROCESSING FOR FUSION (ถอดแบบ preall.py)
# ==========================================
def process_signal_data(data_seq):
    """ทำ Normalization ด้วย Z-Score (ใช้ Median และ IQR เหมือนตอนเทรน)"""
    if not isinstance(data_seq, (list, np.ndarray)) or len(data_seq) == 0:
        return np.zeros((MAX_SEQ_LENGTH, 1))
    
    df = pd.DataFrame(data_seq)
    if ENABLE_SMOOTHING: 
        df = df.ewm(span=SMOOTH_SPAN, adjust=False).mean()
    
    vals = df.astype(float).fillna(0.0).values
    median = np.median(vals, axis=0)
    q75, q25 = np.percentile(vals, [75, 25], axis=0)
    iqr = q75 - q25
    iqr[iqr == 0] = 1.0  # ป้องกันหาร 0
    
    normalized = (vals - median) / iqr
    return np.clip(normalized, -4.0, 4.0)

def preprocess_motion(json_data):
    frames = json_data.get('data', [])
    if not frames: 
        raise ValueError("No frames in JSON")
    
    # 1. เช็ค Gate Threshold สำหรับ Background
    all_vars = [f.get('bg_variance', 0) for f in frames]
    is_gate_open = np.mean(all_vars) >= GATE_THRESHOLD if all_vars else False
    
    lm_seq, sn_seq, bg_seq = [], [], []
    prev_lm = None
    
    for f in frames:
        meta = f.get('meta', {}) or {}
        mult = -1 if meta.get('camera_facing') == 'environment' else 1
        
        # --- จัดการ Landmarks (คำนวณ Delta หรือความแตกต่างระหว่างเฟรม) ---
        raw_lm = f.get('faceMesh')
        if not raw_lm: 
            continue
            
        lm_array = np.array(raw_lm)
        if len(lm_array) == NUM_LANDMARKS:
            # ถ้าส่งมาแค่ 9 ค่า (3 จุด) พอดี
            curr_face_features = lm_array
        else:
            # ถ้าส่ง FaceMesh แบบเต็มมา
            lm_matrix = lm_array.reshape(-1, 3)
            nose = lm_matrix[8] if len(lm_matrix) > 8 else [0,0,0]
            left_eye = lm_matrix[12] if len(lm_matrix) > 12 else [0,0,0]
            right_eye = lm_matrix[16] if len(lm_matrix) > 16 else [0,0,0]
            curr_face_features = np.concatenate([nose, left_eye, right_eye])
        
        # คำนวณ Delta
        if prev_lm is None:
            d_lm = np.zeros_like(curr_face_features)
        else:
            d_lm = curr_face_features - prev_lm
        
        lm_seq.append(d_lm)
        prev_lm = curr_face_features

        # --- จัดการ Sensors ---
        s = f.get('sensors', {}) or {}
        a = s.get('accel') or {}
        g = s.get('gyro') or {}
        
        def sf(x): return float(x) if x is not None else 0.0
        
        sn_seq.append([sf(a.get('x')), sf(a.get('y')), sf(a.get('z'))*mult,
                       sf(g.get('x')), sf(g.get('y')), sf(g.get('z'))])

        # --- จัดการ Background ---
        if is_gate_open:
            m = f.get('motion_analysis', {}) or {}
            bg_seq.append([sf(m.get('face_dx')), sf(m.get('face_dy')),
                           sf(m.get('bg_dx')), sf(m.get('bg_dy')),
                           sf(m.get('relative_magnitude')), sf(f.get('bg_variance'))])
        else: 
            bg_seq.append([0.0] * NUM_BG)
    
    if len(lm_seq) < 5: 
        raise ValueError("Too few frames for analysis")
        
    # 2. ทำ Normalization (IQR/Median) แบบเดียวกับ preall.py
    X_lm = process_signal_data(np.array(lm_seq))
    X_sn = process_signal_data(np.array(sn_seq))
    X_bg = process_signal_data(np.array(bg_seq))

    # 3. Crop/Pad ให้ได้ความยาว 80 เฟรมเป๊ะๆ
    def crop_pad(arr):
        if len(arr) > MAX_SEQ_LENGTH:
            start = (len(arr) - MAX_SEQ_LENGTH) // 2
            return arr[start : start + MAX_SEQ_LENGTH]
        elif len(arr) < MAX_SEQ_LENGTH:
            return np.vstack((arr, np.zeros((MAX_SEQ_LENGTH - len(arr), arr.shape[1]))))
        return arr
        
    # คืนค่าเป็น Dictionary ให้ Model .predict()
    return {
        "lm": np.expand_dims(crop_pad(X_lm), axis=0),
        "sn": np.expand_dims(crop_pad(X_sn), axis=0),
        "bg": np.expand_dims(crop_pad(X_bg), axis=0)
    }

# ==========================================
# 📷 VISION MODEL PROCESSING
# ==========================================
def process_video_frames(video_path):
    """
    ดึงเฟรมจากวิดีโอมา 5 เฟรม (กระจายทั่วคลิป) เพื่อส่งเข้า Vision Model
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < 1:
        return np.array([])

    indices = np.linspace(0, total_frames - 1, 5, dtype=int)
    batch_images = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch_images.append(frame)
    
    cap.release()
    return np.array(batch_images) # (N, 224, 224, 3)

# ==========================================
# 🌐 API ENDPOINT
# ==========================================
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
        motion_score = 0.0

    # --- 2. VISION PREDICTION ---
    vision_score = 0.0
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await video_file.read())
            tmp_path = tmp.name
        
        frames = process_video_frames(tmp_path)
        os.unlink(tmp_path) 

        if len(frames) > 0:
            v_preds = vision_model.predict(frames)
            vision_score = float(np.mean(v_preds))
            logger.info(f"👁️ Vision Score: {vision_score}")
        
    except Exception as e:
        logger.error(f"Vision Error: {e}")
        vision_score = 0.0

    # --- 3. FUSION LOGIC ---
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