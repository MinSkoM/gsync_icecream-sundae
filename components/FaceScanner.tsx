import React, { useRef, useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';
import { SensorData } from '../types'; 

interface FaceScannerProps {
  onScanComplete: (data: any, blob: Blob) => void;
  onCancel: () => void;
}

const MAX_FRAMES = 80;
const SELECTED_LANDMARKS = [1, 33, 263]; 

const FaceScanner: React.FC<FaceScannerProps> = ({ onScanComplete, onCancel }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const collectedFramesRef = useRef<any[]>([]); 
  const isRecordingRef = useRef(false);
  const animationFrameIdRef = useRef<number | null>(null);
  const detectorRef = useRef<faceLandmarksDetection.FaceLandmarksDetector | null>(null);
  
  const [status, setStatus] = useState('Initializing AI...');
  const [progress, setProgress] = useState(0);

  const sensorRef = useRef<SensorData>({
    accel: { x: 0, y: 0, z: 0 },
    gyro: { x: 0, y: 0, z: 0 },
  });

  // --- 1. SENSOR SETUP (เหมือนเดิมแต่อยู่ใน useEffect เดียวกันเพื่อความคลีน) ---
  useEffect(() => {
    const handleMotion = (event: DeviceMotionEvent) => {
      if (event.accelerationIncludingGravity) {
        sensorRef.current.accel = {
          x: event.accelerationIncludingGravity.x || 0,
          y: event.accelerationIncludingGravity.y || 0,
          z: event.accelerationIncludingGravity.z || 0,
        };
      }
      if (event.rotationRate) {
        sensorRef.current.gyro = {
          x: event.rotationRate.alpha || 0,
          y: event.rotationRate.beta || 0,
          z: event.rotationRate.gamma || 0,
        };
      }
    };

    if (typeof (DeviceMotionEvent as any).requestPermission === 'function') {
      (DeviceMotionEvent as any).requestPermission().then((state: string) => {
        if (state === 'granted') window.addEventListener('devicemotion', handleMotion);
      });
    } else {
      window.addEventListener('devicemotion', handleMotion);
    }

    return () => window.removeEventListener('devicemotion', handleMotion);
  }, []);

  // --- 2. FAST INITIALIZATION LOGIC ---
  useEffect(() => {
    let active = true;

    const setupApp = async () => {
      try {
        // 🔥 STEP A: ตั้งค่า Backend และโหลดกล้องไปพร้อมกัน (Parallel)
        const [_, stream] = await Promise.all([
          tf.setBackend('webgl').then(() => tf.ready()),
          navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'user', width: 640, height: 480 }, 
            audio: false 
          })
        ]);

        if (!active) return;

        // แสดงภาพจากกล้องทันทีแม้ AI จะยังโหลดไม่เสร็จ (ลดความรู้สึกว่าเครื่องค้าง)
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }

        // 🔥 STEP B: โหลด Detector โดยใช้ Mediapipe Runtime (เร็วกว่า TFJS runtime มาก)
        const detector = await faceLandmarksDetection.createDetector(
          faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh,
          {
            runtime: 'mediapipe', 
            solutionPath: `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh`,
            refineLandmarks: true,
            maxFaces: 1,
          }
        );

        if (!active) return;
        detectorRef.current = detector;

        // 🔥 STEP C: Warm-up Model (รันรอบแรกทิ้งเพื่อให้รอบจริงไม่กระตุก)
        if (videoRef.current) {
            await detector.estimateFaces(videoRef.current);
        }

        setStatus('Scanning...');
        startRecording(stream);
        scanFrame();

      } catch (err) {
        console.error("Initialization failed:", err);
        setStatus('Error: Camera/AI not ready');
      }
    };

    setupApp();
    return () => { active = false; };
  }, []);

  const startRecording = (stream: MediaStream) => {
    isRecordingRef.current = true;
    recordedChunksRef.current = [];
    collectedFramesRef.current = []; 

    const options = { mimeType: 'video/mp4' };
    try {
        mediaRecorderRef.current = new MediaRecorder(stream, MediaRecorder.isTypeSupported('video/mp4') ? options : undefined);
    } catch (e) {
        mediaRecorderRef.current = new MediaRecorder(stream);
    }

    mediaRecorderRef.current.ondataavailable = (e) => e.data.size > 0 && recordedChunksRef.current.push(e.data);
    mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(recordedChunksRef.current, { type: 'video/mp4' });
        const finalData = {
            type: "LIVENESS_SCAN",
            scenario: "Production",
            data: collectedFramesRef.current,
            meta: { userAgent: navigator.userAgent }
        };
        onScanComplete(finalData, blob);
    };
    mediaRecorderRef.current.start();
  };

  const scanFrame = async () => {
    if (!videoRef.current || !canvasRef.current || !isRecordingRef.current || !detectorRef.current) {
      // ถ้าตัวแปรไม่ครบ ให้ Log ดูว่าขาดอะไร
      console.log("Waiting for components...", { 
        video: !!videoRef.current, 
        isRecording: isRecordingRef.current, 
        detector: !!detectorRef.current 
      });
      return;
    }

    const faces = await detectorRef.current.estimateFaces(videoRef.current, { flipHorizontal: false });
    
    // --- เพิ่ม Log ตรงนี้ ---
    if (faces.length === 0) {
      console.warn("No face detected! Please show your face to the camera.");
      // อัปเดต Status บอกผู้ใช้
      setStatus('Scanning... (No face detected)');
    } else {
      if (status !== 'Scanning...') setStatus('Scanning...');
    }
    // ----------------------

    const ctx = canvasRef.current.getContext('2d');
    if (ctx && faces.length > 0) {
        // ... โค้ดส่วนประมวลผล Landmarks เดิม ...
        
        // ลอง Log ดูว่าเฟรมที่เท่าไหร่แล้ว
        if (collectedFramesRef.current.length % 10 === 0) {
          console.log(`Collected ${collectedFramesRef.current.length} frames`);
        }

        // ... ส่วนที่เหลือของฟังก์ชัน ...
    }

    animationFrameIdRef.current = requestAnimationFrame(scanFrame);
  };

  // Cleanup เหมือนเดิม...
  useEffect(() => {
    return () => {
      if (animationFrameIdRef.current) cancelAnimationFrame(animationFrameIdRef.current);
      if (videoRef.current && videoRef.current.srcObject) {
        (videoRef.current.srcObject as MediaStream).getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  return (
    <div className="flex flex-col items-center gap-4 w-full">
      <div className="relative w-full max-w-sm aspect-[3/4] rounded-2xl overflow-hidden shadow-2xl border-4 border-indigo-500/30 bg-black">
        <video ref={videoRef} autoPlay playsInline muted className="absolute inset-0 w-full h-full object-cover transform -scale-x-100" />
        <canvas ref={canvasRef} width="640" height="480" className="absolute inset-0 w-full h-full pointer-events-none" />
        
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black via-black/80 to-transparent p-6 text-center">
          <p className="font-bold text-xl text-white mb-3 tracking-wide">{status}</p>
          <div className="w-full bg-gray-800/50 rounded-full h-2.5 backdrop-blur-md overflow-hidden">
            <div 
                className="bg-indigo-500 h-full transition-all duration-150 ease-out" 
                style={{ width: `${progress}%` }} 
            />
          </div>
          <p className="text-xs text-indigo-300 mt-2 font-mono">{Math.round(progress)}% COMPLETE</p>
        </div>
      </div>
      <button onClick={onCancel} className="px-6 py-2 text-gray-400 hover:text-white transition-colors">Cancel</button>
    </div>
  );
};

export default FaceScanner;