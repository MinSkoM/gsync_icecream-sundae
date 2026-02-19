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
  // --- 2. FAST INITIALIZATION LOGIC ---
  useEffect(() => {
    let active = true;

    const setupApp = async () => {
      try {
        const [_, stream] = await Promise.all([
          tf.setBackend('webgl').then(() => tf.ready()),
          navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'user', width: 640, height: 480 }, 
            audio: false 
          })
        ]);

        if (!active) return;

        // 🔥 สำคัญ: ต้องรอให้วิดีโอโหลดภาพเฟรมแรกเสร็จก่อน ไม่งั้น AI จะพัง (Error 0x0)
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await new Promise<void>((resolve) => {
            if (videoRef.current) {
              videoRef.current.onloadeddata = () => resolve();
            } else {
              resolve();
            }
          });
        }

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

        // 🔥 ป้องกันซ้อนอีกชั้น: Warm-up เฉพาะตอนที่ขนาดวิดีโอ > 0
        if (videoRef.current && videoRef.current.videoWidth > 0) {
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
    // เพิ่มการเช็ค videoWidth === 0 และพร้อมใช้งานหรือไม่
    if (!videoRef.current || !canvasRef.current || !isRecordingRef.current || !detectorRef.current || videoRef.current.videoWidth === 0) {
        animationFrameIdRef.current = requestAnimationFrame(scanFrame);
        return;
    }

    try {
        const faces = await detectorRef.current.estimateFaces(videoRef.current, { flipHorizontal: false });
        const ctx = canvasRef.current.getContext('2d');
        
        if (ctx && faces.length > 0) {
            // ... (โค้ดเดิมทั้งหมดใน block นี้ปล่อยไว้เหมือนเดิมครับ) ...
            const face = faces[0];
            const width = videoRef.current.videoWidth;
            const height = videoRef.current.videoHeight;
            
            ctx.clearRect(0, 0, 640, 480);
            ctx.save();
            ctx.scale(-1, 1);
            ctx.translate(-640, 0);

            const flatFaceMesh: number[] = [];
            SELECTED_LANDMARKS.forEach(index => {
                const p = face.keypoints[index];
                flatFaceMesh.push(p.x / width, p.y / height, (p as any).z ? (p as any).z / width : 0);
                
                ctx.fillStyle = '#6366f1';
                ctx.beginPath();
                ctx.arc(p.x, p.y, 4, 0, 2 * Math.PI);
                ctx.fill();
            });

            collectedFramesRef.current.push({
                timestamp: Date.now(),
                faceMesh: flatFaceMesh,
                sensors: { ...sensorRef.current }
            });

            const progressVal = (collectedFramesRef.current.length / MAX_FRAMES) * 100;
            setProgress(progressVal);

            if (collectedFramesRef.current.length >= MAX_FRAMES) {
                isRecordingRef.current = false;
                mediaRecorderRef.current?.stop();
                return; 
            }
            ctx.restore();
        }
    } catch (e) {
        console.warn("Frame skipped due to error: ", e);
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