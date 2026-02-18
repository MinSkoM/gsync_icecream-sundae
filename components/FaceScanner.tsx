import React, { useRef, useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';
import { LivenessData, SensorData } from '../types'; 
// หมายเหตุ: ตรวจสอบ types.ts ของคุณให้รองรับ structure นี้ หรือใช้ any ชั่วคราวได้ครับ

interface FaceScannerProps {
  onScanComplete: (data: any, blob: Blob) => void; // ใช้ any เพื่อความยืดหยุ่นกับ JSON structure
  onCancel: () => void;
}

const MAX_FRAMES = 80;
// จุด Landmark สำคัญ 3 จุด (จมูก, ตาซ้าย, ตาขวา) -> รวม 9 ค่า (x,y,z)
// ต้องเรียงลำดับตามนี้เพื่อให้ตรงกับโมเดล Motion
const SELECTED_LANDMARKS = [1, 33, 263]; 

const FaceScanner: React.FC<FaceScannerProps> = ({ onScanComplete, onCancel }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  
  // Buffer สำหรับเก็บข้อมูล JSON
  const collectedFramesRef = useRef<any[]>([]); 
  const isRecordingRef = useRef(false);
  const animationFrameIdRef = useRef<number | null>(null);
  
  const [status, setStatus] = useState('Initializing AI...');
  const [progress, setProgress] = useState(0);

  // Sensor Store (เก็บค่าล่าสุดแบบ Real-time)
  const sensorRef = useRef<SensorData>({
    accel: { x: 0, y: 0, z: 0 },
    gyro: { x: 0, y: 0, z: 0 },
  });

  // 1. Setup Sensors (Accelerometer & Gyroscope)
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

    // Permission Request for iOS 13+
    const setupSensors = async () => {
        if (typeof (DeviceMotionEvent as any).requestPermission === 'function') {
            try {
                const permissionState = await (DeviceMotionEvent as any).requestPermission();
                if (permissionState === 'granted') {
                    window.addEventListener('devicemotion', handleMotion);
                }
            } catch (error) {
                console.error("Sensor permission error:", error);
            }
        } else {
            window.addEventListener('devicemotion', handleMotion);
        }
    };

    setupSensors();
    return () => window.removeEventListener('devicemotion', handleMotion);
  }, []);

  // 2. Setup Camera & AI Model
  useEffect(() => {
    const runFaceMesh = async () => {
      await tf.ready();
      const model = await faceLandmarksDetection.createDetector(
        faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh,
        {
          runtime: 'tfjs',
          refineLandmarks: true,
          maxFaces: 1,
        }
      );
      setStatus('Ready. Hold still.');
      startVideo(model);
    };
    runFaceMesh();
  }, []);

  const startVideo = (model: faceLandmarksDetection.FaceLandmarksDetector) => {
    navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'user', width: 640, height: 480 }, 
        audio: false 
    }).then((stream) => {
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadeddata = () => {
            startRecording(stream);
            scanFrame(model);
        };
      }
    });
  };

  const startRecording = (stream: MediaStream) => {
    isRecordingRef.current = true;
    recordedChunksRef.current = [];
    collectedFramesRef.current = []; 

    // Setup MediaRecorder
    const options = { mimeType: 'video/mp4' };
    try {
        mediaRecorderRef.current = new MediaRecorder(stream, MediaRecorder.isTypeSupported('video/mp4') ? options : undefined);
    } catch (e) {
        mediaRecorderRef.current = new MediaRecorder(stream);
    }

    mediaRecorderRef.current.ondataavailable = (event) => {
      if (event.data.size > 0) recordedChunksRef.current.push(event.data);
    };

    mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(recordedChunksRef.current, { type: 'video/mp4' });
        
        // Construct Final JSON (Structure เดียวกับ data.json)
        const finalData = {
            type: "UNKNOWN",
            scenario: "Production",
            motion: "unknown",
            data: collectedFramesRef.current, // Array of frames
            meta: {
                userAgent: navigator.userAgent
            }
        };
        onScanComplete(finalData, blob);
    };

    mediaRecorderRef.current.start();
    setStatus('Scanning...');
  };

  const scanFrame = async (model: faceLandmarksDetection.FaceLandmarksDetector) => {
    if (!videoRef.current || !canvasRef.current || !isRecordingRef.current) return;

    // Detect Face
    const faces = await model.estimateFaces(videoRef.current, { flipHorizontal: false });
    
    // Draw Feedback
    const ctx = canvasRef.current.getContext('2d');
    if (ctx) {
        ctx.clearRect(0, 0, 640, 480);
        ctx.save();
        ctx.scale(-1, 1);
        ctx.translate(-640, 0);
        
        if (faces.length > 0) {
            const face = faces[0];
            const timestamp = Date.now();
            const width = videoRef.current.videoWidth;
            const height = videoRef.current.videoHeight;

            // --- Extract & Normalize Landmarks ---
            // เราต้องการ 9 ค่า (x,y,z ของ 3 จุด) แบบ Flat Array
            const flatFaceMesh: number[] = [];
            
            SELECTED_LANDMARKS.forEach(index => {
                const p = face.keypoints[index];
                // Normalize ให้เป็น 0.0 - 1.0 ตาม data.json
                flatFaceMesh.push(p.x / width);  
                flatFaceMesh.push(p.y / height);
                // Z ใน TFJS เป็น pixel scale ประมาณการ, หาร width เพื่อ normalize คร่าวๆ
                flatFaceMesh.push((p as any).z ? (p as any).z / width : 0); 
                
                // Draw debug points
                ctx.fillStyle = '#00FF00';
                ctx.beginPath();
                ctx.arc(p.x, p.y, 3, 0, 2 * Math.PI);
                ctx.fill();
            });

            // Push Data Frame
            collectedFramesRef.current.push({
                timestamp: timestamp,
                faceMesh: flatFaceMesh, // [x1,y1,z1, x2,y2,z2, x3,y3,z3]
                sensors: {
                    accel: { ...sensorRef.current.accel },
                    gyro: { ...sensorRef.current.gyro }
                }
            });

            // Update Progress
            const progressVal = (collectedFramesRef.current.length / MAX_FRAMES) * 100;
            setProgress(progressVal);

            if (collectedFramesRef.current.length >= MAX_FRAMES) {
                isRecordingRef.current = false;
                mediaRecorderRef.current?.stop();
                return; 
            }
        }
        ctx.restore();
    }

    animationFrameIdRef.current = requestAnimationFrame(() => scanFrame(model));
  };

  // Cleanup
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
      <div className="relative w-full max-w-sm aspect-[3/4] rounded-lg overflow-hidden shadow-2xl border-2 border-indigo-500 bg-black">
        <video ref={videoRef} autoPlay playsInline muted className="absolute inset-0 w-full h-full object-cover transform -scale-x-100" />
        <canvas ref={canvasRef} width="640" height="480" className="absolute inset-0 w-full h-full" />
        
        {/* UI Overlay */}
        <div className="absolute bottom-0 left-0 right-0 bg-black/70 p-4 text-center backdrop-blur-sm">
          <p className="font-semibold text-lg text-white mb-2">{status}</p>
          <div className="w-full bg-gray-700 rounded-full h-3 overflow-hidden">
            <div 
                className="bg-gradient-to-r from-indigo-500 to-purple-500 h-full transition-all duration-100 ease-linear" 
                style={{ width: `${progress}%` }} 
            />
          </div>
          <p className="text-xs text-gray-400 mt-1">{Math.round(progress)}%</p>
        </div>
      </div>
      <button onClick={onCancel} className="text-gray-400 hover:text-white mt-2">Cancel</button>
    </div>
  );
};

export default FaceScanner;