
import React, { useRef, useEffect, useState, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';
import { LivenessData, FrameData, SensorData } from '../types';

interface FaceScannerProps {
  onScanComplete: (data: LivenessData, blob: Blob) => void;
  onCancel: () => void;
}

const MAX_FRAMES = 80;
const FACE_LANDMARKS = 28; // As required by the Python model logic

const FaceScanner: React.FC<FaceScannerProps> = ({ onScanComplete, onCancel }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const frameCountRef = useRef(0);
  const capturedFramesRef = useRef<FrameData[]>([]);
  const sensorDataRef = useRef<SensorData>({
    accel: { x: null, y: null, z: null },
    gyro: { x: null, y: null, z: null },
  });
  // FIX: Initialize useRef with null to avoid potential errors with older @types/react versions.
  const animationFrameIdRef = useRef<number | null>(null);
  const [status, setStatus] = useState('Initializing...');
  
  const prevFaceCentroidRef = useRef<{x: number, y: number, z: number} | null>(null);
  const prevTimestampRef = useRef<number | null>(null);

  const handleDeviceMotion = useCallback((event: DeviceMotionEvent) => {
    sensorDataRef.current = {
      accel: event.accelerationIncludingGravity ? {
        x: event.accelerationIncludingGravity.x,
        y: event.accelerationIncludingGravity.y,
        z: event.accelerationIncludingGravity.z,
      } : { x: 0, y: 0, z: 0 },
      gyro: event.rotationRate ? {
        x: event.rotationRate.alpha,
        y: event.rotationRate.beta,
        z: event.rotationRate.gamma,
      } : { x: 0, y: 0, z: 0 },
    };
  }, []);

  const cleanup = useCallback(() => {
    if (animationFrameIdRef.current) {
      cancelAnimationFrame(animationFrameIdRef.current);
    }
    window.removeEventListener('devicemotion', handleDeviceMotion);
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
        mediaRecorderRef.current.stop();
    }
  }, [handleDeviceMotion]);

  useEffect(() => {
    const setup = async () => {
      try {
        await tf.setBackend('webgl');
        
        // Permission is now handled by the parent component before this mounts.
        // We just attach the listener.
        window.addEventListener('devicemotion', handleDeviceMotion);
        
        setStatus('Setting up camera...');
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480 },
          audio: false,
        });

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }

        setStatus('Loading face model...');
        const model = await faceLandmarksDetection.createDetector(
          faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh,
          {
            runtime: 'tfjs', // Using tfjs runtime to avoid MediaPipe loading issues.
            refineLandmarks: true, 
          }
        );
        
        setStatus('Waiting to start scan...');
        startScan(model, stream);

      } catch (error) {
        console.error("Setup failed:", error);
        setStatus(`Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
        cleanup();
      }
    };

    setup();
    return cleanup;
  }, [cleanup, handleDeviceMotion]);

  const startScan = (model: faceLandmarksDetection.FaceLandmarksDetector, stream: MediaStream) => {
    mediaRecorderRef.current = new MediaRecorder(stream, { mimeType: 'video/webm' });
    mediaRecorderRef.current.ondataavailable = (event) => {
      if (event.data.size > 0) {
        recordedChunksRef.current.push(event.data);
      }
    };
    mediaRecorderRef.current.onstop = () => {
        const videoBlob = new Blob(recordedChunksRef.current, { type: 'video/webm' });
        const finalData: LivenessData = {
            type: "REAL", // Assuming real for capture, backend will verify
            scenario: "Normal",
            motion: "orbital_LR", // Example value
            data: capturedFramesRef.current,
        };
        onScanComplete(finalData, videoBlob);
    };
    mediaRecorderRef.current.start();
    frameCountRef.current = 0;
    capturedFramesRef.current = [];
    scanFrame(model);
  };
  
  const scanFrame = async (model: faceLandmarksDetection.FaceLandmarksDetector) => {
    if (frameCountRef.current >= MAX_FRAMES) {
      setStatus('Scan complete. Finalizing...');
      cleanup();
      return;
    }

    if (videoRef.current && videoRef.current.readyState === 4) {
      // FIX: The estimateFaces method now takes the video element directly.
      const faces = await model.estimateFaces(videoRef.current);
      
      if (faces.length > 0) {
        const keypoints = faces[0].keypoints.slice(0, FACE_LANDMARKS);
        const faceMesh = keypoints.flatMap(p => [p.x / 640, p.y / 480, p.z ?? 0]);

        const currentTimestamp = performance.now();
        const dt = prevTimestampRef.current ? (currentTimestamp - prevTimestampRef.current) / 1000 : 1 / 30;

        // FIX: Ensure 'z' property is handled correctly, resolving potential type errors by providing a fallback value.
        // FIX: Explicitly type the accumulator in the reduce function to ensure `currentCentroid` has the correct type with a non-optional `z` property, resolving the assignment error.
        const currentCentroid = keypoints.reduce((acc: {x: number, y: number, z: number}, p) => ({x: acc.x + p.x, y: acc.y + p.y, z: acc.z + (p.z ?? 0)}), {x:0, y:0, z:0});
        currentCentroid.x /= keypoints.length;
        currentCentroid.y /= keypoints.length;
        currentCentroid.z /= keypoints.length;

        let face_dx = 0, face_dy = 0;
        if(prevFaceCentroidRef.current) {
            face_dx = (currentCentroid.x - prevFaceCentroidRef.current.x) / dt;
            face_dy = (currentCentroid.y - prevFaceCentroidRef.current.y) / dt;
        }
        
        // Lightweight proxy for background motion using gyroscope
        const gyro = sensorDataRef.current.gyro;
        const bg_dx = (gyro.y || 0) * -5; // Gyro Y-axis rotation approximates horizontal motion
        const bg_dy = (gyro.x || 0) * 5;  // Gyro X-axis rotation approximates vertical motion

        const faceMagnitude = Math.sqrt(face_dx**2 + face_dy**2);
        const bgMagnitude = Math.sqrt(bg_dx**2 + bg_dy**2);

        const frame: FrameData = {
          timestamp: Date.now(),
          faceMesh: faceMesh,
          sensors: JSON.parse(JSON.stringify(sensorDataRef.current)), // Deep copy
          opticalFlowStats: {
            count: 30, // Dummy value
            avgX: bg_dx,
            avgY: bg_dy,
            avgMag: bgMagnitude,
            variance: Math.abs(gyro.z || 0) * 0.1, // Proxy variance from Z-axis rotation
          },
          motion_analysis: {
            face_dx,
            face_dy,
            bg_dx,
            bg_dy,
            relative_magnitude: Math.abs(faceMagnitude - bgMagnitude),
          },
          bg_variance: Math.abs(gyro.z || 0) * 0.1,
          meta: { camera_facing: "user" }
        };
        capturedFramesRef.current.push(frame);
        
        prevFaceCentroidRef.current = currentCentroid;
        prevTimestampRef.current = currentTimestamp;
        
        frameCountRef.current++;
        setStatus(`Scanning... ${frameCountRef.current}/${MAX_FRAMES}`);

        // Draw on canvas
        if (canvasRef.current) {
          const ctx = canvasRef.current.getContext('2d');
          if (ctx) {
            ctx.clearRect(0, 0, 640, 480);
            ctx.fillStyle = 'rgba(0, 255, 0, 0.7)';
            keypoints.forEach(point => {
              ctx.beginPath();
              ctx.arc(point.x, point.y, 2, 0, 2 * Math.PI);
              ctx.fill();
            });
          }
        }
      } else {
        if (canvasRef.current) {
          const ctx = canvasRef.current.getContext('2d');
          ctx?.clearRect(0, 0, 640, 480);
        }
      }
    }
    
    animationFrameIdRef.current = requestAnimationFrame(() => scanFrame(model));
  };


  return (
    <div className="flex flex-col items-center gap-4 w-full">
      <div className="relative w-full max-w-sm sm:max-w-md aspect-[4/3] rounded-lg overflow-hidden shadow-lg border-2 border-indigo-500">
        <video ref={videoRef} autoPlay playsInline muted className="absolute inset-0 w-full h-full object-cover transform -scale-x-100" />
        <canvas ref={canvasRef} width="640" height="480" className="absolute inset-0 w-full h-full transform -scale-x-100" />
        <div className="absolute bottom-0 left-0 right-0 bg-black/50 p-2 text-center">
          <p className="font-semibold text-lg">{status}</p>
          <div className="w-full bg-gray-600 rounded-full h-2.5 mt-2">
            <div className="bg-indigo-500 h-2.5 rounded-full" style={{ width: `${(frameCountRef.current / MAX_FRAMES) * 100}%` }}></div>
          </div>
        </div>
      </div>
       <button
        onClick={onCancel}
        className="w-full max-w-sm bg-gray-600 hover:bg-gray-700 text-white font-bold py-3 px-4 rounded-lg transition"
      >
        Cancel Scan
      </button>
    </div>
  );
};

export default FaceScanner;