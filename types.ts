
export interface SensorData {
  accel: { x: number | null; y: number | null; z: number | null };
  gyro: { x: number | null; y: number | null; z: number | null };
}

export interface OpticalFlowStats {
  count: number;
  avgX: number;
  avgY: number;
  avgMag: number;
  variance: number;
}

export interface MotionAnalysis {
  face_dx: number;
  face_dy: number;
  bg_dx: number;
  bg_dy: number;
  relative_magnitude: number;
}

export interface FrameData {
  timestamp: number;
  faceMesh: number[];
  sensors: SensorData;
  opticalFlowStats: OpticalFlowStats;
  motion_analysis: MotionAnalysis;
  bg_variance: number;
  image?: string; // Optional: As per original JSON structure
  meta?: { camera_facing: string }; // Optional: As per original JSON structure
}

export interface LivenessData {
  type: string;
  scenario: string;
  motion: string;
  data: FrameData[];
}

export interface MotionModelResult {
  status: string;
  label: 'REAL' | 'SPOOF';
  score: number;
  confidence: string;
  message?: string;
}

export interface VisionModelResult {
  status: string;
  label: 'LIVE' | 'SPOOF';
  score: number;
  message?: string;
}

export interface LivenessApiResponse {
  motion_model: MotionModelResult;
  vision_model: VisionModelResult;
  final_verdict: 'LIVENESS CONFIRMED' | 'LIVENESS DENIED';
}
