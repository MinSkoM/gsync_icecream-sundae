// types.ts

export type Scenario = 'Normal' | 'LowLight';

// โครงสร้างข้อมูล Sensor (เหมือนเดิม)
export interface SensorValue {
  x: number | null;
  y: number | null;
  z: number | null;
}

export interface SensorData {
  accel: SensorValue;
  gyro: SensorValue;
}

// โครงสร้าง Frame ที่เปลี่ยนไป (faceMesh เป็น number[] ตาม data.json)
export interface FrameData {
  timestamp: number;
  faceMesh: number[]; // เปลี่ยนจาก Object เป็น Array ของตัวเลข [x1, y1, z1, x2, ...]
  sensors: SensorData;
}

// โครงสร้าง JSON ที่จะส่งไป Backend
export interface LivenessData {
  type: string;
  scenario: string;
  motion: string;
  data: FrameData[];
  meta?: {
    userAgent: string;
  };
}

// โครงสร้าง Response จาก Backend
export interface LivenessApiResponse {
  final_verdict: string;
  score: number; // ✅ เพิ่ม score รูทรวม (final_score) ที่ Backend ส่งมา
  details: {
    motion: {
      score: number;
      label: 'REAL' | 'SPOOF'; // ✅ ลบ passed ออกเพราะ Backend ส่งมาแค่ score กับ label
    };
    vision: {
      score: number;
      label: 'LIVE' | 'SPOOF'; // ✅ ลบ passed ออกเช่นกัน
    };
  };
  error?: string;
}