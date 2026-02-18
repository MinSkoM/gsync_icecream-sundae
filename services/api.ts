import { LivenessApiResponse, LivenessData } from '../types';

// ✅ แก้ตรงนี้: ใช้ import.meta.env.VITE_API_URL สำหรับ Vite
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/predict/liveness';

export const predictLiveness = async (
  videoBlob: Blob,
  jsonData: LivenessData
): Promise<LivenessApiResponse> => {
  const formData = new FormData();

  formData.append('video_file', videoBlob, 'scan.mp4');

  const jsonBlob = new Blob([JSON.stringify(jsonData)], { type: 'application/json' });
  formData.append('json_file', jsonBlob, 'data.json');

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'An unknown error occurred' }));
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error("API call failed:", error);
    if (error instanceof Error) {
        throw new Error(`Failed to get liveness prediction: ${error.message}`);
    }
    throw new Error('An unexpected error occurred during the API call.');
  }
};