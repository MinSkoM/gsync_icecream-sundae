
import { LivenessApiResponse, LivenessData } from '../types';

// IMPORTANT: This URL must be configured as an environment variable in your deployment environment (e.g., Vercel).
// For local development, you can create a .env.local file with:
// API_URL=http://localhost:8000/api/predict/liveness
const API_URL = process.env.API_URL || 'http://localhost:8000/api/predict/liveness';

export const predictLiveness = async (
  videoBlob: Blob,
  jsonData: LivenessData
): Promise<LivenessApiResponse> => {
  const formData = new FormData();

  // 1. Append the video file
  formData.append('video_file', videoBlob, 'scan.mp4');

  // 2. Append the JSON file
  const jsonBlob = new Blob([JSON.stringify(jsonData)], { type: 'application/json' });
  formData.append('json_file', jsonBlob, 'data.json');

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      body: formData,
      // Note: Do not set 'Content-Type' header when using FormData with fetch,
      // the browser will set it automatically with the correct boundary.
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
