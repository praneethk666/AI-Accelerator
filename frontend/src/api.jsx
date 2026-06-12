import axios from 'axios';

// Configure API base URL - change localhost:8000 to your backend server
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const API = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add response interceptor for error handling
API.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error);
    if (error.response?.status === 404) {
      throw new Error('Resource not found');
    } else if (error.response?.status === 400) {
      throw new Error(error.response?.data?.detail || 'Invalid request');
    } else if (error.response?.status === 500) {
      throw new Error('Server error - please try again later');
    } else if (error.message === 'Network Error') {
      throw new Error('Cannot connect to server. Make sure the backend is running on ' + API_BASE_URL);
    }
    throw error;
  }
);

/**
 * Upload a file for categorization
 * @param {File} file - The file to upload
 * @returns {Promise} File metadata with categorization results
 */
export const uploadFile = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const config = {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress: (progressEvent) => {
      const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
      console.log(`Upload progress: ${percentCompleted}%`);
    },
  };
  
  return API.post('/upload', formData, config);
};

/**
 * Get list of all processed files
 * @returns {Promise} Array of file metadata
 */
export const getFiles = async () => {
  return API.get('/files');
};

/**
 * Get details of a specific file
 * @param {number} fileId - The file ID
 * @returns {Promise} File metadata
 */
export const getFile = async (fileId) => {
  return API.get(`/files/${fileId}`);
};

/**
 * Send a chat message about documents
 * @param {string} question - The question to ask
 * @param {number} fileId - Optional file ID to scope the question
 * @returns {Promise} Chat response with answer and sources
 */
export const sendChat = async (question, fileId = null) => {
  const payload = {
    question,
    ...(fileId && { file_id: fileId }),
  };
  return API.post('/chat', payload);
};

/**
 * Get chat history
 * @returns {Promise} Array of chat messages
 */
export const getChatHistory = async () => {
  return API.get('/chat-history');
};

/**
 * Delete a file
 * @param {number} fileId - The file ID to delete
 * @returns {Promise} Deletion confirmation
 */
export const deleteFile = async (fileId) => {
  return API.delete(`/files/${fileId}`);
};

/**
 * Health check - verify backend is running
 * @returns {Promise} Health status
 */
export const healthCheck = async () => {
  return API.get('/health');
};

export default API;