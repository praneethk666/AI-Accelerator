import axios from 'axios';

// Configure API base URL - change localhost:8000 to your backend server
export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// A finite default timeout matters: /agent/chat is synchronous and can run an ingest
// inline. With no timeout a hung backend never settles the promise, so the chat's
// `finally` never runs and the composer stays disabled forever (only a reload recovers).
const DEFAULT_TIMEOUT_MS = 60_000;
// Agent turns legitimately take minutes (multi-step retrieval, or an approved ingest),
// so they get a longer bound rather than the default.
export const AGENT_TIMEOUT_MS = 600_000;

const API = axios.create({
  baseURL: API_BASE_URL,
  timeout: DEFAULT_TIMEOUT_MS,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add response interceptor for error handling
API.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error);
    if (error.code === 'ECONNABORTED' || error.code === 'ETIMEDOUT') {
      throw new Error('The server took too long to respond. It may still be working — check the Ingestion page, or try a narrower question.');
    } else if (error.response?.status === 404) {
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
 * Get live ingestion progress for a document (per-step status + timings).
 * @param {string} fileId - The document id
 * @returns {Promise} { status, metrics:[{step, ms, status}], document_type, ... }
 */
export const getProgress = async (fileId) => {
  return API.get(`/files/${fileId}/progress`);
};


/**
 * Send a message to the agent — it picks which tool to call (ingest a file,
 * search the corpus, or run a read-only SQL query) instead of always doing
 * direct retrieval. A write action (e.g. ingest) comes back with
 * status: "needs_approval" instead of running immediately — re-send the SAME
 * message with approvedWrites=true to actually run it.
 * @param {string} message
 * @param {string} sessionId - keeps tool-calling conversation memory server-side
 * @param {boolean} approvedWrites
 * @returns {Promise} { status, answer, pending, tool_calls }
 */
export const sendAgentChat = async (message, sessionId, approvedWrites = false, approvedCalls = null, signal = null, activeDocumentId = null) => {
  return API.post('/agent/chat', {
    message,
    session_id: sessionId,
    approved_writes: approvedWrites,
    approved_calls: approvedCalls,
    active_document_id: activeDocumentId,
  }, { timeout: AGENT_TIMEOUT_MS, signal });
};


/** Past agent-chat conversations, most recently active first (for the sidebar). */
export const getAgentSessions = async () => API.get('/agent/sessions');

/** One conversation's full turn history — same shape /agent/chat returns per turn. */
export const getAgentSession = async (sessionId) => API.get(`/agent/sessions/${sessionId}`);

export const deleteAgentSession = async (sessionId) => API.delete(`/agent/sessions/${sessionId}`);

/** Update session metadata: { title?, pinned? }. */
export const patchAgentSession = async (sessionId, data) => API.patch(`/agent/sessions/${sessionId}`, data);

/**
 * Save a file to disk WITHOUT ingesting it — for the chat's attach-a-file flow.
 * The agent decides (with approval) whether/how to ingest it; this just gets
 * the bytes onto the server so there's a file_path to hand the agent.
 * @param {File} file
 * @returns {Promise} { file_path, filename }
 */
export const stageFile = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  return API.post('/files/stage', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};

export const ingestStagedFile = async ({ file_path, filename, session_id, message_id }) => {
  return API.post('/files/ingest-staged', { file_path, filename, session_id, message_id });
};

export const initDirectIngest = async (sessionId, { filename, staged_path, message_id }) => {
  return API.post(`/agent/sessions/${sessionId}/init-direct-ingest`, { filename, staged_path, message_id });
};

export const cancelDirectIngest = async (sessionId, { message_id, filename }) => {
  return API.post(`/agent/sessions/${sessionId}/cancel-direct-ingest`, { message_id, filename });
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

// ── Config / customer profiles (Settings page) ──────────────────────────────
export const getSettings = async () => API.get('/config/settings');
export const saveSettings = async (settings, saveAs = null) =>
  API.put('/config/settings', { settings, save_as: saveAs });
export const getConfigRaw = async () => API.get('/config/raw');
export const saveConfig = async (yamlText) => API.put('/config', { yaml: yamlText });
export const getProfiles = async () => API.get('/config/profiles');
export const saveProfile = async (name, yamlText) =>
  API.post('/config/profiles', { name, yaml: yamlText });
export const activateProfile = async (name) =>
  API.post('/config/activate', { name });

export const checkDoclingServer = async (url, mode) => API.get('/health/docling-server', { params: { url, mode }, timeout: 8000 });

export default API;