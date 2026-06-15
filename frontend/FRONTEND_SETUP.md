
# AI-Accelerator Frontend Setup & Backend Integration Guide

## 🎯 Overview
This guide provides step-by-step instructions to run the document categorization application locally, connecting the React frontend with the FastAPI backend.

## 📋 Prerequisites
- Node.js 16+ and npm
- Python 3.8+
- Git

## 🚀 Quick Start (5 Minutes)

### Step 1: Start the Backend
```bash
cd d:\AI-Accelerator\AI-Accelerator

# Install backend dependencies
pip install -r requirements.txt
pip install -r backend_api/requirements.txt

# Run the backend server
cd backend_api
python -m uvicorn main:app --reload --port 8000
```

✅ **Backend should now be running at** `http://localhost:8000`

### Step 2: Start the Frontend (New Terminal)
```bash
cd d:\AI-Accelerator\AI-Accelerator\frontend

# Install dependencies (if not already done)
npm install

# Start development server
npm run dev
```

✅ **Frontend should now be running at** `http://localhost:5173`

### Step 3: Open in Browser
Navigate to `http://localhost:5173` and start uploading documents!

---

## 📁 Project Structure

```
AI-Accelerator/
├── backend_api/           # FastAPI backend
│   ├── main.py           # API endpoints
│   ├── config_loader.py  # Configuration
│   └── requirements.txt   # Python dependencies
├── frontend/             # React frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── IngestionPage.jsx    # Document upload interface
│   │   │   ├── ChatPage.jsx          # Chat interface
│   │   │   └── Sidebar.jsx           # Navigation sidebar
│   │   ├── api.jsx                   # API client
│   │   ├── App.jsx                   # Main app router
│   │   └── index.css                 # Tailwind styles
│   ├── package.json
│   └── vite.config.js
└── backend/              # Core AI logic
```

---

## 🔌 API Endpoints

### Document Upload & Management

**POST /upload**
- Upload and categorize documents
- Returns: `{ id, filename, route, document_type, industry, confidence, file_type, errors, upload_time }`

**GET /files**
- Get list of all processed files
- Returns: Array of file metadata

**GET /files/{file_id}**
- Get details of a specific file

**DELETE /files/{file_id}**
- Delete a file

### Chat

**POST /chat**
- Send a question about documents
- Body: `{ question, file_id (optional) }`
- Returns: `{ answer, sources, message_id }`

**GET /chat-history**
- Get all previous chat messages

### System

**GET /health**
- Health check endpoint
- Returns: `{ status, timestamp, files_processed, chat_messages }`

---

## 🎨 Frontend Features

### Ingestion Page (`/`)
- **Drag & Drop Upload**: Upload multiple files at once
- **File Validation**: Supports PDF, Excel, PowerPoint, Images
- **Real-time Status**: Shows categorization status (Ready, Review)
- **Metadata Display**: Route, document type, industry, confidence score
- **Quick Actions**: Chat about document or delete

### Chat Page (`/chat?fileId=ID`)
- **Document-aware Chat**: Ask questions about specific documents
- **Source References**: AI responses show source documents
- **Chat History**: Persistent conversation history
- **Real-time Feedback**: Animated loading indicators
- **Error Handling**: Clear error messages with recovery options

---

## 🔧 Configuration

### Backend Configuration
Edit `backend_api/main.py` to customize:

```python
CATEGORIZATION_CONFIG = {
    "type_to_route": {
        "cad_drawing": "cad_route",
        "circuit_diagram": "circuit_route",
        # ... add your routes here
    },
    "default_industry": "automotive",
    "categorization": {
        "industry_keywords": {
            "automotive": [...],
            "electronics": [...],
            # ... add your keywords here
        }
    }
}
```

### Frontend Configuration
Edit `frontend/.env` or `frontend/vite.config.js`:

```env
VITE_API_URL=http://localhost:8000
```

### CORS Settings
The backend accepts requests from:
- `http://localhost:5173`
- `http://localhost:3000`
- `http://127.0.0.1:5173`
- `http://127.0.0.1:3000`

To add more origins, edit `backend_api/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["YOUR_NEW_ORIGIN"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 🧪 Testing the Setup

### 1. Test Backend
```bash
# In a new terminal
curl http://localhost:8000/health
```

Should return:
```json
{
  "status": "healthy",
  "timestamp": "2026-06-12T10:30:45.123456",
  "files_processed": 0,
  "chat_messages": 0
}
```

### 2. Test File Upload
```bash
curl -X POST -F "file=@yourfile.pdf" \
  http://localhost:8000/upload
```

### 3. Test Chat
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}' \
  http://localhost:8000/chat
```

---

## 🐛 Troubleshooting

### "Cannot connect to backend"
1. ✅ Verify backend is running: `http://localhost:8000/health`
2. ✅ Check if port 8000 is already in use: `netstat -ano | findstr :8000`
3. ✅ Change backend port in `main.py` and update frontend API URL

### "CORS Error"
- Backend CORS is configured in `backend_api/main.py`
- Frontend is configured to send requests to `http://localhost:8000`
- If running on different machine, update CORS origins in backend

### "Module not found" errors
```bash
# Backend
pip install --upgrade -r requirements.txt

# Frontend  
npm install
```

### Files not uploading
1. Check file size (should be < 50MB by default)
2. Verify file type is supported
3. Check `temp_uploads/` directory exists
4. Review backend logs for errors

---

## 📦 Deployment

### Production Deployment Checklist

1. **Environment Variables**
   - Set `BACKEND_URL` for production
   - Update CORS origins
   - Configure database connection

2. **Build Frontend**
   ```bash
   npm run build
   # Output in frontend/dist/
   ```

3. **Run Backend**
   ```bash
   python -m uvicorn main:app --host 0.0.0.0 --port 8000
   ```

4. **Serve Frontend**
   - Use nginx/Apache for static files
   - Set up proxy to backend

5. **Database**
   - Replace in-memory storage with real database
   - Update file storage to use cloud storage (S3, Azure Blob, etc.)

---

## 📚 API Client Usage

The frontend uses the `api.jsx` module for all API calls:

```javascript
import { uploadFile, getFiles, sendChat, deleteFile } from '../api';

// Upload a file
const response = await uploadFile(fileObject);
console.log(response.data); // File metadata

// Get all files
const files = await getFiles();
console.log(files.data); // Array of files

// Send chat message
const answer = await sendChat("What is this?", fileId);
console.log(answer.data); // { answer, sources }

// Delete file
await deleteFile(fileId);
```

---

## 🎓 Component Guide

### IngestionPage Component
**File**: `frontend/src/components/IngestionPage.jsx`

Key features:
- Dropzone for file uploads
- Real-time file list with metadata
- Pipeline status visualization
- Error handling with recovery
- Server health check

### ChatPage Component
**File**: `frontend/src/components/ChatPage.jsx`

Key features:
- Document selection sidebar
- Multi-turn conversation
- Source citation display
- Chat history persistence
- Auto-scroll on new messages

### Sidebar Component
**File**: `frontend/src/components/Sidebar.jsx`

Key features:
- Navigation between pages
- Recent documents list
- Settings access
- Active page highlighting

---

## 🔐 Security Notes

### Current Implementation
- ⚠️ In-memory file storage (data lost on restart)
- ⚠️ No authentication/authorization
- ⚠️ Files stored in `temp_uploads/` locally

### For Production
- ✅ Implement database for file metadata
- ✅ Use cloud storage for documents (S3, Azure Blob)
- ✅ Add authentication (JWT, OAuth)
- ✅ Implement rate limiting
- ✅ Add input validation & sanitization
- ✅ Use HTTPS only
- ✅ Implement audit logging

---

## 🤝 Contributing

To extend the application:

1. **Add new document types**: Edit `CATEGORIZATION_CONFIG` in `main.py`
2. **Add new routes**: Create endpoint in `main.py`
3. **Update UI**: Modify component in `frontend/src/components/`
4. **Style changes**: Update Tailwind classes

---

## 📞 Support

For issues or questions:
1. Check troubleshooting section above
2. Review API documentation
3. Check console for detailed error messages
4. Review backend logs

---

**Last Updated**: 2026-06-12  
**Version**: 1.0.0
