# Frontend & Backend Integration Guide

## 🎯 Complete Setup Instructions (Step-by-Step)

### Prerequisites Check
Before starting, ensure you have:
```bash
node --version        # Should be 16 or higher
npm --version         # Should be 8 or higher
python --version      # Should be 3.8 or higher
pip --version         # Package manager for Python
```

---

## Part 1: Backend Setup (FastAPI)

### 1.1 Install Dependencies

```bash
# Navigate to project root
cd d:\AI-Accelerator\AI-Accelerator

# Install main requirements
pip install -r requirements.txt

# Install backend-specific requirements
pip install -r backend_api/requirements.txt
```

**Required packages:**
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `python-multipart` - File upload support
- `pydantic` - Data validation

### 1.2 Verify Backend Structure

```bash
# Check backend_api folder
dir backend_api\

# Should see:
# - main.py (the API server)
# - config_loader.py (configuration)
# - requirements.txt (dependencies)
```

### 1.3 Start Backend Server

```bash
# Navigate to backend_api
cd backend_api

# Start Uvicorn server
python -m uvicorn main:app --reload --port 8000

# Expected output:
# INFO:     Uvicorn running on http://127.0.0.1:8000
# INFO:     Application startup complete
```

✅ **Backend is now ready!**

**Test it:**
```bash
# Open another terminal
curl http://localhost:8000/health
```

Should return:
```json
{
  "status": "healthy",
  "timestamp": "2026-06-12T...",
  "files_processed": 0,
  "chat_messages": 0
}
```

---

## Part 2: Frontend Setup (React + Vite)

### 2.1 Install Dependencies

```bash
# Navigate to frontend folder
cd d:\AI-Accelerator\AI-Accelerator\frontend

# Install npm dependencies
npm install

# This installs:
# - react & react-dom
# - vite (build tool)
# - axios (HTTP client)
# - react-dropzone (file upload)
# - @heroicons/react (icons)
# - tailwindcss (styling)
```

### 2.2 Verify Environment Configuration

```bash
# Verify .env.local exists
type .env.local

# Should contain:
# VITE_API_URL=http://localhost:8000
```

If missing, create it:
```bash
echo "VITE_API_URL=http://localhost:8000" > .env.local
```

### 2.3 Start Development Server

```bash
# From frontend directory
npm run dev

# Expected output:
# VITE v5.x.x  ready in xxx ms
# ➜  Local:   http://localhost:5173/
# ➜  press h + enter to show help
```

✅ **Frontend is now running!**

---

## Part 3: Complete Integration Test

### 3.1 Test Connection

1. Open browser: `http://localhost:5173`
2. You should see the **Document Ingestion** page
3. Check for "Backend Connected" indicator in top-right

If you see "Backend Offline":
```bash
# 1. Verify backend is running (port 8000)
netstat -ano | findstr :8000

# 2. Check if port is in use
Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess

# 3. Kill if needed
Stop-Process -Id <PID> -Force

# 4. Restart backend
cd backend_api && python -m uvicorn main:app --reload --port 8000
```

### 3.2 Test File Upload

1. Download a test PDF from the web
2. Drag and drop into the upload area (or click "Browse files")
3. Wait for upload to complete
4. Verify file appears in "RECENT FILES" section

**Expected response:**
```json
{
  "id": 1,
  "filename": "document.pdf",
  "status": "Ready",
  "route": "text_default",
  "document_type": "report",
  "industry": "automotive",
  "confidence": 0.85,
  "file_type": "pdf",
  "errors": []
}
```

### 3.3 Test Chat Functionality

1. Click the 💬 (chat) icon on any uploaded file
2. Type a question: "What is this document about?"
3. Click send or press Enter
4. Wait for response

**Expected response:**
```json
{
  "answer": "Based on the document 'document.pdf', I found information related to your question: 'What is this document about?'. The document is categorized as 'report' in the 'automotive' industry.",
  "sources": [{
    "filename": "document.pdf",
    "page": 1,
    "snippet": "Document: document.pdf, Type: report, Industry: automotive"
  }],
  "message_id": 0
}
```

---

## Part 4: Project File Structure

```
d:\AI-Accelerator\AI-Accelerator\
│
├── backend/                           # Core AI logic (categorization, etc.)
│   ├── categorize/
│   │   ├── categorize_tool.py
│   │   ├── classifier.py
│   │   └── taxonomy.py
│   ├── extraction/
│   ├── embeddings/
│   └── ... (other modules)
│
├── backend_api/                       # FastAPI Server
│   ├── main.py                        # ⭐ API endpoints
│   ├── routes.py
│   ├── config_loader.py
│   ├── requirements.txt
│   └── __init__.py
│
├── frontend/                          # React Frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── IngestionPage.jsx      # ⭐ Upload page
│   │   │   ├── ChatPage.jsx           # ⭐ Chat page
│   │   │   ├── Sidebar.jsx            # ⭐ Navigation
│   │   │   └── index.jsx
│   │   ├── api.jsx                    # ⭐ API client
│   │   ├── App.jsx                    # ⭐ Router & main app
│   │   ├── main.jsx
│   │   ├── index.css
│   │   └── index.js
│   ├── public/
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.cjs
│   ├── postcss.config.js
│   ├── .env.local
│   └── index.html
│
├── config/
│   ├── global.yaml                    # Global configuration
│   └── pipeline.example.yaml
│
├── tests/
│   ├── test_categorize.py
│   ├── test_extraction_tools.py
│   └── ... (other tests)
│
├── FRONTEND_SETUP.md                  # 📖 Setup guide
├── INTEGRATION.md                     # 📖 This file
├── README.md
└── requirements.txt
```

---

## Part 5: Key API Endpoints Reference

### Document Management

```
POST /upload
  ├─ Request: multipart file upload
  ├─ Response: { id, filename, route, document_type, industry, confidence, file_type, errors, upload_time }
  └─ Example: curl -F "file=@document.pdf" http://localhost:8000/upload

GET /files
  ├─ Request: None
  ├─ Response: Array of file objects
  └─ Example: curl http://localhost:8000/files

GET /files/{file_id}
  ├─ Request: file_id (integer)
  ├─ Response: Single file object
  └─ Example: curl http://localhost:8000/files/1

DELETE /files/{file_id}
  ├─ Request: file_id (integer)
  ├─ Response: { message: "File deleted successfully" }
  └─ Example: curl -X DELETE http://localhost:8000/files/1
```

### Chat

```
POST /chat
  ├─ Request: { question: string, file_id?: number }
  ├─ Response: { answer, sources, message_id }
  └─ Example: curl -X POST -H "Content-Type: application/json" \
     -d '{"question":"What?","file_id":1}' \
     http://localhost:8000/chat

GET /chat-history
  ├─ Request: None
  ├─ Response: Array of chat messages
  └─ Example: curl http://localhost:8000/chat-history
```

### System

```
GET /health
  ├─ Request: None
  ├─ Response: { status, timestamp, files_processed, chat_messages }
  └─ Example: curl http://localhost:8000/health

GET /
  ├─ Request: None
  ├─ Response: API documentation
  └─ Example: curl http://localhost:8000/
```

---

## Part 6: Component Communication Flow

### File Upload Flow
```
IngestionPage.jsx
    ↓
uploadFile() in api.jsx
    ↓
axios.post('/upload', formData)
    ↓
main.py: POST /upload
    ↓
CategorizeTool.run()
    ↓
Returns file metadata
    ↓
Updates state in IngestionPage
    ↓
Display in file list
```

### Chat Flow
```
ChatPage.jsx
    ↓
sendChat(question, fileId) in api.jsx
    ↓
axios.post('/chat', { question, file_id })
    ↓
main.py: POST /chat
    ↓
(Future: RAG/LLM processing)
    ↓
Returns { answer, sources }
    ↓
Updates messages in ChatPage
    ↓
Display with source citations
```

---

## Part 7: Debugging & Troubleshooting

### Backend Issues

**"ModuleNotFoundError: No module named 'fastapi'"**
```bash
pip install fastapi uvicorn
```

**"Port 8000 already in use"**
```bash
# Find process using port
netstat -ano | findstr :8000

# Kill process (replace PID)
taskkill /PID <PID> /F

# Or use different port
python -m uvicorn main:app --port 8001
# Then update VITE_API_URL in frontend
```

**"CORS errors in browser console"**
- Backend CORS is already configured
- Verify frontend is accessing correct API URL
- Check Network tab in DevTools

### Frontend Issues

**"Cannot find module '../api'"**
```bash
# Make sure api.jsx exists in src/
ls frontend/src/api.jsx
```

**"Blank page or infinite loading"**
1. Check browser console for errors (F12)
2. Verify `http://localhost:5173` loads
3. Check Network tab - verify API calls to localhost:8000
4. Refresh page (Ctrl+F5)

**"npm install fails"**
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and lock file
rm -r node_modules package-lock.json

# Reinstall
npm install
```

### Common Error Solutions

| Error | Solution |
|-------|----------|
| `Error: Cannot connect to backend` | Start backend first: `python -m uvicorn main:app --port 8000` |
| `CORS error in console` | Backend is running but CORS not configured (verify main.py) |
| `File upload fails` | Check temp_uploads directory permissions |
| `Chat doesn't respond` | Verify `/chat` endpoint in main.py |
| `Styling looks broken` | Run `npm install` to get tailwindcss |

---

## Part 8: Running Backend with Auto-Reload (Development)

### Option 1: Uvicorn with reload (Recommended)
```bash
cd backend_api
python -m uvicorn main:app --reload --port 8000
```
- Auto-restarts when files change
- Perfect for development

### Option 2: Run without reload
```bash
cd backend_api
python -m uvicorn main:app --port 8000
```

### Option 3: Run as Python script
```bash
cd backend_api
python main.py
# (Make sure main.py has: if __name__ == "__main__": uvicorn.run(...))
```

---

## Part 9: Running Frontend with Hot Module Replacement

```bash
cd frontend
npm run dev
```

**Features:**
- Hot Module Replacement (HMR) - changes appear instantly
- Browser auto-refreshes
- Error overlays in browser
- Accessible at `http://localhost:5173`

---

## Part 10: Environment Variables Reference

### Backend (.env or main.py)
```python
# Backend port
BACKEND_PORT = 8000

# CORS Origins
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
]

# File upload
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
TEMP_UPLOAD_DIR = "temp_uploads"
```

### Frontend (.env.local)
```bash
# API endpoint
VITE_API_URL=http://localhost:8000

# Environment
VITE_ENV=development
```

---

## Part 11: Database & Storage (For Production)

### Current Implementation (Development)
- ✅ In-memory storage (fast, good for testing)
- ✅ Temp file storage (simple, no database needed)

### Production Requirements
- ❌ Replace in-memory with database (PostgreSQL, MongoDB)
- ❌ Use cloud storage (Azure Blob, S3, Google Cloud Storage)
- ❌ Implement authentication
- ❌ Add rate limiting

### Database Schema Example
```sql
CREATE TABLE documents (
  id SERIAL PRIMARY KEY,
  filename VARCHAR(255),
  route VARCHAR(100),
  document_type VARCHAR(50),
  industry VARCHAR(50),
  confidence DECIMAL(3,2),
  file_type VARCHAR(20),
  file_size INT,
  upload_time TIMESTAMP,
  file_path VARCHAR(255),
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE chat_messages (
  id SERIAL PRIMARY KEY,
  file_id INT REFERENCES documents(id),
  question TEXT,
  answer TEXT,
  created_at TIMESTAMP DEFAULT NOW()
);
```

---

## Part 12: Quick Commands Reference

### Start Everything
```bash
# Terminal 1: Backend
cd d:\AI-Accelerator\AI-Accelerator\backend_api
python -m uvicorn main:app --reload --port 8000

# Terminal 2: Frontend
cd d:\AI-Accelerator\AI-Accelerator\frontend
npm run dev

# Terminal 3: Open browser
# http://localhost:5173
```

### Build for Production
```bash
# Frontend
cd frontend
npm run build
# Output: frontend/dist/

# Backend (already built, just deploy the folder)
# Deploy: backend_api/
```

### Clean & Reinstall
```bash
# Backend
pip install --upgrade -r requirements.txt

# Frontend
npm cache clean --force
rm -r node_modules package-lock.json
npm install
```

---

## 📞 Support Matrix

| Issue | Check | Solution |
|-------|-------|----------|
| Backend won't start | Port 8000 in use | Change port: `--port 8001` |
| Frontend won't load | Dependencies missing | `npm install` |
| Can't upload files | Backend offline | Start backend first |
| Chat not responding | API endpoint broken | Check /chat in main.py |
| Styles look wrong | Tailwind not loaded | `npm install` |
| CORS errors | Backend config | Verify CORSMiddleware in main.py |

---

**Total Setup Time:** ~10 minutes  
**Success Indicators:**
✅ Backend running on `http://localhost:8000`  
✅ Frontend running on `http://localhost:5173`  
✅ "Backend Connected" shows in UI  
✅ Can upload files  
✅ Can ask questions in chat  

**Next Steps:**
1. Upload a test document
2. Ask a question about it in chat
3. Review API responses in Network tab (DevTools)
4. Explore component code to understand flow

---

**Last Updated:** 2026-06-12  
**Version:** 1.0.0
