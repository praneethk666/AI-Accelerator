from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys
import shutil
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import CategorizeTool from backend.categorize module
from backend.categorize.categorize_tool import CategorizeTool

try:
    from .config_loader import load_config
except ImportError:
    load_config = lambda: {}

app = FastAPI(
    title="Document Categorization API",
    description="API for document categorization and chat",
    version="1.0.0"
)
# Enable CORS for React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for processed files (use a database for production)
processed_files: List[Dict[str, Any]] = []
chat_history: List[Dict[str, str]] = []

# Pydantic models for request/response
class ChatRequest(BaseModel):
    question: str
    file_id: Optional[int] = None

class UploadResponse(BaseModel):
    id: int
    filename: str
    status: str
    route: str
    document_type: str
    industry: str
    confidence: float
    file_type: str
    errors: List[str]
    upload_time: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    message_id: int

# Categorization configuration
CATEGORIZATION_CONFIG = {
    "type_to_route": {
        "cad_drawing": "cad_route",
        "circuit_diagram": "circuit_route",
        "datasheet": "diagram_route",
        "report": "text_default",
        "invoice": "text_default",
        "presentation": "presentation_route",
        "spreadsheet": "text_default",
        "image": "image_route",
        "unknown": "text_default"
    },
    "default_industry": "automotive",
    "categorization": {
        "industry_keywords": {
            "automotive": ["toyota", "ford", "bmw", "vehicle", "torque", "engine", "chassis", "transmission", "automotive"],
            "electronics": ["circuit", "semiconductor", "resistor", "capacitor", "pcb", "schematic", "voltage", "signal"],
            "manufacturing": ["assembly", "drawing", "tolerance", "weld", "machining", "fixture", "jig", "bom", "part number"],
            "finance": ["invoice", "balance sheet", "profit", "loss", "revenue", "ledger", "audit", "fiscal", "equity"],
            "legal": ["contract", "agreement", "clause", "court", "law", "nda", "litigation", "compliance", "liability"],
            "healthcare": ["patient", "diagnosis", "treatment", "medical", "pharma", "clinical", "dosage", "trial", "disease", "symptom"],
        },
        "confidence_thresholds": {"categorization_low_confidence": 0.5}
    }
}

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and categorize a document.
    Supports: PDF, Excel, PowerPoint, Images
    """
    try:
        # Validate file type
        allowed_extensions = {'.pdf', '.xlsx', '.xls', '.pptx', '.ppt', '.png', '.jpg', '.jpeg', '.gif', '.bmp'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_ext} not supported. Allowed: {', '.join(allowed_extensions)}"
            )

        # Save uploaded file
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run categorization pipeline using CategorizeTool
        tool = CategorizeTool()
        state = {"file_path": file_path}
        
        # Call the actual CategorizeTool.run() which returns:
        # {route, document_type, industry, confidence, file_type, reasoning, errors}
        result = tool.run(state, CATEGORIZATION_CONFIG)

        # Create file entry with metadata from categorization result
        file_entry = {
            "id": len(processed_files) + 1,
            "filename": file.filename,
            "status": "Ready" if result.get("confidence", 0) > 0.5 else "Review",
            "route": result.get("route", "text_default"),
            "document_type": result.get("document_type", "unknown"),
            "industry": result.get("industry", "general"),
            "confidence": round(result.get("confidence", 0.0), 3),
            "file_type": result.get("file_type", file_ext.strip('.')),
            "errors": result.get("errors", []),
            "upload_time": datetime.now().isoformat(),
            "file_path": file_path,  # Store for later retrieval
            "file_size": os.path.getsize(file_path)
        }
        
        processed_files.append(file_entry)
        return file_entry

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/files")
async def list_files():
    """Return list of all processed files with their metadata."""
    return sorted(processed_files, key=lambda x: x.get("upload_time", ""), reverse=True)


@app.get("/files/{file_id}")
async def get_file(file_id: int):
    """Get details of a specific file."""
    for f in processed_files:
        if f["id"] == file_id:
            return f
    raise HTTPException(status_code=404, detail="File not found")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint with RAG (Retrieval Augmented Generation).
    In production, this would retrieve relevant document chunks and generate answers.
    """
    try:
        question = request.question
        file_id = request.file_id
        
        if not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        # Find the relevant file
        current_file = None
        if file_id:
            current_file = next((f for f in processed_files if f["id"] == file_id), None)
            if not current_file:
                raise HTTPException(status_code=404, detail="File not found")

        # In a real implementation, you would:
        # 1. Use a vector database (e.g., Pinecone, Weaviate) to find relevant chunks
        # 2. Use an LLM (Claude, GPT-4, etc.) to generate the answer
        # 3. Return actual sources from the document
        
        # Mock response for demo
        answer = f"Based on the document '{current_file['filename'] if current_file else 'uploaded documents'}', "
        answer += f"I found information related to your question: '{question}'. "
        answer += f"The document is categorized as '{current_file['document_type'] if current_file else 'unknown'}' "
        answer += f"in the '{current_file['industry'] if current_file else 'general'}' industry."
        
        sources = []
        if current_file:
            sources.append({
                "filename": current_file["filename"],
                "page": 1,
                "snippet": f"Document: {current_file['filename']}, Type: {current_file['document_type']}, Industry: {current_file['industry']}"
            })

        # Add to chat history
        message_id = len(chat_history)
        chat_history.append({
            "id": message_id,
            "question": question,
            "answer": answer,
            "file_id": file_id,
            "timestamp": datetime.now().isoformat()
        })

        return ChatResponse(
            answer=answer,
            sources=sources,
            message_id=message_id
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@app.get("/chat-history")
async def get_chat_history():
    """Get chat history."""
    return chat_history


@app.delete("/files/{file_id}")
async def delete_file(file_id: int):
    """Delete a file from the processed files list."""
    global processed_files
    file_to_delete = next((f for f in processed_files if f["id"] == file_id), None)
    
    if not file_to_delete:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Delete file from disk
    try:
        if os.path.exists(file_to_delete["file_path"]):
            os.remove(file_to_delete["file_path"])
    except Exception as e:
        print(f"Warning: Could not delete file from disk: {str(e)}")
    
    # Remove from list
    processed_files = [f for f in processed_files if f["id"] != file_id]
    return {"message": "File deleted successfully"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "files_processed": len(processed_files),
        "chat_messages": len(chat_history)
    }


@app.get("/")
async def root():
    """Root endpoint with API documentation."""
    return {
        "name": "Document Categorization & Chat API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "POST /upload - Upload and categorize documents",
            "files": "GET /files - List all processed files",
            "file_detail": "GET /files/{file_id} - Get specific file details",
            "chat": "POST /chat - Chat about documents",
            "chat_history": "GET /chat-history - Get chat history",
            "delete_file": "DELETE /files/{file_id} - Delete a file",
            "health": "GET /health - Health check"
        }
    }