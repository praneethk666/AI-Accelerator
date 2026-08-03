# React UI Frontend

The React frontend provides a responsive user interface for document ingestion, real-time logging, and multi-turn agent chats.

## Technology Stack

* **Vite**: High-performance development server and bundle builder.
* **React**: Component-based UI library.
* **React Router**: Controls routing between pages.
* **Tailwind CSS / PostCSS**: Modern styling framework.
* **lucide-react**: UI icons.

## Page Layouts & Logic

### 1. Ingestion Dashboard (`components/IngestionPage.jsx`)
* **File Ingest**: Uses a file dropzone supporting PDFs, Word files, Excel files, PowerPoint files, and images.
* **Progress Tracking**: Polls the backend API using `GET /files/{id}` to fetch document status and logs.
* **Process Logging**: Renders logs from active background steps (e.g. classification, extraction, embedding).

### 2. Chat Panel (`components/ChatPage.jsx`)
* **Conversational Agent**: Chat interface with the document agent.
* **Inline Citations**: Renders clickable citation links. Clicking a citation opens a preview panel showing the referenced table data or page crop.
* **Tool-Call Badges**: Displays badges when the agent calls tools (e.g. `search_documents`, `sql_read`).
* **Action Approvals**: Renders approval cards for write actions. When the agent requests permission to run `ingest_document`, it displays an interactive prompt asking the user to confirm the upload.
* **Sidebar History**: Lists historical conversation threads retrieved via `GET /agent/sessions`, allowing users to open or delete history.

### 3. Configuration Panel (`components/SettingsPage.jsx`)
* **LLM / VLM Model Routing**: Select models and providers for different steps.
* **Prompt Customization**: Text areas to view and modify system prompts and instructions.
* **Hyperparameters**: Inputs to adjust parameters like chunk size, overlap, temperature, and retrieval strategies.
```

## Integration & API Client

All communications with the backend FastAPI service are routed through `frontend/src/api.jsx` using standard fetch requests.
