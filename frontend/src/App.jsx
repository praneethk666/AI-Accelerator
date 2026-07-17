import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import IngestionPage from './components/IngestionPage';
import ChatPage from './components/ChatPage';
import SettingsPage from './components/SettingsPage';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<ChatPage />} />
        <Route path="/chat" element={<ChatPage />} />
        <Route path="/settings" element={<SettingsPage />} />
        <Route path="/ingest" element={<IngestionPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;