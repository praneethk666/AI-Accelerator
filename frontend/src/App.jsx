import { BrowserRouter, Routes, Route } from 'react-router-dom';
import IngestionPage from './components/IngestionPage';
import ChatPage from './components/ChatPage';
import SettingsPage from './components/SettingsPage';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<IngestionPage />} />
        <Route path="/chat" element={<ChatPage />} />
        <Route path="/settings" element={<SettingsPage />} />
        <Route path="*" element={<IngestionPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;