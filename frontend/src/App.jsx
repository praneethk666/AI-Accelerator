import { BrowserRouter, Routes, Route } from 'react-router-dom';
import IngestionPage from './components/IngestionPage';
import ChatPage from './components/ChatPage';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<IngestionPage />} />
        <Route path="/chat" element={<ChatPage />} />
        <Route path="*" element={<IngestionPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;