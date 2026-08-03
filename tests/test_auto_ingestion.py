import os
import shutil
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

# Add parent directory to path so we can import from backend
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.config import load_config
from backend.api.main import get_unique_path, auto_ingestion_loop, _SETTINGS_MAP

class TestAutoIngestionWatcher(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.watch_dir = os.path.join(self.temp_dir, "watch")
        self.uploads_dir = os.path.join(self.temp_dir, "uploads")
        
        # Mock settings
        self.mock_config = {
            "auto_ingestion": {
                "enabled": True,
                "watch_dir": self.watch_dir,
                "poll_interval": 1,
                "on_success": "move",
                "on_failure": "move"
            }
        }
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_settings_map_entries(self):
        self.assertIn("auto_ingestion_enabled", _SETTINGS_MAP)
        self.assertIn("auto_ingestion_watch_dir", _SETTINGS_MAP)
        self.assertIn("auto_ingestion_poll_interval", _SETTINGS_MAP)
        self.assertIn("auto_ingestion_on_success", _SETTINGS_MAP)
        self.assertIn("auto_ingestion_on_failure", _SETTINGS_MAP)

    def test_get_unique_path(self):
        os.makedirs(self.watch_dir, exist_ok=True)
        filename = "test.pdf"
        p1 = get_unique_path(self.watch_dir, filename)
        self.assertEqual(os.path.basename(p1), filename)
        
        # Create a file
        with open(p1, "w") as f:
            f.write("dummy")
            
        p2 = get_unique_path(self.watch_dir, filename)
        self.assertEqual(os.path.basename(p2), "test_1.pdf")

    @patch("backend.api.main._config")
    @patch("backend.api.main.PostgresStore")
    @patch("backend.api.main._run_ingestion")
    def test_watcher_loop_processes_file(self, mock_run_ingest, mock_pg_store, mock_config):
        # Setup mocks
        mock_config.get.side_effect = lambda k, d=None: self.mock_config.get(k, d)
        mock_config.__getitem__.side_effect = lambda k: self.mock_config[k]
        # Real finding, 3-Aug: patching UPLOAD_DIR with a bare MagicMock() (no
        # `new=`) let os.path.join(UPLOAD_DIR, ...) + os.makedirs(...) silently
        # create a REAL directory literally named "MagicMock/..." in the repo
        # root on every test run — patch with a real string path instead.
        self._upload_dir_patch = patch("backend.api.main.UPLOAD_DIR", self.uploads_dir)
        self._upload_dir_patch.start()
        self.addCleanup(self._upload_dir_patch.stop)

        # Mock PostgresStore behavior
        mock_pg_instance = MagicMock()
        mock_pg_store.return_value = mock_pg_instance
        # Simulate successful ingestion status ("ready")
        mock_pg_instance.get_document.return_value = {"status": "ready"}
        
        # Create watch folder and place a PDF there
        os.makedirs(self.watch_dir, exist_ok=True)
        test_pdf = os.path.join(self.watch_dir, "test.pdf")
        with open(test_pdf, "w") as f:
            f.write("PDF Content")
            
        # We need to run auto_ingestion_loop once and break out of it.
        # We can patch time.sleep to raise KeyboardInterrupt to exit the infinite loop after 1 cycle.
        cycle_count = 0
        original_sleep = time.sleep
        
        def mock_sleep(seconds):
            nonlocal cycle_count
            if seconds == 2: # the file size verification sleep
                return
            cycle_count += 1
            if cycle_count >= 1:
                raise KeyboardInterrupt("Break loop")
        
        with patch("time.sleep", side_effect=mock_sleep):
            try:
                auto_ingestion_loop()
            except KeyboardInterrupt:
                pass
                
        # Check that the file was picked up, copied to uploads, processed, and moved to processed subfolder
        self.assertTrue(mock_run_ingest.called)
        self.assertTrue(mock_pg_instance.insert_document.called)
        
        processed_file = os.path.join(self.watch_dir, "processed", "test.pdf")
        self.assertTrue(os.path.exists(processed_file))
        self.assertFalse(os.path.exists(test_pdf))

    @patch("backend.api.main._config")
    @patch("backend.api.main.PostgresStore")
    def test_watcher_loop_unsupported_file(self, mock_pg_store, mock_config):
        mock_config.get.side_effect = lambda k, d=None: self.mock_config.get(k, d)
        self._upload_dir_patch = patch("backend.api.main.UPLOAD_DIR", self.uploads_dir)
        self._upload_dir_patch.start()
        self.addCleanup(self._upload_dir_patch.stop)

        os.makedirs(self.watch_dir, exist_ok=True)
        test_unknown = os.path.join(self.watch_dir, "test.unsupported_extension")
        with open(test_unknown, "w") as f:
            f.write("some contents")
            
        cycle_count = 0
        def mock_sleep(seconds):
            nonlocal cycle_count
            if seconds == 2:
                return
            cycle_count += 1
            if cycle_count >= 1:
                raise KeyboardInterrupt("Break loop")
                
        with patch("time.sleep", side_effect=mock_sleep):
            try:
                auto_ingestion_loop()
            except KeyboardInterrupt:
                pass
                
        # Unsupported file should be moved to failed/ directly without ingestion
        failed_file = os.path.join(self.watch_dir, "failed", "test.unsupported_extension")
        self.assertTrue(os.path.exists(failed_file))
        self.assertFalse(os.path.exists(test_unknown))

if __name__ == "__main__":
    unittest.main()
