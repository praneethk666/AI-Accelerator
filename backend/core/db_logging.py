import logging
import logging.handlers
import queue
import time
from backend.storage.postgres_store import PostgresStore

class PostgresLogHandler(logging.Handler):
    """A log handler that batches logs and writes them to PostgreSQL.
    This handler expects to be run in a separate thread via QueueListener
    to ensure database writes do not block the application."""

    def __init__(self, batch_size=50, flush_interval=5.0):
        super().__init__()
        self._pg = None
        self.batch = []
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.last_flush = time.time()

    def get_pg(self):
        if not self._pg:
            # Initialize PostgresStore only when we actually need to flush
            # (which happens in the QueueListener background thread)
            self._pg = PostgresStore()
        return self._pg

    def emit(self, record):
        try:
            msg = self.format(record)
            exc_text = None
            if record.exc_info:
                exc_text = self.formatter.formatException(record.exc_info)
            
            self.batch.append({
                "level": record.levelname,
                "logger_name": record.name,
                "message": msg,
                "exception": exc_text
            })

            now = time.time()
            if len(self.batch) >= self.batch_size or (now - self.last_flush) >= self.flush_interval:
                self.flush()

        except Exception:
            self.handleError(record)

    def flush(self):
        """Write the current batch to the database."""
        if not self.batch:
            return
        
        try:
            pg = self.get_pg()
            pg.write_log_batch(self.batch)
        except Exception:
            # We swallow exceptions here because if the DB is down, we don't
            # want to crash the logger. The application console will still show it.
            pass
        finally:
            self.batch = []
            self.last_flush = time.time()

def setup_db_logging(level=logging.INFO):
    """Configure asynchronous database logging. Call this after basicConfig."""
    log_queue = queue.Queue(-1)  # infinite size queue
    
    # Create the handler that actually writes to the DB
    pg_handler = PostgresLogHandler()
    pg_handler.setLevel(level)
    
    # Optional: basic formatting for the message column
    formatter = logging.Formatter('%(message)s')
    pg_handler.setFormatter(formatter)
    
    # We use QueueHandler to put records onto the queue (non-blocking)
    queue_handler = logging.handlers.QueueHandler(log_queue)
    queue_handler.setLevel(level)
    
    # The listener runs in a background thread and passes records to pg_handler
    listener = logging.handlers.QueueListener(log_queue, pg_handler)
    listener.start()
    
    # Attach the QueueHandler to the root logger so it catches everything
    logging.getLogger().addHandler(queue_handler)
    
    return listener
