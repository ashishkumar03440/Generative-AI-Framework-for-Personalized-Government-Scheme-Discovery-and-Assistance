"""Logging utilities for the FreshWeb-RAG backend.

Provides a module-level ``logger`` configured with a sensible format. The logger can be
imported from anywhere in the code base:

    from .utils import logging as app_logging
    app_logging.logger.info("message")
"""

import logging
from pathlib import Path

# Ensure a logs directory exists
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Basic configuration – in a real system this could be extended to rotate files,
# integrate with structlog, or emit JSON.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8"),
    ],
)

logger = logging.getLogger("freshweb_rag")

