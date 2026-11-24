"""Bootstrap script to start the FreshWeb-RAG FastAPI server directly with uvicorn."""

import os
import sys
import uvicorn

project_root = os.path.abspath(os.path.dirname(__file__))
freshweb_dir = os.path.join(project_root, "Freshweb_RAG")

for p in [project_root, freshweb_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

if __name__ == "__main__":
    from app.config import settings
    print(f"Starting FreshWeb-RAG backend on http://localhost:{settings.app_port} ...")
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,
        log_level="info",
    )
