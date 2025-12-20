"""
Web Server for Concord AI Chat Interface
Serves the frontend and proxies API requests
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

app = FastAPI(title="Concord AI Web Interface")

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")

# Serve static files
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
async def root():
    """Serve the main page."""
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


@app.get("/style.css")
async def get_css():
    """Serve CSS."""
    return FileResponse(
        os.path.join(WEB_DIR, "style.css"),
        media_type="text/css"
    )


@app.get("/app.js")
async def get_js():
    """Serve JavaScript."""
    return FileResponse(
        os.path.join(WEB_DIR, "app.js"),
        media_type="application/javascript"
    )


if __name__ == "__main__":
    print("\n" + "="*60)
    print("CONCORD AI WEB INTERFACE")
    print("="*60)
    print("\nWeb UI: http://localhost:3000")
    print("\nRequired services:")
    print("  - RAG API: http://localhost:8000")
    print("  - Analytics API: http://localhost:8001")
    print("="*60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=3000)
