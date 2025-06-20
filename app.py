"""
app.py

Clean, minimal FastAPI for Dolphin OCR‑VQA and Document Processing

Endpoints:
  • POST /inference  – Image + question ⇒ JSON {answer}
  • POST /element    – Image + element_type ⇒ JSON {text, results}
  • POST /markdown   – PDF or image ⇒ Generated Markdown
"""
from __future__ import annotations
import io
import os
import tempfile
import uvicorn
import anyio
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from omegaconf import OmegaConf
from PIL import Image

from chat import DOLPHIN
from processor_api import process_element, process_document, generate_markdown

# Load configuration
CFG_PATH = os.getenv("DOLPHIN_CONFIG", "config/Dolphin.yaml")
CONFIG = OmegaConf.load(CFG_PATH)

# Singleton model instance
def get_model() -> DOLPHIN:
    if not hasattr(get_model, "_model") or get_model._model is None:
        get_model._model = DOLPHIN(CONFIG)
    return get_model._model  # type: ignore

app = FastAPI(
    title="Dolphin OCR‑VQA API",
    version="0.1.0",
    description="Image question answering, element parsing, and Markdown generation",
    docs_url="/docs",
)

@app.post("/inference", response_model=dict[str, str])
async def inference(
    file: UploadFile = File(..., description="PNG/JPEG image file"),
    question: str = Form(..., description="Prompt/question for the image"),
) -> JSONResponse:
    if not file.content_type.startswith("image/"):
        raise HTTPException(415, "Only image uploads supported")
    try:
        data = await file.read()
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")
    model = get_model()
    answer = await anyio.to_thread.run_sync(model.chat, question, image)
    return JSONResponse({"answer": answer})

@app.post("/element", response_model=dict[str, object])
async def element(
    file: UploadFile = File(..., description="PNG/JPEG image of a document element"),
    element_type: str = Form(..., description="Type of element: text/table/formula"),
) -> JSONResponse:
    if not file.content_type.startswith("image/"):
        raise HTTPException(415, "Only image uploads supported")
    # write upload to temp file
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    model = get_model()
    text, results = await anyio.to_thread.run_sync(
        process_element, tmp_path, model, element_type, None
    )
    return JSONResponse({"text": text, "results": results})

@app.post("/markdown", response_class=PlainTextResponse)
async def markdown(
    file: UploadFile = File(..., description="PDF or image file"),
) -> PlainTextResponse:
    # save to temp
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    model = get_model()
    save_dir = os.path.dirname(tmp_path)
    _, results = await anyio.to_thread.run_sync(
        process_document,
        tmp_path, model, save_dir, CONFIG.model.max_batch_size if hasattr(CONFIG.model, 'max_batch_size') else 4
    )
    md = generate_markdown(results)
    return PlainTextResponse(md, media_type="text/markdown")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
