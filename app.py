"""app.py
FastAPI wrapper for Dolphin OCR‑VQA – low‑memory & tidy outputs
==============================================================

* Saves **all** artefacts to `<project>/result/`:

```
result/
 ├─ markdown/     # *.md and combined JSON
 └─ figures/      # cropped images
```
* Frees GPU VRAM after every heavy call (`clear_cuda()`).
* Sets `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` automatically if not present.
"""
from __future__ import annotations

import gc
import io
import os
import tempfile
from pathlib import Path

import anyio
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from omegaconf import OmegaConf
from PIL import Image

from chat import DOLPHIN
from processor_api import generate_markdown, process_document, process_element
from all_utils.utils import setup_output_dirs

# -----------------------------------------------------------------------------
# Helper: free reserved-but-unused GPU memory
# -----------------------------------------------------------------------------

def clear_cuda() -> None:
    """Release cached and IPC-held VRAM so next request starts fresh."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# Ensure allocator uses low‑fragmentation mode unless user overrode it
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

# -----------------------------------------------------------------------------
# Config + shared model instance
# -----------------------------------------------------------------------------
CFG_PATH = os.getenv("DOLPHIN_CONFIG", "config/Dolphin.yaml")
CONFIG = OmegaConf.load(CFG_PATH)
ROOT_OUT = Path.cwd() / "result"          # <project>/result
setup_output_dirs(ROOT_OUT)                # create result/, markdown/, figures/


def get_model() -> DOLPHIN:  # lazy singleton
    if not hasattr(get_model, "_model") or get_model._model is None:  # type: ignore[attr-defined]
        get_model._model = DOLPHIN(CONFIG)  # type: ignore[attr-defined]
        # Uncomment next two lines if ваш GPU поддерживает FP16 и хочется ещё -35 % VRAM
        # get_model._model.model.half()
        # get_model._model.vpm.half()
    return get_model._model  # type: ignore[return-value]

# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Dolphin test",
    version="0.1.0",
    description="For test",
    docs_url="/docs",
)

# -----------------------------------------------------------------------------
# /inference : single‑image Q&A
# -----------------------------------------------------------------------------
@app.post("/inference", response_model=dict[str, str])
async def inference(
    file: UploadFile = File(..., description="PNG/JPEG image file"),
    question: str = Form(..., description="Prompt / question for the image"),
) -> JSONResponse:
    if not file.content_type.startswith("image/"):
        raise HTTPException(415, "Only image uploads are supported")

    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, "Invalid image file") from exc

    answer = await anyio.to_thread.run_sync(get_model().chat, question, img)
    clear_cuda()
    return JSONResponse({"answer": answer})

# -----------------------------------------------------------------------------
# /element : OCR one element (text / table / formula)
# -----------------------------------------------------------------------------
@app.post("/element", response_model=dict[str, object])
async def element(
    file: UploadFile = File(..., description="PNG/JPEG of a document element"),
    element_type: str = Form(..., description="text | table | formula"),
) -> JSONResponse:
    if not file.content_type.startswith("image/"):
        raise HTTPException(415, "Only image uploads are supported")

    suffix = Path(file.filename).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    text, results = await anyio.to_thread.run_sync(
        process_element, tmp_path, get_model(), element_type, str(ROOT_OUT)
    )
    clear_cuda()
    return JSONResponse({"text": text, "results": results})

# -----------------------------------------------------------------------------
# /markdown : full PDF / image → Markdown
# -----------------------------------------------------------------------------
@app.post("/markdown", response_class=PlainTextResponse)
async def markdown(file: UploadFile = File(..., description="PDF or image")) -> PlainTextResponse:
    # 1) save upload to a temp file on disk (required by pymupdf)
    suffix = Path(file.filename).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # 2) run heavy parsing in a worker thread
    _, results = await anyio.to_thread.run_sync(
        process_document,
        tmp_path,
        get_model(),
        str(ROOT_OUT),
        CONFIG.model.max_batch_size if hasattr(CONFIG.model, "max_batch_size") else 4,
    )

    # 3) generate Markdown text
    md = generate_markdown(results)

    # 4) optional: remove temp source file to save disk space
    try:
        os.remove(tmp_path)
    except OSError:
        pass

    clear_cuda()
    return PlainTextResponse(md, media_type="text/markdown")

# -----------------------------------------------------------------------------
# Dev entry‑point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
