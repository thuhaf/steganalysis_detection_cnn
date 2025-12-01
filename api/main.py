from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
import torch
import sys
from pathlib import Path
import tempfile
import shutil

# Fix import
sys.path.append(str(Path(__file__).parent.parent))

# DÙNG HẾT CODE THẬT TRONG SRC
from src.inference.pipeline import StegAnalysisPipeline, PipelineConfig
from src.inference.explainer import StegExplainer  # hoặc tên class bạn đặt

app = FastAPI(
    title="StegDetect Pro - Steganalysis API",
    description="API phát hiện giấu tin ảnh & âm thanh - Full chức năng",
    version="2.0"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# === TẢI MODEL CHỈ 1 LẦN ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Đang khởi tạo pipeline trên {device}...")

pipeline = StegAnalysisPipeline(PipelineConfig(
    model_path="checkpoints/best_model.pth",
    model_name="resnet50_steg",
    modality="image",
    device=device,
    batch_size=16
))

explainer = StegExplainer(pipeline.model, device=device)  # dùng explainer thật của bạn
print("Pipeline & Explainer đã sẵn sàng! Chạy trên GPU:", device == "cuda")

# === ENDPOINTS CHUẨN ===
@app.get("/")
async def root():
    return {"message": "StegDetect Pro API đang chạy!", "docs": "/docs"}

@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        raise HTTPException(400, "Chỉ hỗ trợ ảnh PNG/JPG/BMP")

    temp_path = Path(tempfile.mktemp(suffix=Path(file.filename).suffix))
    try:
        with temp_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        result = pipeline.predict_single(str(temp_path))  # DÙNG HÀM THẬT 100%

        # Chuyển tensor → float để tránh lỗi 422
        return {
            "filename": file.filename,
            "prediction": result["prediction"],
            "confidence": float(result["confidence"]),
            "class_probabilities": {
                k: float(v) for k, v in result.get("class_probabilities", {}).items()
            },
            "inference_time": round(result.get("total_time", 0), 4),
            "message": "CÓ DẤU HIỆU GIẤU TIN!" if result["prediction"] == "Stego" else "Ảnh an toàn"
        }
    finally:
        temp_path.unlink(missing_ok=True)

@app.post("/detect/batch")
async def detect_batch(files: list[UploadFile] = File(...)):
    results = []
    temp_paths = []
    try:
        for file in files:
            temp_path = Path(tempfile.mktemp(suffix=Path(file.filename).suffix))
            with temp_path.open("wb") as f:
                shutil.copyfileobj(file.file, f)
            temp_paths.append(str(temp_path))

        batch_results = pipeline.predict_batch(temp_paths)  # DÙNG HÀM BATCH THẬT

        for r, file in zip(batch_results, files):
            results.append({
                "filename": file.filename,
                "prediction": r["prediction"],
                "confidence": float(r["confidence"])
            })
        return {"total": len(results), "results": results}
    finally:
        for p in temp_paths:
            Path(p).unlink(missing_ok=True)

@app.post("/explain")
async def explain(file: UploadFile = File(...)):
    temp_path = Path(tempfile.mktemp(suffix=".jpg"))
    try:
        with temp_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        result = explainer.explain_prediction(str(temp_path))
        return FileResponse(result["heatmap_path"], media_type="image/png", filename="heatmap_explanation.png")
    finally:
        temp_path.unlink(missing_ok=True)

# === CHẠY ===
if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
