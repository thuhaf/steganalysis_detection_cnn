from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime


class DetectionResult(BaseModel):
    """Single detection result"""
    prediction: str = Field(..., description="Cover or Stego")
    predicted_class: int = Field(..., description="0=Cover, 1=Stego")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    class_probabilities: Dict[str, float] = Field(..., description="Probabilities for each class")
    inference_time: float = Field(..., description="Inference time in seconds")
    file_name: Optional[str] = Field(None, description="Original filename")
    file_path: Optional[str] = Field(None, description="Temporary file path")
    threshold_used: Optional[float] = Field(None, description="Custom threshold if applied")
    features: Optional[Dict[str, Any]] = Field(None, description="Extracted features")

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "Stego",
                "predicted_class": 1,
                "confidence": 0.8543,
                "class_probabilities": {
                    "Cover": 0.1457,
                    "Stego": 0.8543
                },
                "inference_time": 0.0234,
                "file_name": "suspicious_image.jpg"
            }
        }


class BatchDetectionResult(BaseModel):
    """Batch detection results"""
    total_files: int = Field(..., description="Total files processed")
    stego_detected: int = Field(..., description="Number of stego files found")
    cover_detected: int = Field(..., description="Number of cover files found")
    results: List[DetectionResult] = Field(..., description="Individual results")
    total_time: float = Field(..., description="Total processing time")
    avg_time_per_file: float = Field(..., description="Average time per file")

    class Config:
        json_schema_extra = {
            "example": {
                "total_files": 10,
                "stego_detected": 3,
                "cover_detected": 7,
                "total_time": 1.234,
                "avg_time_per_file": 0.1234,
                "results": []
            }
        }


class DetectionRequest(BaseModel):
    """Detection configuration"""
    threshold: float = Field(0.5, ge=0, le=1, description="Classification threshold")
    return_features: bool = Field(False, description="Include feature extraction")
    use_ensemble: bool = Field(False, description="Use ensemble of models")

    @validator('threshold')
    def validate_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Threshold must be between 0 and 1')
        return v


class ModelInfo(BaseModel):
    """Information about available model"""
    name: str = Field(..., description="Model name")
    modality: str = Field(..., description="image or audio")
    description: str = Field(..., description="Model description")
    accuracy: Optional[float] = Field(None, description="Model accuracy")
    size_mb: Optional[float] = Field(None, description="Model size in MB")


class HealthCheck(BaseModel):
    """Health check response"""
    status: str = Field(..., description="healthy or unhealthy")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    model_loaded: Optional[bool] = Field(None, description="Is model loaded")
    gpu_available: Optional[bool] = Field(None, description="Is GPU available")
    error: Optional[str] = Field(None, description="Error message if unhealthy")


class ExplanationResult(BaseModel):
    """Explanation/interpretability result"""
    prediction: str
    confidence: float
    heatmap_url: str = Field(..., description="URL to heatmap image")
    important_regions: List[Dict[str, Any]] = Field(..., description="Key regions")
    method: str = Field(..., description="Explanation method used")


class ErrorResponse(BaseModel):
    """Error response"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error info")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
