from fastapi import HTTPException, Request
from typing import Optional
import time
from collections import defaultdict
from datetime import datetime, timedelta
import os
from pathlib import Path

# Global pipeline instance (singleton)
_pipeline = None
_audio_pipeline = None
_model = None

# API Keys (in production, use database or environment variables)
VALID_API_KEYS = {
    "demo_key_12345": {"name": "Demo User", "tier": "free"},
    "prod_key_67890": {"name": "Production User", "tier": "premium"},
}


# ============================================================================
# Pipeline Management
# ============================================================================

async def get_pipeline():
    """Get or create pipeline instance (singleton)"""
    global _pipeline

    if _pipeline is None:
        from src.inference.pipeline import StegAnalysisPipeline, PipelineConfig

        # Load config
        config = PipelineConfig(
            model_path=os.getenv("MODEL_PATH", "checkpoints/best_model.pth"),
            model_name=os.getenv("MODEL_NAME", "resnet50_steg"),
            modality="image",
            device=os.getenv("DEVICE", "cuda"),
            batch_size=int(os.getenv("BATCH_SIZE", "32")),
            use_amp=True
        )

        _pipeline = StegAnalysisPipeline(config)
        print("✅ Image pipeline loaded successfully")

    return _pipeline


async def get_audio_pipeline():
    """Get or create audio pipeline instance"""
    global _audio_pipeline

    if _audio_pipeline is None:
        from src.inference.pipeline import StegAnalysisPipeline, PipelineConfig

        config = PipelineConfig(
            model_path=os.getenv("AUDIO_MODEL_PATH", "checkpoints/audio_model.pth"),
            model_name=os.getenv("AUDIO_MODEL_NAME", "audio_steg_cnn"),
            modality="audio",
            device=os.getenv("DEVICE", "cuda"),
            use_amp=True
        )

        _audio_pipeline = StegAnalysisPipeline(config)
        print("✅ Audio pipeline loaded successfully")

    return _audio_pipeline


async def get_model():
    """Get raw model for explainability"""
    global _model

    if _model is None:
        pipeline = await get_pipeline()
        _model = pipeline.model

    return _model


# ============================================================================
# Authentication
# ============================================================================

async def verify_api_key(api_key: str) -> dict:
    """
    Verify API key is valid

    Args:
        api_key: API key from Authorization header

    Returns:
        User info dict

    Raises:
        HTTPException if invalid
    """
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return VALID_API_KEYS[api_key]


# ============================================================================
# Rate Limiting
# ============================================================================

class RateLimiter:
    """Simple in-memory rate limiter"""

    def __init__(self, max_calls: int = 100, time_window: int = 60):
        """
        Args:
            max_calls: Maximum calls allowed
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = defaultdict(list)

    async def check_rate_limit(self, api_key: str):
        """Check if API key has exceeded rate limit"""
        now = datetime.now()

        # Clean old entries
        cutoff = now - timedelta(seconds=self.time_window)
        self.calls[api_key] = [
            call_time for call_time in self.calls[api_key]
            if call_time > cutoff
        ]

        # Check limit
        if len(self.calls[api_key]) >= self.max_calls:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Max {self.max_calls} calls per {self.time_window}s"
            )

        # Record this call
        self.calls[api_key].append(now)


# ============================================================================
# Request Validation
# ============================================================================

async def validate_file_size(request: Request, max_size_mb: int = 10):
    """Validate uploaded file size"""
    content_length = request.headers.get('content-length')

    if content_length:
        content_length = int(content_length)
        max_size_bytes = max_size_mb * 1024 * 1024

        if content_length > max_size_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {max_size_mb}MB"
            )
