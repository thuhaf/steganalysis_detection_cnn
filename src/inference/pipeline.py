import json
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from .predictor import StegPredictor, load_model_for_inference
from .batch_predictor import BatchPredictor
from data.preprocessing import ImagePreprocessor, AudioPreprocessor
from models.model_registry import ModelRegistry
from src.utils.logger import InferenceLogger
# from utils.logger import InferenceLogger
class InferenceLogger:
    def log_prediction(self, file_path, result):
        pass  # Tạm thời tắt log


@dataclass
class PipelineConfig:
    """Configuration for inference pipeline"""
    model_path: str
    model_name: str
    modality: str = 'image'          # 'image' or 'audio'
    device: str = 'cuda'
    batch_size: int = 32
    threshold: float = 0.5
    use_amp: bool = True
    num_workers: int = 4
    img_size: int = 224
    sample_rate: int = 16000
    audio_duration: float = 3.0


class StegAnalysisPipeline:
    """
    Complete end-to-end pipeline for steganalysis
    Handles model loading, preprocessing, prediction, and reporting
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

        # Setup logger
        self.logger = InferenceLogger()

        # Load model
        self.model = self._load_model()

        # Setup preprocessing
        self.transform = self._setup_transform()

        # Create predictors
        self.single_predictor = StegPredictor(
            model=self.model,
            device=str(self.device),
            transform=self.transform
        )

        self.batch_predictor = BatchPredictor(
            model=self.model,
            device=str(self.device),
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            use_amp=config.use_amp
        )

        print(f"Pipeline initialized successfully")
        print(f"  Model: {config.model_name}")
        print(f"  Modality: {config.modality}")
        print(f"  Device: {self.device}")

    def _load_model(self) -> nn.Module:
        """Load the trained model"""
        print(f"Loading model from {self.config.model_path}...")
        model_info = ModelRegistry.get_model_info(self.config.model_name)
        model_class = model_info['class']

        model = load_model_for_inference(
            model_path=self.config.model_path,
            model_class=model_class,
            device=str(self.device),
            num_classes=2
        )
        return model

    def _setup_transform(self):
        """Setup preprocessing transforms"""
        if self.config.modality == 'image':
            return ImagePreprocessor.get_val_transforms(self.config.img_size)
        else:
            return AudioPreprocessor(
                sample_rate=self.config.sample_rate,
                n_mels=128
            )

    def predict_single(self, file_path: str, return_details: bool = True) -> Dict:
        """Predict a single file"""
        start_time = time.time()

        if self.config.modality == 'image':
            result = self.single_predictor.predict_image(
                file_path,
                return_confidence=return_details
            )
        else:
            result = self.single_predictor.predict_audio(
                file_path,
                sample_rate=self.config.sample_rate,
                duration=self.config.audio_duration,
                return_confidence=return_details
            )

        # Custom threshold
        if self.config.threshold != 0.5 and 'class_probabilities' in result:
            stego_prob = result['class_probabilities'].get('Stego', 0.0)
            result['prediction_thresholded'] = 'Stego' if stego_prob >= self.config.threshold else 'Cover'
            result['threshold_used'] = self.config.threshold

        result['total_time'] = time.time() - start_time
        self.logger.log_prediction(file_path, result)
        return result

    def predict_batch(self, file_paths: List[str], show_progress: bool = True) -> List[Dict]:
        """Predict multiple files"""
        start_time = time.time()

        if self.config.modality == 'image':
            results = self.batch_predictor.predict_images(
                file_paths,
                self.transform,
                show_progress=show_progress
            )
        else:  # audio
            results = self.batch_predictor.predict_audios(
                file_paths,
                self.transform,
                sample_rate=self.config.sample_rate,
                duration=self.config.audio_duration,
                show_progress=show_progress
            )

        total_time = time.time() - start_time
        for r in results:
            r['total_time'] = total_time
        return results

    def predict_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
        file_extensions: Optional[List[str]] = None
    ) -> List[Dict]:
        """Predict all files in a directory"""
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise ValueError(f"Directory not found: {directory}")

        # Collect file paths
        if file_extensions is None:
            file_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.wav', '.mp3', '.flac']

        paths = []
        pattern = "**/*" if recursive else "*"
        for ext in file_extensions:
            paths.extend(dir_path.glob(f"{pattern}{ext}"))
            paths.extend(dir_path.glob(f"{pattern}{ext.upper()}"))

        paths = [str(p) for p in paths if p.is_file()]
        print(f"Found {len(paths)} files in {directory}")
        return self.predict_batch(paths, show_progress=True)

    def analyze_and_report(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path] = './results',
        save_format: str = 'json'
    ) -> Dict:
        """
        Complete analysis with automatic report generation
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nStarting analysis of: {input_path}")
        print("=" * 60)

        # Predict
        if input_path.is_file():
            results = [self.predict_single(str(input_path))]
        else:
            results = self.predict_directory(str(input_path))

        # Summary
        summary = self.batch_predictor.get_summary(results)

        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"predictions_{timestamp}.{save_format}"
        self.batch_predictor.save_results(results, str(results_file), format=save_format)

        summary_file = output_dir / f"summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Generate & save report
        report = self._generate_report(results, summary)
        report_file = output_dir / f"report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(report)

        self.batch_predictor.print_summary(results)

        print(f"\nResults saved to: {output_dir}")
        print(f"  Predictions: {results_file}")
        print(f"  Summary:     {summary_file}")
        print(f"  Report:      {report_file}")

        return {
            'results': results,
            'summary': summary,
            'output_files': {
                'predictions': str(results_file),
                'summary': str(summary_file),
                'report': str(report_file)
            }
        }

    def _generate_report(self, results: List[Dict], summary: Dict) -> str:
        """Generate beautiful text report"""
        lines = [
            "=" * 80,
            "STEGANALYSIS DETECTION REPORT",
            "=" * 80,
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Model: {self.config.model_name}",
            f"Modality: {self.config.modality}",
            "",
            "SUMMARY",
            "-" * 80,
            f"Total files analyzed:     {summary['total_files']}",
            f"Successfully processed:   {summary['successful']}",
            f"Processing errors:        {summary['errors']}",
            f"Error rate:               {summary['error_rate']:.2%}",
            "",
            f"Steganography detected:   {summary['stego_detected']} ({summary['stego_percentage']:.2f}%)",
            f"Clean (cover) files:      {summary['cover_images']} ({summary['cover_percentage']:.2f}%)",
            f"Average confidence:       {summary['average_confidence']:.4f}",
            "",
            "DETAILED RESULTS",
            "-" * 80,
        ]

        # Stego files
        stego = [r for r in results if r.get('prediction') == 'Stego']
        if stego:
            lines.append("\nFiles with HIDDEN DATA (Stego):")
            for r in sorted(stego, key=lambda x: x.get('confidence', 0), reverse=True):
                name = Path(r['file_path']).name
                conf = r.get('confidence', 0.0)
                lines.append(f"  {name:<50} → Confidence: {conf:.4f}")

        # Top clean files
        cover = [r for r in results if r.get('prediction') == 'Cover']
        if cover:
            lines.append("\nTop 10 CLEAN files (highest confidence):")
            for r in sorted(cover, key=lambda x: x.get('confidence', 0), reverse=True)[:10]:
                name = Path(r['file_path']).name
                conf = r.get('confidence', 0.0)
                lines.append(f"  {name:<50} → Confidence: {conf:.4f}")

        lines.extend(["", "=" * 80])
        return "\n".join(lines)

    def scan_and_filter(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        copy_stego: bool = True,
        min_confidence: float = 0.7
    ) -> Dict:
        """
        Scan directory and optionally copy high-confidence stego files
        """
        results = self.predict_directory(input_dir)
        stego_files = [
            r for r in results
            if r.get('prediction') == 'Stego' and r.get('confidence', 0) >= min_confidence
        ]

        print(f"\nFound {len(stego_files)} high-confidence stego files (>= {min_confidence})")

        if copy_stego and stego_files:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            print(f"Copying to {output_dir}...")
            for r in stego_files:
                shutil.copy2(r['file_path'], out_path / Path(r['file_path']).name)
            print("Copy completed!")

        return {
            'total_scanned': len(results),
            'high_confidence_stego': len(stego_files),
            'files': [r['file_path'] for r in stego_files]
        }


# ========================== Helper Functions ==========================

def create_pipeline_from_config(config_path: str) -> StegAnalysisPipeline:
    """Load pipeline from JSON config file"""
    with open(config_path) as f:
        config_dict = json.load(f)
    config = PipelineConfig(**config_dict)
    return StegAnalysisPipeline(config)


def quick_predict(
    file_path: str,
    model_path: str,
    model_name: str = 'resnet50_steg',
    device: str = 'cuda'
) -> Dict:
    """One-liner prediction"""
    config = PipelineConfig(model_path=model_path, model_name=model_name, device=device)
    pipeline = StegAnalysisPipeline(config)
    return pipeline.predict_single(file_path)
