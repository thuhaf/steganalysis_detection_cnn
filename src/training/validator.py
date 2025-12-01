import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

class StegValidator:
    """Validator for steganalysis models"""

    def __init__(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        device: str = 'cuda',
        use_amp: bool = True
    ):
        """
        Args:
            model: Model to validate
            val_loader: Validation data loader
            device: Device to use
            use_amp: Use automatic mixed precision
        """
        self.model = model.to(device)
        self.val_loader = val_loader
        self.device = device
        self.use_amp = use_amp and device == 'cuda'

    @torch.no_grad()
    def validate(
        self,
        criterion: Optional[nn.Module] = None,
        return_predictions: bool = False,
        verbose: bool = True
    ) -> Dict:
        """
        Validate the model

        Args:
            criterion: Loss function (optional)
            return_predictions: Whether to return all predictions
            verbose: Show progress bar

        Returns:
            Dictionary containing metrics and optionally predictions
        """
        self.model.eval()

        all_preds = []
        all_targets = []
        all_probs = []
        total_loss = 0.0

        # Progress bar
        if verbose:
            pbar = tqdm(self.val_loader, desc='Validating')
        else:
            pbar = self.val_loader

        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            if self.use_amp:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = self.model(inputs)
                    if criterion:
                        loss = criterion(outputs, targets)
            else:
                outputs = self.model(inputs)
                if criterion:
                    loss = criterion(outputs, targets)

            # Get predictions
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            # Store results
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            if criterion:
                total_loss += loss.item() * inputs.size(0)

            if verbose and isinstance(pbar, tqdm):
                acc = (predicted == targets).float().mean().item()
                pbar.set_postfix({'acc': f'{acc:.4f}'})

        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)

        # Calculate metrics
        metrics = self._calculate_metrics(all_targets, all_preds, all_probs)

        # Add loss if calculated
        if criterion:
            metrics['loss'] = total_loss / len(all_targets)

        # Add confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        metrics['confusion_matrix'] = cm.tolist()

        # Calculate specificity and sensitivity
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        result: Dict = {'metrics': metrics}

        if return_predictions:
            result['predictions'] = {
                'y_true': all_targets,
                'y_pred': all_preds,
                'y_prob': all_probs
            }

        return result

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict:
        """Calculate comprehensive metrics"""
        metrics: Dict = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average='binary', zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average='binary', zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, average='binary', zero_division=0)),
        }

        # Macro and weighted metrics
        metrics['precision_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['recall_macro'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))

        metrics['precision_weighted'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
        metrics['recall_weighted'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
        metrics['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))

        # AUC
        try:
            if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
                metrics['auc'] = float(roc_auc_score(y_true, y_prob[:, 1]))
            else:
                metrics['auc'] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics['auc'] = 0.0

        return metrics

    def validate_with_thresholds(
        self,
        thresholds: Optional[List[float]] = None
    ) -> Dict[float, Dict]:
        """
        Validate with different classification thresholds

        Args:
            thresholds: List of thresholds to try

        Returns:
            Dictionary mapping thresholds to metrics
        """
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

        # Get predictions once
        result = self.validate(return_predictions=True, verbose=False)
        y_true = result['predictions']['y_true']
        y_prob = result['predictions']['y_prob']

        threshold_results = {}

        for threshold in thresholds:
            # Apply threshold
            y_pred = (y_prob[:, 1] >= threshold).astype(int)

            # Calculate metrics
            metrics = self._calculate_metrics(y_true, y_pred, y_prob)
            threshold_results[threshold] = metrics

        return threshold_results

    def find_best_threshold(
        self,
        metric: str = 'f1',
        thresholds: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """
        Find optimal classification threshold

        Args:
            metric: Metric to optimize
            thresholds: Array of thresholds to try

        Returns:
            (best_threshold, best_metric_value)
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.05)

        # Get predictions
        result = self.validate(return_predictions=True, verbose=False)
        y_true = result['predictions']['y_true']
        y_prob = result['predictions']['y_prob']

        best_threshold = 0.5
        best_score = 0.0

        for threshold in thresholds:
            y_pred = (y_prob[:, 1] >= threshold).astype(int)
            metrics = self._calculate_metrics(y_true, y_pred, y_prob)

            score = metrics.get(metric, 0.0)
            if score > best_score:
                best_score = score
                best_threshold = threshold

        return float(best_threshold), float(best_score)

    def validate_per_class(self) -> Dict:
        """Get per-class performance metrics"""
        result = self.validate(return_predictions=True, verbose=False)

        y_true = result['predictions']['y_true']
        y_pred = result['predictions']['y_pred']

        # Calculate per-class metrics
        from sklearn.metrics import precision_recall_fscore_support

        precision_arr, recall_arr, f1_arr, support_arr = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        # Ensure arrays are numpy arrays for consistent indexing
        precision_arr = np.atleast_1d(np.array(precision_arr)) if precision_arr is not None else np.array([])
        recall_arr = np.atleast_1d(np.array(recall_arr)) if recall_arr is not None else np.array([])
        f1_arr = np.atleast_1d(np.array(f1_arr)) if f1_arr is not None else np.array([])
        support_arr = np.atleast_1d(np.array(support_arr)) if support_arr is not None else np.array([])

        per_class = {
            'cover': {
                'precision': float(precision_arr[0]),
                'recall': float(recall_arr[0]),
                'f1': float(f1_arr[0]),
                'support': int(support_arr[0])
            },
            'stego': {
                'precision': float(precision_arr[1]) if len(precision_arr) > 1 else 0.0,
                'recall': float(recall_arr[1]) if len(recall_arr) > 1 else 0.0,
                'f1': float(f1_arr[1]) if len(f1_arr) > 1 else 0.0,
                'support': int(support_arr[1]) if len(support_arr) > 1 else 0
            }
        }

        return per_class

    def save_validation_results(
        self,
        save_path: str,
        include_predictions: bool = False
    ):
        """Save validation results to file"""
        result = self.validate(
            return_predictions=include_predictions,
            verbose=True
        )

        path_obj = Path(save_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data for JSON
        save_data = {'metrics': result['metrics']}

        if include_predictions:
            save_data['predictions'] = {
                'y_true': result['predictions']['y_true'].tolist(),
                'y_pred': result['predictions']['y_pred'].tolist(),
                'y_prob': result['predictions']['y_prob'].tolist()
            }

        with open(path_obj, 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"\nValidation results saved to {save_path}")

    def generate_report(self) -> str:
        """Generate a detailed validation report"""
        result = self.validate(return_predictions=True, verbose=False)
        metrics = result['metrics']

        report = []
        report.append("="*60)
        report.append("VALIDATION REPORT")
        report.append("="*60)
        report.append("")

        # Overall metrics
        report.append("Overall Metrics:")
        report.append(f"  Accuracy:  {metrics['accuracy']:.4f}")
        report.append(f"  Precision: {metrics['precision']:.4f}")
        report.append(f"  Recall:    {metrics['recall']:.4f}")
        report.append(f"  F1 Score:  {metrics['f1']:.4f}")
        if 'auc' in metrics:
            report.append(f"  AUC-ROC:   {metrics['auc']:.4f}")
        report.append("")

        # Confusion matrix
        cm = np.array(metrics['confusion_matrix'])
        report.append("Confusion Matrix:")
        report.append(f"                Predicted")
        report.append(f"              Cover  Stego")
        report.append(f"  Actual Cover  {cm[0,0]:4d}  {cm[0,1]:4d}")
        report.append(f"        Stego  {cm[1,0]:4d}  {cm[1,1]:4d}")
        report.append("")

        # Sensitivity and Specificity
        if 'sensitivity' in metrics:
            report.append(f"  Sensitivity (TPR): {metrics['sensitivity']:.4f}")
        if 'specificity' in metrics:
            report.append(f"  Specificity (TNR): {metrics['specificity']:.4f}")
        report.append("")

        # Per-class metrics
        per_class = self.validate_per_class()
        report.append("Per-Class Metrics:")
        report.append("  Cover Images:")
        report.append(f"    Precision: {per_class['cover']['precision']:.4f}")
        report.append(f"    Recall:    {per_class['cover']['recall']:.4f}")
        report.append(f"    F1 Score:  {per_class['cover']['f1']:.4f}")
        report.append(f"    Support:   {per_class['cover']['support']}")
        report.append("")
        report.append("  Stego Images:")
        report.append(f"    Precision: {per_class['stego']['precision']:.4f}")
        report.append(f"    Recall:    {per_class['stego']['recall']:.4f}")
        report.append(f"    F1 Score:  {per_class['stego']['f1']:.4f}")
        report.append(f"    Support:   {per_class['stego']['support']}")
        report.append("")

        # Classification report (FIXED)
        y_true = result['predictions']['y_true']
        y_pred = result['predictions']['y_pred']
        class_report = classification_report(
            y_true, y_pred,
            target_names=['Cover', 'Stego'],
            zero_division=0
        )
        # Handle both string and dict return types
        if isinstance(class_report, dict):
            import json
            class_report_str = json.dumps(class_report, indent=2)
        else:
            class_report_str = str(class_report)

        report.append("Detailed Classification Report:")
        report.append(class_report_str)
        report.append("="*60)

        return "\n".join(report)


class ModelComparator:
    """Compare multiple models on the same validation set"""

    def __init__(
        self,
        models: Dict[str, nn.Module],
        val_loader: DataLoader,
        device: str = 'cuda'
    ):
        self.models = models
        self.val_loader = val_loader
        self.device = device

    def compare_models(self, criterion: Optional[nn.Module] = None) -> Dict:
        """Compare all models"""
        results = {}

        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            validator = StegValidator(model, self.val_loader, self.device)
            result = validator.validate(criterion=criterion, verbose=True)
            results[name] = result['metrics']

        return results

    def print_comparison(self, results: Dict):
        """Print comparison table"""
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)

        # Header
        print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10}")
        print("-"*80)

        # Each model
        for name, metrics in results.items():
            print(f"{name:<20} "
                  f"{metrics['accuracy']:<10.4f} "
                  f"{metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<10.4f} "
                  f"{metrics['f1']:<10.4f} "
                  f"{metrics.get('auc', 0):<10.4f}")

        print("="*80)
