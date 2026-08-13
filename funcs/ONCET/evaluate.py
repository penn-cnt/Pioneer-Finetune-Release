"""
evaluate.py

Evaluate trained model on held-out test set.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

from model import LightweightSeizureDetector, count_parameters
from data_utils import SeizureDataset, create_dataloaders
from metrics import (
    compute_comprehensive_metrics,
    print_metrics,
    plot_roc_pr_curves,
    compute_optimal_threshold
)


class ModelEvaluator:
    """Evaluate trained model on test set."""
    
    def __init__(
        self,
        model: nn.Module,
        test_loader,
        device: str = 'cuda'
    ):
        """
        Args:
            model: Trained model
            test_loader: Test data loader
            device: Device to use
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        
    def predict(self) -> tuple:
        """
        Get predictions on test set.
        
        Returns:
            labels: True labels [N]
            probs: Predicted probabilities [N]
            metadata: List of metadata dicts
        """
        self.model.eval()
        
        all_labels = []
        all_probs = []
        all_metadata = []
        
        with torch.no_grad():
            for signals, labels, metadata in self.test_loader:
                signals = signals.to(self.device)
                
                logits = self.model(signals)
                probs = torch.softmax(logits, dim=1)[:, 1]
                
                all_labels.append(labels.numpy())
                all_probs.append(probs.cpu().numpy())
                all_metadata.extend(metadata)
        
        labels = np.concatenate(all_labels)
        probs = np.concatenate(all_probs)
        
        return labels, probs, all_metadata
    
    def evaluate(
        self,
        threshold: float = 0.5,
        find_optimal_threshold: bool = True,
        save_dir: str = 'results/test_evaluation'
    ) -> Dict:
        """
        Comprehensive evaluation on test set.
        
        Args:
            threshold: Classification threshold (if not finding optimal)
            find_optimal_threshold: Whether to find optimal threshold
            save_dir: Directory to save results
            
        Returns:
            Dictionary of metrics
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*60)
        print("TEST SET EVALUATION")
        print("="*60)
        
        # Get predictions
        print("\nGenerating predictions...")
        labels, probs, metadata = self.predict()
        
        print(f"Test set size: {len(labels)}")
        print(f"Class distribution: {np.bincount(labels)}")
        print(f"  Not seizing: {(labels == 0).sum()} ({(labels == 0).mean()*100:.1f}%)")
        print(f"  Seizing:     {(labels == 1).sum()} ({(labels == 1).mean()*100:.1f}%)")
        
        # Find optimal threshold if requested
        if find_optimal_threshold:
            print("\nFinding optimal threshold...")
            
            # Try multiple metrics
            thresh_f1, f1 = compute_optimal_threshold(labels, probs, metric='f1')
            thresh_f2, f2 = compute_optimal_threshold(labels, probs, metric='f2')
            thresh_youden, youden = compute_optimal_threshold(labels, probs, metric='youden')
            
            print(f"  F1 optimal:     threshold={thresh_f1:.3f}, F1={f1:.3f}")
            print(f"  F2 optimal:     threshold={thresh_f2:.3f}, F2={f2:.3f}")
            print(f"  Youden optimal: threshold={thresh_youden:.3f}, Youden={youden:.3f}")
            
            # Use F2 for clinical scenario (prefer sensitivity)
            threshold = thresh_f2
            print(f"\nUsing F2-optimal threshold: {threshold:.3f}")
        
        # Compute comprehensive metrics
        metrics = compute_comprehensive_metrics(
            labels,
            probs,
            threshold=threshold,
            prefix='test_'
        )
        
        # Print metrics
        print_metrics(metrics, "Test Set Metrics")
        
        # Plot ROC and PR curves
        print("\nGenerating curves...")
        plot_roc_pr_curves(
            labels,
            probs,
            save_path=save_dir / 'roc_pr_curves.png'
        )
        
        # Analyze by patient
        print("\nAnalyzing per-patient performance...")
        patient_metrics = self._analyze_by_patient(labels, probs, metadata, threshold)
        
        # Save patient-level analysis
        self._save_patient_analysis(patient_metrics, save_dir)
        
        # Analyze by electrode location (if available)
        print("\nAnalyzing by electrode location...")
        location_metrics = self._analyze_by_location(labels, probs, metadata, threshold)
        
        # Save all results
        results = {
            'overall_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                              for k, v in metrics.items()},
            'threshold': float(threshold),
            'patient_metrics': patient_metrics,
            'location_metrics': location_metrics
        }
        
        with open(save_dir / 'test_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nResults saved to {save_dir}/")
        
        return metrics
    
    def _analyze_by_patient(
        self,
        labels: np.ndarray,
        probs: np.ndarray,
        metadata: List[Dict],
        threshold: float
    ) -> Dict:
        """Analyze performance per patient."""
        # Group by patient
        patient_results = {}
        
        for i, meta in enumerate(metadata):
            patient_id = meta['patient_id']
            
            if patient_id not in patient_results:
                patient_results[patient_id] = {
                    'labels': [],
                    'probs': []
                }
            
            patient_results[patient_id]['labels'].append(labels[i])
            patient_results[patient_id]['probs'].append(probs[i])
        
        # Compute metrics per patient
        patient_metrics = {}
        
        for patient_id, data in patient_results.items():
            y_true = np.array(data['labels'])
            y_probs = np.array(data['probs'])
            
            if len(np.unique(y_true)) < 2:
                # Skip patients with only one class
                continue
            
            metrics = compute_comprehensive_metrics(
                y_true,
                y_probs,
                threshold=threshold,
                prefix=''
            )
            
            patient_metrics[patient_id] = {
                k: float(v) if isinstance(v, (int, float, np.number)) else v
                for k, v in metrics.items()
            }
        
        print(f"  Evaluated {len(patient_metrics)} patients")
        
        # Summary statistics
        auprcs = [m['auprc_normalized'] for m in patient_metrics.values()]
        print(f"  Mean AUPRC (norm): {np.mean(auprcs):.3f} ± {np.std(auprcs):.3f}")
        
        return patient_metrics
    
    def _analyze_by_location(
        self,
        labels: np.ndarray,
        probs: np.ndarray,
        metadata: List[Dict],
        threshold: float
    ) -> Dict:
        """Analyze performance by electrode location."""
        # Extract brain region from channel name (if available)
        # This is dataset-specific - adapt to your naming convention
        
        location_results = {}
        
        for i, meta in enumerate(metadata):
            channel = meta['channel_name']
            
            # Extract location (example: "LA1" -> "LA", "RH3" -> "RH")
            if len(channel) >= 2:
                location = channel[:2]
            else:
                location = 'unknown'
            
            if location not in location_results:
                location_results[location] = {
                    'labels': [],
                    'probs': []
                }
            
            location_results[location]['labels'].append(labels[i])
            location_results[location]['probs'].append(probs[i])
        
        # Compute metrics per location
        location_metrics = {}
        
        for location, data in location_results.items():
            y_true = np.array(data['labels'])
            y_probs = np.array(data['probs'])
            
            if len(np.unique(y_true)) < 2 or len(y_true) < 10:
                # Skip locations with too few samples or only one class
                continue
            
            metrics = compute_comprehensive_metrics(
                y_true,
                y_probs,
                threshold=threshold,
                prefix=''
            )
            
            location_metrics[location] = {
                k: float(v) if isinstance(v, (int, float, np.number)) else v
                for k, v in metrics.items()
                if k not in ['true_positives', 'false_positives', 'true_negatives', 'false_negatives']
            }
            
            location_metrics[location]['n_samples'] = len(y_true)
        
        print(f"  Evaluated {len(location_metrics)} locations")
        
        return location_metrics
    
    def _save_patient_analysis(self, patient_metrics: Dict, save_dir: Path):
        """Save patient-level analysis with visualization."""
        # Extract metrics for plotting
        patient_ids = list(patient_metrics.keys())
        auprcs = [patient_metrics[p]['auprc_normalized'] for p in patient_ids]
        sensitivities = [patient_metrics[p]['sensitivity'] for p in patient_ids]
        specificities = [patient_metrics[p]['specificity'] for p in patient_ids]
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # AUPRC distribution
        axes[0, 0].hist(auprcs, bins=20, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(np.mean(auprcs), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(auprcs):.3f}')
        axes[0, 0].set_xlabel('Normalized AUPRC')
        axes[0, 0].set_ylabel('Number of Patients')
        axes[0, 0].set_title('Per-Patient AUPRC Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Sensitivity vs Specificity
        axes[0, 1].scatter(specificities, sensitivities, alpha=0.6, s=50)
        axes[0, 1].axhline(np.mean(sensitivities), color='red', linestyle='--', alpha=0.5)
        axes[0, 1].axvline(np.mean(specificities), color='red', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('Specificity')
        axes[0, 1].set_ylabel('Sensitivity')
        axes[0, 1].set_title('Sensitivity vs Specificity per Patient')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Sample size vs AUPRC
        sample_sizes = [patient_metrics[p]['n_samples'] for p in patient_ids]
        axes[1, 0].scatter(sample_sizes, auprcs, alpha=0.6, s=50)
        axes[1, 0].set_xlabel('Number of Samples')
        axes[1, 0].set_ylabel('Normalized AUPRC')
        axes[1, 0].set_title('AUPRC vs Sample Size')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Summary statistics table
        axes[1, 1].axis('off')
        summary_text = f"""
        Per-Patient Performance Summary
        ═══════════════════════════════
        
        Number of patients:     {len(patient_ids)}
        
        AUPRC (normalized):
          Mean:  {np.mean(auprcs):.3f}
          Std:   {np.std(auprcs):.3f}
          Min:   {np.min(auprcs):.3f}
          Max:   {np.max(auprcs):.3f}
        
        Sensitivity:
          Mean:  {np.mean(sensitivities):.3f}
          Std:   {np.std(sensitivities):.3f}
        
        Specificity:
          Mean:  {np.mean(specificities):.3f}
          Std:   {np.std(specificities):.3f}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, 
                       verticalalignment='center', family='monospace')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'patient_analysis.png', dpi=150, bbox_inches='tight')
        print(f"  Patient analysis plot saved")
        plt.close()


def evaluate_model_from_checkpoint(
    checkpoint_path: str,
    h5_path: str,
    test_patient_ids: List[str],
    config: Dict,
    save_dir: str = 'results/test_evaluation'
):
    """
    Load model from checkpoint and evaluate on test set.
    
    Args:
        checkpoint_path: Path to model checkpoint
        h5_path: Path to HDF5 data file
        test_patient_ids: List of test patient IDs
        config: Configuration dict (must match training config)
        save_dir: Directory to save results
    """
    print("Loading model from checkpoint...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
    
    # Create model with same architecture
    model = LightweightSeizureDetector(
        num_classes=2,
        base_filters=config['base_filters'],
        num_blocks=config['num_blocks'],
        dilations=config.get('dilations', [2, 4, 8]),
        dropout_blocks=config.get('dropout_blocks', 0.2),
        dropout_head=config.get('dropout_head', 0.3)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Create test data loader
    print("\nCreating test data loader...")
    normalize_per_sample = config.get('normalize_per_sample', False)
    test_dataset = SeizureDataset(
        h5_path,
        patient_ids=test_patient_ids,
        combine_onset_spread=True,
        normalize_per_sample=normalize_per_sample
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Test set: {len(test_dataset)} samples from {len(test_patient_ids)} patients")
    print(f"Class distribution: {test_dataset.get_class_distribution()}")
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        test_loader=test_loader,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Evaluate
    metrics = evaluator.evaluate(
        find_optimal_threshold=True,
        save_dir=save_dir
    )
    
    return metrics


if __name__ == "__main__":
    # Example: Load best model from cross-validation and evaluate on test set
    
    # Configuration (should match training)
    config = {
        'base_filters': 32,
        'num_blocks': 3,
        'dilations': [2, 4, 8],
        'dropout_blocks': 0.2,
        'dropout_head': 0.3
    }
    
    # Test patient IDs (these should be held out from training/validation)
    test_patient_ids = [
        'patient_091', 'patient_092', 'patient_093', 'patient_094', 'patient_095',
        'patient_096', 'patient_097', 'patient_098', 'patient_099', 'patient_100'
    ]
    
    # Evaluate
    metrics = evaluate_model_from_checkpoint(
        checkpoint_path='checkpoints/best_model.pth',
        h5_path='seizure_data.h5',
        test_patient_ids=test_patient_ids,
        config=config,
        save_dir='results/final_test_evaluation'
    )