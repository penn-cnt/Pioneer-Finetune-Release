"""
data_utils.py

Utilities for organizing and loading seizure detection data.
Updated for HDF5 structure: patient_name/onset_time/data and labels
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def concat_collate(batch):
    # batch is a list of tensors, each of shape (17, 1, 256)
    return torch.cat(batch, dim=0)  # (B*17, 1, 256)

class CustomDataset(Dataset):
    def __init__(self, data_df, w_size, w_stride, fs=256):
        self.data_df = data_df
        self.w_size = w_size
        self.w_stride = w_stride
        self.fs = fs
        self.win_starts = self.get_win_times()
        self.window_sample = int(w_size * fs)
        self.window_stride = int(w_stride * fs)
    
    def get_win_times(self):
        data_len = len(self.data_df) / self.fs
        n_windows = np.floor((data_len - self.w_size)/self.w_stride) + 1
        return np.arange(n_windows) * self.w_stride

    def __len__(self):
        return len(self.win_starts)
    
    def __getitem__(self, idx):
        data = self.data_df.iloc[idx*self.window_stride:idx*self.window_stride+self.window_sample,:].values.astype('float32')
        data = torch.from_numpy(data).T.unsqueeze(1)  # [channels, 1, samples]
        return data
    
class SeizureDataset(Dataset):
    """
    PyTorch Dataset for seizure detection.
    Loads data from HDF5 file with structure:
        patient_name/onset_time/data [n_clips, 256]
        patient_name/onset_time/labels [n_clips]
    """
    
    def __init__(
        self,
        h5_path: str,
        patient_ids: Optional[List[str]] = None,
        combine_onset_spread: bool = True,
        return_original_label: bool = False,
        cache_labels: bool = False,
        normalize_per_sample: bool = False,
        transform=None
    ):
        """
        Args:
            h5_path: Path to HDF5 file
            patient_ids: List of patient IDs to include (None = all)
            combine_onset_spread: If True, combine labels 1,2 into single "seizing" class
            return_original_label: If True, include original 3-class label in metadata
            cache_labels: If True, preload all labels into memory for faster access
            normalize_per_sample: If True, z-score normalize each sample independently
            transform: Optional transform to apply
        """
        self.h5_path = Path(h5_path)
        self.combine_onset_spread = combine_onset_spread
        self.return_original_label = return_original_label
        self.normalize_per_sample = normalize_per_sample
        self.transform = transform
        
        # Build index of all samples
        self.sample_index = self._build_index(patient_ids)
        
        # Cache labels in memory if requested
        self._label_cache = None
        self._original_label_cache = None
        if cache_labels:
            self._preload_labels()
        
        print(f"Dataset created with {len(self.sample_index)} samples")
        print(f"Class distribution: {self.get_class_distribution()}")
        
    def _build_index(self, patient_ids: Optional[List[str]]) -> List[Dict]:
        """
        Build index of all samples for efficient access.
        
        Returns:
            List of dicts with keys: patient_id, onset_time, clip_idx
        """
        index = []
        
        with h5py.File(self.h5_path, 'r') as f:
            # Get patients to include
            patients = patient_ids if patient_ids else list(f.keys())
            
            for patient_id in patients:
                if patient_id not in f:
                    print(f"Warning: Patient {patient_id} not found in dataset")
                    continue
                
                patient_group = f[patient_id]
                
                # Iterate through seizure onset times
                for onset_time in patient_group.keys():
                    onset_group = patient_group[onset_time]
                    
                    # Get number of clips for this seizure
                    n_clips = len(onset_group['labels'])
                    
                    # Add each clip to index
                    for clip_idx in range(n_clips):
                        index.append({
                            'patient_id': patient_id,
                            'onset_time': onset_time,
                            'clip_idx': clip_idx
                        })
        
        return index
    
    def _preload_labels(self):
        """Cache all labels in memory for fast access."""
        print("Preloading labels into memory...")
        self._label_cache = []
        self._original_label_cache = []
        
        with h5py.File(self.h5_path, 'r') as f:
            for sample_info in self.sample_index:
                onset_group = f[sample_info['patient_id']][sample_info['onset_time']]
                original_label = int(onset_group['labels'][sample_info['clip_idx']])
                
                # Apply combine_onset_spread if needed
                if self.combine_onset_spread and original_label > 0:
                    label = 1
                else:
                    label = original_label
                
                self._label_cache.append(label)
                self._original_label_cache.append(original_label)
        
        print(f"Cached {len(self._label_cache)} labels")
    
    def __len__(self) -> int:
        return len(self.sample_index)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        """
        Get a single sample.
        
        Returns:
            signal: [1, 256] torch.Tensor
            label: int (0=not seizing, 1=seizing)
            metadata: dict with patient_id, onset_time, clip_idx, and optionally original_label
        """
        sample_info = self.sample_index[idx]
        
        # Use cached labels if available
        if self._label_cache is not None:
            label = self._label_cache[idx]
            original_label = self._original_label_cache[idx]
            
            # Load only signal from h5
            with h5py.File(self.h5_path, 'r') as f:
                onset_group = f[sample_info['patient_id']][sample_info['onset_time']]
                signal = onset_group['data'][sample_info['clip_idx']]
                signal = torch.FloatTensor(signal).unsqueeze(0)  # [1, 256]
        else:
            # Load both signal and label from h5
            with h5py.File(self.h5_path, 'r') as f:
                onset_group = f[sample_info['patient_id']][sample_info['onset_time']]
                
                # Load signal [256]
                signal = onset_group['data'][sample_info['clip_idx']]
                signal = torch.FloatTensor(signal).unsqueeze(0)  # [1, 256]
                
                # Load label
                original_label = int(onset_group['labels'][sample_info['clip_idx']])
                
                # Combine onset (1) and spread (2) into seizing class (1)
                if self.combine_onset_spread and original_label > 0:
                    label = 1
                else:
                    label = original_label
        
        # Normalize per sample if requested
        if self.normalize_per_sample:
            mean = signal.mean()
            std = signal.std()
            if std > 1e-8:  # Avoid division by zero
                signal = (signal - mean) / std
        
        # Metadata
        metadata = {
            'patient_id': sample_info['patient_id'],
            'onset_time': sample_info['onset_time'],
            'clip_idx': sample_info['clip_idx']
        }
        
        # Optionally include original 3-class label
        if self.return_original_label:
            metadata['original_label'] = original_label
        
        # Apply transform if specified
        if self.transform:
            signal = self.transform(signal)
        
        return signal, label, metadata
    
    def get_patient_ids(self) -> List[str]:
        """Get list of all patient IDs in dataset."""
        patient_ids = [s['patient_id'] for s in self.sample_index]
        return sorted(list(set(patient_ids)))
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get distribution of classes."""
        # Use cached labels if available
        if self._label_cache is not None:
            labels = self._label_cache
        else:
            labels = []
            with h5py.File(self.h5_path, 'r') as f:
                for sample_info in self.sample_index:
                    onset_group = f[sample_info['patient_id']][sample_info['onset_time']]
                    label = int(onset_group['labels'][sample_info['clip_idx']])
                    
                    if self.combine_onset_spread and label > 0:
                        label = 1
                    
                    labels.append(label)
        
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique, counts))
    
    def get_patient_distribution(self) -> Dict[str, Dict[int, int]]:
        """Get class distribution per patient."""
        patient_dist = {}
        
        # Use cached labels if available
        if self._label_cache is not None:
            for patient_id in self.get_patient_ids():
                labels = []
                for idx, sample_info in enumerate(self.sample_index):
                    if sample_info['patient_id'] == patient_id:
                        labels.append(self._label_cache[idx])
                
                unique, counts = np.unique(labels, return_counts=True)
                patient_dist[patient_id] = dict(zip(unique, counts))
        else:
            with h5py.File(self.h5_path, 'r') as f:
                for patient_id in self.get_patient_ids():
                    labels = []
                    
                    for sample_info in self.sample_index:
                        if sample_info['patient_id'] == patient_id:
                            onset_group = f[sample_info['patient_id']][sample_info['onset_time']]
                            label = int(onset_group['labels'][sample_info['clip_idx']])
                            
                            if self.combine_onset_spread and label > 0:
                                label = 1
                            
                            labels.append(label)
                    
                    unique, counts = np.unique(labels, return_counts=True)
                    patient_dist[patient_id] = dict(zip(unique, counts))
        
        return patient_dist


def create_dataloaders(
    h5_path: str,
    train_patient_ids: List[str],
    val_patient_ids: List[str],
    batch_size: int = 128,
    num_workers: int = 4,
    return_original_label: bool = False,
    normalize_per_sample: bool = False,
    transform=None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        h5_path: Path to HDF5 file
        train_patient_ids: List of patient IDs for training
        val_patient_ids: List of patient IDs for validation
        batch_size: Batch size
        num_workers: Number of workers for DataLoader
        return_original_label: If True, include original 3-class labels in metadata
        normalize_per_sample: If True, z-score each sample independently
        transform: Optional transform
        
    Returns:
        train_loader, val_loader
    """
    train_dataset = SeizureDataset(
        h5_path,
        patient_ids=train_patient_ids,
        combine_onset_spread=True,
        return_original_label=return_original_label,
        normalize_per_sample=normalize_per_sample,
        transform=transform
    )
    
    val_dataset = SeizureDataset(
        h5_path,
        patient_ids=val_patient_ids,
        combine_onset_spread=True,
        return_original_label=return_original_label,
        normalize_per_sample=normalize_per_sample,
        transform=None  # No augmentation for validation
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\nData Loaders Created:")
    print(f"  Train: {len(train_dataset)} samples from {len(train_patient_ids)} patients")
    print(f"  Val:   {len(val_dataset)} samples from {len(val_patient_ids)} patients")
    print(f"\nTrain class distribution: {train_dataset.get_class_distribution()}")
    print(f"Val class distribution:   {val_dataset.get_class_distribution()}")
    
    return train_loader, val_loader


def analyze_dataset(h5_path: str):
    """
    Analyze HDF5 dataset structure and content.
    
    Args:
        h5_path: Path to HDF5 file
    """
    print("\n" + "="*60)
    print("DATASET ANALYSIS")
    print("="*60)
    
    with h5py.File(h5_path, 'r') as f:
        # Count patients
        patients = list(f.keys())
        n_patients = len(patients)
        
        print(f"\nTotal patients: {n_patients}")
        
        # Analyze structure
        total_seizures = 0
        total_clips = 0
        clip_lengths = []
        labels_all = []
        
        print("\nPatient breakdown:")
        print(f"{'Patient':<20} {'N Seizures':<12} {'N Clips':<10} {'Class Dist'}")
        print("-" * 60)
        
        for patient_id in patients:
            patient_group = f[patient_id]
            onset_times = list(patient_group.keys())
            n_seizures = len(onset_times)
            total_seizures += n_seizures
            
            patient_clips = 0
            patient_labels = []
            
            for onset_time in onset_times:
                onset_group = patient_group[onset_time]
                n_clips = len(onset_group['labels'])
                patient_clips += n_clips
                total_clips += n_clips
                
                # Check data shape
                data_shape = onset_group['data'].shape
                clip_lengths.append(data_shape[1])
                
                # Collect labels
                labels = onset_group['labels'][:]
                patient_labels.extend(labels)
                labels_all.extend(labels)
            
            # Class distribution for this patient
            unique, counts = np.unique(patient_labels, return_counts=True)
            class_dist = {int(k): int(v) for k, v in zip(unique, counts)}
            
            print(f"{patient_id:<20} {n_seizures:<12} {patient_clips:<10} {class_dist}")
        
        print("-" * 60)
        print(f"{'TOTAL':<20} {total_seizures:<12} {total_clips:<10}")
        
        # Overall statistics
        print(f"\n{'='*60}")
        print("OVERALL STATISTICS")
        print(f"{'='*60}")
        print(f"Total patients:  {n_patients}")
        print(f"Total seizures:  {total_seizures}")
        print(f"Total clips:     {total_clips}")
        print(f"Clips per patient: {total_clips/n_patients:.1f} ± {np.std([len([s for s in f[p].keys()]) for p in patients]):.1f}")
        
        # Clip lengths
        print(f"\nClip lengths:")
        print(f"  Min:  {min(clip_lengths)}")
        print(f"  Max:  {max(clip_lengths)}")
        print(f"  Mean: {np.mean(clip_lengths):.1f}")
        
        # Class distribution
        unique, counts = np.unique(labels_all, return_counts=True)
        print(f"\nClass distribution (overall):")
        for label, count in zip(unique, counts):
            percentage = count / len(labels_all) * 100
            label_name = {0: "Not seizing", 1: "Onset", 2: "Spread"}.get(label, f"Class {label}")
            print(f"  {label_name:<15} ({label}): {count:6d} ({percentage:5.1f}%)")
        
        # After combining onset and spread
        labels_combined = np.array([1 if l > 0 else 0 for l in labels_all])
        unique, counts = np.unique(labels_combined, return_counts=True)
        print(f"\nClass distribution (after combining onset+spread):")
        for label, count in zip(unique, counts):
            percentage = count / len(labels_combined) * 100
            label_name = {0: "Not seizing", 1: "Seizing"}.get(label, f"Class {label}")
            print(f"  {label_name:<15} ({label}): {count:6d} ({percentage:5.1f}%)")
        
        # Imbalance ratio
        if len(unique) == 2:
            imbalance_ratio = counts[0] / counts[1]
            print(f"\nImbalance ratio: {imbalance_ratio:.2f}:1 (not seizing : seizing)")


def get_patient_groups(h5_path: str) -> Dict[str, List[str]]:
    """
    Get mapping of patients to their seizure onset times.
    
    Args:
        h5_path: Path to HDF5 file
        
    Returns:
        Dictionary mapping patient_id to list of onset_times
    """
    patient_groups = {}
    
    with h5py.File(h5_path, 'r') as f:
        for patient_id in f.keys():
            onset_times = list(f[patient_id].keys())
            patient_groups[patient_id] = onset_times
    
    return patient_groups


def verify_data_integrity(h5_path: str):
    """
    Verify data integrity of HDF5 file.
    
    Args:
        h5_path: Path to HDF5 file
    """
    print("\n" + "="*60)
    print("DATA INTEGRITY CHECK")
    print("="*60)
    
    issues = []
    
    with h5py.File(h5_path, 'r') as f:
        for patient_id in f.keys():
            patient_group = f[patient_id]
            
            for onset_time in patient_group.keys():
                onset_group = patient_group[onset_time]
                
                # Check required datasets exist
                if 'data' not in onset_group:
                    issues.append(f"{patient_id}/{onset_time}: Missing 'data' dataset")
                    continue
                
                if 'labels' not in onset_group:
                    issues.append(f"{patient_id}/{onset_time}: Missing 'labels' dataset")
                    continue
                
                # Check shapes match
                data_shape = onset_group['data'].shape
                labels_shape = onset_group['labels'].shape
                
                if data_shape[0] != labels_shape[0]:
                    issues.append(
                        f"{patient_id}/{onset_time}: Shape mismatch - "
                        f"data: {data_shape}, labels: {labels_shape}"
                    )
                
                # Check data is 256 samples
                if data_shape[1] != 256:
                    issues.append(
                        f"{patient_id}/{onset_time}: Expected 256 samples, got {data_shape[1]}"
                    )
                
                # Check labels are valid (0, 1, or 2)
                labels = onset_group['labels'][:]
                unique_labels = np.unique(labels)
                invalid_labels = [l for l in unique_labels if l not in [0, 1, 2]]
                
                if invalid_labels:
                    issues.append(
                        f"{patient_id}/{onset_time}: Invalid labels found: {invalid_labels}"
                    )
                
                # Check for NaN or Inf in data
                data_sample = onset_group['data'][0]
                if np.any(np.isnan(data_sample)) or np.any(np.isinf(data_sample)):
                    issues.append(
                        f"{patient_id}/{onset_time}: NaN or Inf values detected in data"
                    )
    
    if issues:
        print(f"\n❌ Found {len(issues)} issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ All checks passed! Data integrity verified.")
    
    return len(issues) == 0


if __name__ == "__main__":
    # Test data loading
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python data_utils.py <path_to_h5_file>")
        sys.exit(1)
    
    h5_path = sys.argv[1]
    
    print(f"Loading dataset from: {h5_path}")
    
    # Verify integrity
    verify_data_integrity(h5_path)
    
    # Analyze dataset
    analyze_dataset(h5_path)
    
    # Test dataset creation
    print("\n" + "="*60)
    print("TESTING DATASET LOADING")
    print("="*60)
    
    dataset = SeizureDataset(h5_path, combine_onset_spread=True)
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Patients: {dataset.get_patient_ids()}")
    
    # Test sample loading
    print("\nTesting sample loading...")
    signal, label, metadata = dataset[0]
    print(f"  Signal shape: {signal.shape}")
    print(f"  Signal dtype: {signal.dtype}")
    print(f"  Signal range: [{signal.min():.3f}, {signal.max():.3f}]")
    print(f"  Label: {label}")
    print(f"  Metadata: {metadata}")
    
    # Test patient distribution
    print("\nPer-patient class distribution:")
    patient_dist = dataset.get_patient_distribution()
    for patient_id, dist in patient_dist.items():
        total = sum(dist.values())
        print(f"  {patient_id}: {dist} (total: {total})")
    
    print("\n✅ Dataset loading test complete!")