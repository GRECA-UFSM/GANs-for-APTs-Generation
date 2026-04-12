import argparse
import os
import pathlib
from typing import Callable

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torchmetrics import Accuracy, F1Score, MetricCollection

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from experiment import Experiment

class FlowDataset(Dataset):
    def __init__(self, features: pd.DataFrame, labels: pd.Series, sequence_length: int, sequence_stride: int, cat_cols: list[str], cont_cols: list[str], transform: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None):
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride

        self.cat_features = torch.tensor(features[cat_cols].values, dtype=torch.long)
        self.cont_features = torch.tensor(features[cont_cols].values, dtype=torch.float)
        
        self.labels = torch.tensor(labels.values, dtype=torch.long)
        
        self.transform = transform

    def __len__(self) -> int:
        if len(self.labels) < self.sequence_length:
            return 0
        
        return (len(self.labels) - self.sequence_length) // self.sequence_stride + 1

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start_idx = idx * self.sequence_stride
        end_idx = start_idx + self.sequence_length
        
        x_cat = self.cat_features[start_idx:end_idx]
        x_cont = self.cont_features[start_idx:end_idx]
        
        labels = self.labels[start_idx:end_idx]
        
        if self.transform is not None:
            x_cat, x_cont = self.transform(x_cat, x_cont, labels)
        
        y = self.labels[start_idx:end_idx].max()
        
        return x_cat, x_cont, y

def read_flows(path: pathlib.Path) -> pd.DataFrame:
    flows = pd.DataFrame()
    
    for file in path.glob('*.csv'):
        flows = pd.concat([flows, pd.read_csv(file)])
    
    return flows

def prepare_flow_features(flows: pd.DataFrame) -> pd.DataFrame:
    flows['src2dst_packet_rate'] = flows['src2dst_packets'] / (flows['src2dst_duration_ms'] / 1000 + 1e-6)
    flows['dst2src_packet_rate'] = flows['dst2src_packets'] / (flows['dst2src_duration_ms'] / 1000 + 1e-6)
    flows['src2dst_byte_rate'] = flows['src2dst_bytes'] / (flows['src2dst_duration_ms'] / 1000 + 1e-6)
    flows['dst2src_byte_rate'] = flows['dst2src_bytes'] / (flows['dst2src_duration_ms'] / 1000 + 1e-6)
    
    is_ip_private = lambda ip: ip.startswith('192.168.') or ip.startswith('10.') or ip.startswith('172.16.')
    
    flows['src_private'] = flows['src_ip'].apply(is_ip_private)
    flows['dst_private'] = flows['dst_ip'].apply(is_ip_private)

    # flows.loc[flows['activity'] == 'Unknown', 'activity'] = 'Normal'
    # flows.loc[flows['stage'] == 'Unknown', 'stage'] = 'Benign'
    
    flows = flows[(flows['activity'] != 'Unknown') & (flows['stage'] != 'Unknown')].reset_index(drop=True)

    flows.sort_values(by='bidirectional_first_seen_ms', kind='mergesort', inplace=True, ignore_index=True)
    
    return flows

def encode_flow_stages(flows: pd.DataFrame, stages: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    labels = flows['stage'].apply(lambda x: stages.index(x) if x in stages else -1)
    flows = flows[labels != -1].reset_index(drop=True)
    labels = labels[labels != -1].reset_index(drop=True)
    
    return flows, labels
    
def split_flows(features: pd.DataFrame, labels: pd.Series, sequence_length: int, test_size: float, random_seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    sequence_count = len(features) // sequence_length
    sequence_labels = []
    
    for i in range(sequence_count):
        start_idx = i * sequence_length
        end_idx = start_idx + sequence_length
        
        sequence_labels.append(labels.iloc[start_idx:end_idx].max())
    
    _, val_indices = train_test_split(np.arange(sequence_count), test_size=test_size, random_state=random_seed, stratify=sequence_labels)
    
    val_mask = np.isin(np.arange(sequence_count), val_indices)
    val_mask = np.repeat(val_mask, sequence_length)
    val_mask = np.pad(val_mask, (0, len(features) - len(val_mask)), constant_values=False)
    
    val_flows = features[val_mask].reset_index(drop=True)
    val_labels = labels[val_mask].reset_index(drop=True)
    
    train_flows = features[~val_mask].reset_index(drop=True)
    train_labels = labels[~val_mask].reset_index(drop=True)
    
    return train_flows, val_flows, train_labels, val_labels

def create_datasets(train_flows: pd.DataFrame, val_flows: pd.DataFrame, train_labels: pd.Series, val_labels: pd.Series, sequence_length: int, cat_cols: list[str], cont_cols: list[str]) -> tuple[FlowDataset, FlowDataset]:
    def augment(x_cat: torch.Tensor, x_cont: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Performs a group-wise shuffle with stable order inside each label
        sequence_length = len(y)
        
        perm = torch.randperm(sequence_length)
        
        y_shuffled = y[perm]
        x_cat_shuffled = torch.empty_like(x_cat)
        x_cont_shuffled = torch.empty_like(x_cont)
        
        for label in y.unique():
            label_mask = (y == label)
            shuffled_label_mask = (y_shuffled == label)
            
            x_cat_shuffled[shuffled_label_mask] = x_cat[label_mask]
            x_cont_shuffled[shuffled_label_mask] = x_cont[label_mask]
        
        return x_cat_shuffled, x_cont_shuffled
    
    train_ds = FlowDataset(train_flows, train_labels, sequence_length=sequence_length, sequence_stride=1, cat_cols=cat_cols, cont_cols=cont_cols, transform=augment)
    val_ds = FlowDataset(val_flows, val_labels, sequence_length=sequence_length, sequence_stride=sequence_length, cat_cols=cat_cols, cont_cols=cont_cols)
    
    return train_ds, val_ds

def create_dataloaders(train_ds: FlowDataset, val_ds: FlowDataset, batch_size: int) -> tuple[DataLoader, DataLoader]:
    train_labels  = [train_ds[i][2] for i in range(len(train_ds))]

    train_class_counts = np.bincount(train_labels)
    train_label_weights = 1. / train_class_counts   
    train_label_weights = train_label_weights[train_labels]

    train_sampler = WeightedRandomSampler(train_label_weights, num_samples=len(train_label_weights), replacement=True) # type: ignore
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, val_loader

def train_model(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, metrics: MetricCollection | None = None) -> float:
    model.train()
    
    total_loss = 0.0
    
    for x_cat, x_cont, y in train_loader:
        x_cat = x_cat.to(device)
        x_cont = x_cont.to(device)
        y = y.to(device)
        
        logits = model(x_cat, x_cont)
        loss = criterion(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        
        total_loss += loss.item()
    
        if metrics is not None:
            metrics.update(logits, y)
    
    avg_loss = total_loss / len(train_loader)
    
    return avg_loss

def evaluate_model(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device, metrics: MetricCollection | None = None) -> tuple[float, list[int], list[int]]:
    model.eval()
    
    total_loss = 0.0
    
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for x_cat, x_cont, y in val_loader:
            x_cat = x_cat.to(device)
            x_cont = x_cont.to(device)
            y = y.to(device)
            
            logits = model(x_cat, x_cont)
            loss = criterion(logits, y)
            
            total_loss += loss.item()
            
            all_preds.extend(logits.max(1)[1].cpu().numpy())
            all_true.extend(y.cpu().numpy())
            
            if metrics is not None:
                metrics.update(logits, y)
    
    avg_loss = total_loss / len(val_loader)
    
    return avg_loss, all_preds, all_true

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, default='../../datasets/cherry-picked/dapt2020-nfstream')
    parser.add_argument('--output', '-o', type=str, default='out')
    parser.add_argument('--batch-size', '-b', type=int, default=64)
    parser.add_argument('--epochs', '-e', type=int, default=5)
    parser.add_argument('--sequence-length', '-l', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    stages = ['Benign', 'Reconnaissance', 'Lateral Movement']
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    os.makedirs(args.output, exist_ok=True)

    print('Loading flows...')
    flows = read_flows(pathlib.Path(args.input))
    
    print(flows.info())
    
    flows = prepare_flow_features(flows)
    flows, labels = encode_flow_stages(flows, stages)
    
    num_labels = labels.nunique()
    
    print('Splitting flows...')
    train_flows, val_flows, train_labels, val_labels = split_flows(
        flows,
        labels,
        sequence_length=args.sequence_length,
        test_size=0.1,
        random_seed=args.seed
    )

    print('Creating experiment...')
    experiment = Experiment.with_transformer(train_flows, val_flows, args.sequence_length, num_labels)
    
    print('Creating datasets and dataloaders...')
    train_ds, val_ds = create_datasets(
        experiment.train_features,
        experiment.val_features,
        train_labels,
        val_labels,
        sequence_length=args.sequence_length,
        cat_cols=experiment.cat_cols,
        cont_cols=experiment.cont_cols
    )
    
    train_loader, val_loader = create_dataloaders(train_ds, val_ds, batch_size=args.batch_size)

    experiment.model.to(args.device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(experiment.model.parameters(), lr=1e-4)

    train_metrics = MetricCollection({
        'accuracy': Accuracy(task='multiclass', num_classes=num_labels),
        'f1_score': F1Score(task='multiclass', average='macro', num_classes=num_labels)
    }).to(args.device)
    
    val_metrics = train_metrics.clone().to(args.device)

    print('Training model...')
    for epoch in range(args.epochs):
        total_train_loss = train_model(experiment.model, train_loader, criterion, optimizer, args.device, train_metrics)
        total_val_loss, _, _ = evaluate_model(experiment.model, val_loader, criterion, args.device, val_metrics)
        
        info = ''
        info += f'Epoch {epoch+1}/{args.epochs}'
        info += f' - loss: {total_train_loss:.4f}'

        for name, metric in train_metrics.items():
            info += f' - {name}: {metric.compute().item():.4f}'
        
        info += f' - val_loss: {total_val_loss:.4f}'
        
        for name, metric in val_metrics.items():
            info += f' - val_{name}: {metric.compute().item():.4f}'
        
        print(info)
        
        train_metrics.reset()
        val_metrics.reset()
    
    print()
    print('Evaluating model...')
    _, all_preds, all_true = evaluate_model(experiment.model, val_loader, criterion, args.device)
    
    print()
    print(classification_report(all_true, all_preds, target_names=stages))
    print(confusion_matrix(all_true, all_preds))
