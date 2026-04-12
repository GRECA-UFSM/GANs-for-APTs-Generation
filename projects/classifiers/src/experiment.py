from typing_extensions import Self

import torch.nn as nn
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

from models import MLP, Transformer, Transformer2

class Experiment:
    train_features: pd.DataFrame
    val_features: pd.DataFrame
    
    cat_cols: list[str]
    cont_cols: list[str]
    
    model: nn.Module

    def __init__(self):
        self.train_features = pd.DataFrame()
        self.val_features = pd.DataFrame()
        self.cat_cols = []
        self.cont_cols = []
        self.model = nn.Module()

    @classmethod
    def with_mlp(cls, train_flows: pd.DataFrame, val_flows: pd.DataFrame, sequence_length: int, num_labels: int) -> Self:
        cat_cols = ['src_port', 'dst_port', 'protocol']
        cont_cols = [
            'src2dst_duration_ms',
            'dst2src_duration_ms',
            'src2dst_packet_rate',
            'dst2src_packet_rate',
            'src2dst_byte_rate',
            'dst2src_byte_rate',
            'src2dst_mean_piat_ms',
            'dst2src_mean_piat_ms'
        ]
        
        train_features = pd.DataFrame()
        val_features = pd.DataFrame()
        
        cat_sizes = []
        cat_dims = [4, 4, 4]
        
        for c in cat_cols:
            encoder = LabelEncoder()
            
            encoder.fit(pd.concat([train_flows[c], val_flows[c]], ignore_index=True))
            
            train_features[c] = encoder.transform(train_flows[c]) # type: ignore
            val_features[c] = encoder.transform(val_flows[c]) # type: ignore
            
            cat_sizes.append(len(encoder.classes_))
        
        for c in cont_cols:
            scaler = StandardScaler()
            scaler.fit(train_flows[[c]])
            
            train_features[c] = scaler.transform(train_flows[[c]])
            val_features[c] = scaler.transform(val_flows[[c]])
            
        experiment = cls()
        experiment.cat_cols = cat_cols
        experiment.cont_cols = cont_cols
        experiment.train_features = train_features
        experiment.val_features = val_features
        experiment.model = MLP(
            cat_sizes=cat_sizes,
            cat_dims=cat_dims,
            num_cont=len(cont_cols),
            sequence_length=sequence_length,
            num_labels=num_labels
        )
        
        return experiment
    
    @classmethod
    def with_transformer(cls, train_flows: pd.DataFrame, val_flows: pd.DataFrame, sequence_length: int, num_labels: int) -> Self:
        
        cat_cols = [
            'src_port',
            'dst_port',
            'protocol',
            # 'src_private',
            # 'dst_private'
        ]
        cont_cols = [
            'src2dst_duration_ms',
            'dst2src_duration_ms',
            'src2dst_packet_rate',
            'dst2src_packet_rate',
            'src2dst_byte_rate',
            'dst2src_byte_rate',
            # 'bidirectional_mean_piat_ms'
            # 'src2dst_packets',
            # 'dst2src_packets',
            # 'src2dst_bytes',
            # 'dst2src_bytes',
            # 'src2dst_mean_piat_ms',
            # 'dst2src_mean_piat_ms'
        ]
        
        train_features = pd.DataFrame()
        val_features = pd.DataFrame()
        
        cat_sizes = []
        cat_dims = [
            64,
            64,
            2,
            # 2,
            # 2
        ]
        
        embed_dim = 384
        num_heads = 6
        num_layers = 6
        
        for c in cat_cols:
            encoder = LabelEncoder()
            
            encoder.fit(pd.concat([train_flows[c], val_flows[c]], ignore_index=True))
            
            train_features[c] = encoder.transform(train_flows[c]) # type: ignore
            val_features[c] = encoder.transform(val_flows[c]) # type: ignore
            
            cat_sizes.append(len(encoder.classes_))
        
        for c in cont_cols:
            scaler = StandardScaler()
            scaler.fit(train_flows[[c]])
            
            train_features[c] = scaler.transform(train_flows[[c]])
            val_features[c] = scaler.transform(val_flows[[c]])
            
        experiment = cls()
        experiment.cat_cols = cat_cols
        experiment.cont_cols = cont_cols
        experiment.train_features = train_features
        experiment.val_features = val_features
        experiment.model = Transformer2(
            cat_sizes=cat_sizes,
            # cat_dims=cat_dims,
            num_cont=len(cont_cols),
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            sequence_length=sequence_length,
            num_labels=num_labels
        )
        
        return experiment
    
# Epoch 7/11 - loss: 0.0850 - accuracy: 0.9722 - f1_score: 0.9722 - val_loss: 0.2987 - val_accuracy: 0.9231 - val_f1_score: 0.8905
# sequence_length = 
# cat_cols = [
#     'src_port',
#     'dst_port',
#     'protocol',
#     # 'src_private',
#     # 'dst_private'
# ]
# cont_cols = [
#     'src2dst_duration_ms',
#     'dst2src_duration_ms',
#     'src2dst_packet_rate',
#     'dst2src_packet_rate',
#     'src2dst_byte_rate',
#     'dst2src_byte_rate',
#     # 'src2dst_packets',
#     # 'dst2src_packets',
#     # 'src2dst_bytes',
#     # 'dst2src_bytes',
#     # 'src2dst_mean_piat_ms',
#     # 'dst2src_mean_piat_ms'
# ]

# cat_dims = [
#     32,
#     32,
#     2,
#     # 2,
#     # 2
# ]
# embed_dim = 128
# num_heads = 8
# num_layers = 1

# Epoch 2/5 - loss: 0.0797 - accuracy: 0.9758 - f1_score: 0.9757 - val_loss: 0.2584 - val_accuracy: 0.9433 - val_f1_score: 0.9008
# Transformer2
# cat_cols = [
#     'src_port',
#     'dst_port',
#     'protocol',
#     # 'src_private',
#     # 'dst_private'
# ]
# cont_cols = [
#     'src2dst_duration_ms',
#     'dst2src_duration_ms',
#     'src2dst_packet_rate',
#     'dst2src_packet_rate',
#     'src2dst_byte_rate',
#     'dst2src_byte_rate',
#     # 'bidirectional_mean_piat_ms'
#     # 'src2dst_packets',
#     # 'dst2src_packets',
#     # 'src2dst_bytes',
#     # 'dst2src_bytes',
#     # 'src2dst_mean_piat_ms',
#     # 'dst2src_mean_piat_ms'
# ]
# embed_dim = 384
# num_heads = 6
# num_layers = 6
        
  