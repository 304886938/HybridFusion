# Copyright (c) Ruopeng Gao. All Rights Reserved.
import os
import math
import torch
import torch.nn as nn

from typing import List
from .utils import pos_to_pos_embed, logits_to_scores
from torch.utils.checkpoint import checkpoint

from .ffn import FFN
from .mlp import MLP
from structures.track_instances import TrackInstances
from utils.utils import inverse_sigmoid
from utils.box_ops import box_cxcywh_to_xyxy, box_iou_union


class TemporalMemoryAggregation(nn.Module):
    """Temporal appearance memory aggregation module.

    Aggregates *short-term* and *long-term* appearance memories (stored per track)
    via cross-attention, then writes the aggregated appearance back into the track
    query embedding (commonly the 2nd half of `query_embed`).
    """

    def __init__(
            self,
            hidden_dim: int,
            ffn_dim: int,
            tp_drop_ratio: float,
            fp_insert_ratio: float,
            dropout: float,
            use_checkpoint: bool,
            update_threshold: float,
            use_dab: bool = False,
            long_memory_lambda: float = 0.2,
            visualize: bool = False,
    ):
        super(TemporalMemoryAggregation, self).__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.tp_drop_ratio = tp_drop_ratio
        self.fp_insert_ratio = fp_insert_ratio
        self.dropout = dropout

        self.use_checkpoint = use_checkpoint
        self.update_threshold = update_threshold

        self.use_dab = use_dab
        self.long_memory_lambda = float(long_memory_lambda)
        self.visualize = visualize

        self.short_term_memory_attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=8, batch_first=True)
        self.long_term_memory_attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=8, batch_first=True)

        self.memory_dropout = nn.Dropout(self.dropout)
        self.memory_norm = nn.LayerNorm(self.hidden_dim)
        self.memory_ffn = FFN(d_model=self.hidden_dim, d_ffn=self.ffn_dim, dropout=self.dropout)
        self.query_feat_dropout = nn.Dropout(self.dropout)
        self.query_feat_norm = nn.LayerNorm(self.hidden_dim)
        self.query_feat_ffn = FFN(d_model=self.hidden_dim, d_ffn=self.ffn_dim, dropout=self.dropout)
        self.query_pos_head = MLP(
            input_dim=self.hidden_dim*2,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_layers=2
        )
        self.linear_pos1 = nn.Linear(256, 256)
        self.linear_pos2 = nn.Linear(256, 256)
        self.norm_pos = nn.LayerNorm(256)
        self.activation = nn.ReLU(inplace=True)

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,
                previous_tracks: List[TrackInstances],
                new_tracks: List[TrackInstances],
                unmatched_dets: List[TrackInstances] | None,
                no_augment: bool = False):
        tracks = self.select_active_tracks(previous_tracks, new_tracks, unmatched_dets, no_augment=no_augment)
        tracks = self.update_tracks_embedding(tracks=tracks)

        return tracks

    def update_tracks_embedding(self, tracks: List[TrackInstances]):
        for b in range(len(tracks)):
            scores = torch.max(logits_to_scores(logits=tracks[b].logits), dim=1).values
            is_pos = scores > self.update_threshold

            tracks[b].ref_pts[is_pos] = inverse_sigmoid(tracks[b][is_pos].boxes.detach().clone())

            query_pos = pos_to_pos_embed(tracks[b].ref_pts.sigmoid(), num_pos_feats=self.hidden_dim//2)
            current_embed = tracks[b].output_embed
            short_memory = tracks[b].last_output
            long_memory = tracks[b].long_memory.detach()


            # Short Term Memory Appearance Feature
            query_pos = self.query_pos_head(query_pos)
            short_term_query = current_embed + query_pos
            short_term_key = short_memory + query_pos
            short_term_value = short_memory
            short_term_memory = self.short_term_memory_attn(short_term_query[None, :], 
                                                            short_term_key[None, :], 
                                                            short_term_value[None, :])[0][0, :]
            short_term_value = short_term_value + self.memory_dropout(short_term_memory)
            short_term_value = self.memory_norm(short_term_value)
            short_term_value = self.memory_ffn(short_term_value)
            # Short Memory ResNet
            short_term_memory_query_feat = short_memory + self.query_feat_dropout(short_term_value)
            short_term_memory_query_feat = self.query_feat_norm(short_term_memory_query_feat)
            short_term_memory_query_feat = self.query_feat_ffn(short_term_memory_query_feat)

            # Long Term Memory Appearance Feature
            long_term_query = current_embed + query_pos
            long_term_key = long_memory + query_pos
            long_term_value = long_memory
            long_term_memory = self.long_term_memory_attn(long_term_query[None, :], 
                                                            long_term_key[None, :], 
                                                            long_term_value[None, :])[0][0, :]
            long_term_value = long_term_value + self.memory_dropout(long_term_memory)
            long_term_value = self.memory_norm(long_term_value)
            long_term_value = self.memory_ffn(long_term_value)
            # Long Memory ResNet
            long_term_memory_query_feat = long_memory + self.query_feat_dropout(long_term_value)
            long_term_memory_query_feat = self.query_feat_norm(long_term_memory_query_feat)
            long_term_memory_query_feat = self.query_feat_ffn(long_term_memory_query_feat)

            # Long Short Term Memory Appearance Feature Aggregation
            long_short_term_memory_query_feat = (short_term_memory_query_feat + long_term_memory_query_feat) / 2
                       
            # -------------------------- Memory update --------------------------
            if self.long_memory_lambda <= 0:
                updated_long = long_memory
            else:
                updated_long = (1.0 - self.long_memory_lambda) * long_memory + self.long_memory_lambda * current_embed

            tracks[b].long_memory = tracks[b].long_memory * ~is_pos.reshape((is_pos.shape[0], 1)) + \
                                    updated_long * is_pos.reshape((is_pos.shape[0], 1))

            tracks[b].last_output = tracks[b].last_output * ~is_pos.reshape((is_pos.shape[0], 1)) + \
                                    current_embed * is_pos.reshape((is_pos.shape[0], 1))

            if self.use_dab:
                tracks[b].query_embed[is_pos] = long_short_term_memory_query_feat[is_pos]
            else:
                tracks[b].query_embed[:, self.hidden_dim:][is_pos] = long_short_term_memory_query_feat[is_pos]

            # Update query pos (not used in pure DAB query).
            new_query_pos = self.linear_pos2(self.activation(self.linear_pos1(current_embed)))
            query_pos = tracks[b].query_embed[:, :self.hidden_dim]
            query_pos = query_pos + new_query_pos
            query_pos = self.norm_pos(query_pos)
            tracks[b].query_embed[:, :self.hidden_dim][is_pos] = query_pos[is_pos]

        return tracks

    def select_active_tracks(self, previous_tracks: List[TrackInstances],
                             new_tracks: List[TrackInstances],
                             unmatched_dets: List[TrackInstances],
                             no_augment: bool = False):
        tracks = []
        if self.training:
            for b in range(len(new_tracks)):
                # Update fields
                new_tracks[b].last_output = new_tracks[b].output_embed
                if self.use_dab:
                    new_tracks[b].long_memory = new_tracks[b].query_embed
                else:
                    new_tracks[b].long_memory = new_tracks[b].query_embed[:, self.hidden_dim:]
                unmatched_dets[b].last_output = unmatched_dets[b].output_embed
                if self.use_dab:
                    unmatched_dets[b].long_memory = unmatched_dets[b].query_embed
                else:
                    unmatched_dets[b].long_memory = unmatched_dets[b].query_embed[:, self.hidden_dim:]
                if self.tp_drop_ratio == 0.0 and self.fp_insert_ratio == 0.0:
                    active_tracks = TrackInstances.cat_tracked_instances(previous_tracks[b], new_tracks[b])
                    active_tracks = TrackInstances.cat_tracked_instances(active_tracks, unmatched_dets[b])
                    scores = torch.max(logits_to_scores(logits=active_tracks.logits), dim=1).values
                    keep_idxes = (scores > self.update_threshold) | (active_tracks.ids >= 0)
                    active_tracks = active_tracks[keep_idxes]
                    active_tracks.ids[active_tracks.iou < 0.5] = -1
                else:
                    active_tracks = TrackInstances.cat_tracked_instances(previous_tracks[b], new_tracks[b])
                    active_tracks = active_tracks[(active_tracks.iou > 0.5) & (active_tracks.ids >= 0)]
                    if self.tp_drop_ratio > 0.0 and not no_augment:
                        if len(active_tracks) > 0:
                            tp_keep_idx = torch.rand((len(active_tracks), )) > self.tp_drop_ratio
                            active_tracks = active_tracks[tp_keep_idx]
                    if self.fp_insert_ratio > 0.0 and not no_augment:
                        selected_active_tracks = active_tracks[
                            torch.bernoulli(
                                torch.ones((len(active_tracks), )) * self.fp_insert_ratio
                            ).bool()
                        ]
                        if len(unmatched_dets[b]) > 0 and len(selected_active_tracks) > 0:
                            fp_num = len(selected_active_tracks)
                            if fp_num >= len(unmatched_dets[b]):
                                insert_fp = unmatched_dets[b]
                            else:
                                selected_active_boxes = box_cxcywh_to_xyxy(selected_active_tracks.boxes)
                                unmatched_boxes = box_cxcywh_to_xyxy(unmatched_dets[b].boxes)
                                iou, _ = box_iou_union(unmatched_boxes, selected_active_boxes)
                                fp_idx = torch.max(iou, dim=0).indices
                                fp_idx = torch.unique(fp_idx)
                                insert_fp = unmatched_dets[b][fp_idx]
                            active_tracks = TrackInstances.cat_tracked_instances(active_tracks, insert_fp)

                if len(active_tracks) == 0:
                    device = next(self.query_feat_ffn.parameters()).device
                    fake_tracks = TrackInstances(frame_height=1.0, frame_width=1.0, hidden_dim=self.hidden_dim).to(
                        device=device)
                    if self.use_dab:
                        fake_tracks.query_embed = torch.randn((1, self.hidden_dim), dtype=torch.float,
                                                              device=device)
                    else:
                        fake_tracks.query_embed = torch.randn((1, 2 * self.hidden_dim), dtype=torch.float, device=device)
                    fake_tracks.output_embed = torch.randn((1, self.hidden_dim), dtype=torch.float, device=device)
                    if self.use_dab:
                        fake_tracks.ref_pts = torch.randn((1, 4), dtype=torch.float, device=device)
                    else:
                        # fake_tracks.ref_pts = torch.randn((1, 2), dtype=torch.float, device=device)
                        fake_tracks.ref_pts = torch.randn((1, 4), dtype=torch.float, device=device)
                    fake_tracks.ids = torch.as_tensor([-2], dtype=torch.long, device=device)
                    fake_tracks.matched_idx = torch.as_tensor([-2], dtype=torch.long, device=device)
                    fake_tracks.boxes = torch.randn((1, 4), dtype=torch.float, device=device)
                    fake_tracks.logits = torch.randn((1, active_tracks.logits.shape[1]), dtype=torch.float, device=device)
                    fake_tracks.iou = torch.zeros((1,), dtype=torch.float, device=device)
                    fake_tracks.last_output = torch.randn((1, self.hidden_dim), dtype=torch.float, device=device)
                    fake_tracks.long_memory = torch.randn((1, self.hidden_dim), dtype=torch.float, device=device)
                    active_tracks = fake_tracks
                tracks.append(active_tracks)
        else:
            # Eval only has B=1.
            assert len(previous_tracks) == 1 and len(new_tracks) == 1
            new_tracks[0].last_output = new_tracks[0].output_embed
            # new_tracks[0].long_memory = new_tracks[0].query_embed
            if self.use_dab:
                new_tracks[0].long_memory = new_tracks[0].query_embed
            else:
                new_tracks[0].long_memory = new_tracks[0].query_embed[:, self.hidden_dim:]
            active_tracks = TrackInstances.cat_tracked_instances(previous_tracks[0], new_tracks[0])
            active_tracks = active_tracks[active_tracks.ids >= 0]
            tracks.append(active_tracks)
        return tracks


def build(config: dict):
    return TemporalMemoryAggregation(
        hidden_dim=config["HIDDEN_DIM"],
        ffn_dim=config["FFN_DIM"],
        dropout=config["DROPOUT"],
        tp_drop_ratio=config.get("TP_DROP_RATE", 0.0),
        fp_insert_ratio=config.get("FP_INSERT_RATE", 0.0),
        use_checkpoint=config["USE_CHECKPOINT"],
        update_threshold=config["UPDATE_THRESH"],
        use_dab=config.get("USE_DAB", False),
        long_memory_lambda=config.get("LONG_MEMORY_LAMBDA", 0.2),
        visualize=config.get("VISUALIZE", False),
    )

