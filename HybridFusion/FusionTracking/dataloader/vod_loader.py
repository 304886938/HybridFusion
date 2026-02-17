from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Any
import os
import glob
import math
import numpy as np

# These come from your codebase (as in nusc_loader.py)
from pre_processing import arraydet2box, blend_nms  # dictdet2array not needed here


def _safe_float(x: str, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: str, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def _normalize_angle(a: float) -> float:
    # map to [-pi, pi)
    return (a + math.pi) % (2 * math.pi) - math.pi


def _yaw_to_quat_wxyz(yaw: float) -> Tuple[float, float, float, float]:
    """
    Convert BEV yaw (rotation around +z, radians) to quaternion (w, x, y, z).
    This matches the common nuScenes-style yaw-only orientation (axis z).
    """
    yaw = _normalize_angle(yaw)
    half = yaw * 0.5
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _kitti_objtype_to_label(obj_type: str, class_map: Dict[str, int]) -> Optional[int]:
    # ignore regions / invalid types
    if obj_type in ("DontCare", "Dontcare", "Ignore"):
        return None
    return class_map.get(obj_type, None)


def _parse_kitti_line(tokens: List[str],
                      is_tracking_file: bool,
                      class_map: Dict[str, int]) -> Optional[Dict[str, Any]]:
    """
    Parse one KITTI-style line.

    Two supported formats:

    (A) tracking-style (as KITTI tracking):
        frame, track_id, type, trunc, occ, alpha, bbox(4), h,w,l, x,y,z, ry, [score]
    (B) detection-style (as KITTI detection):
        type, trunc, occ, alpha, bbox(4), h,w,l, x,y,z, ry, [score]

    Notes:
    - We only need 3D box in a common ego/lidar-like frame.
    - If vx/vy not present in files, set to 0 (consistent with many detectors).
    - score: if missing, default 1.0 (but usually present for detector outputs).
    """
    if len(tokens) < 15:
        return None

    idx = 0
    frame = track_id = None
    if is_tracking_file:
        frame = _safe_int(tokens[idx]); idx += 1
        track_id = _safe_int(tokens[idx]); idx += 1

    obj_type = tokens[idx]; idx += 1
    cls = _kitti_objtype_to_label(obj_type, class_map)
    if cls is None:
        return None

    # truncation, occlusion, alpha, bbox (4)
    idx += 1 + 1 + 1 + 4

    # dimensions: h w l  (KITTI order)
    h = _safe_float(tokens[idx]); w = _safe_float(tokens[idx + 1]); l = _safe_float(tokens[idx + 2]); idx += 3
    # location: x y z
    x = _safe_float(tokens[idx]); y = _safe_float(tokens[idx + 1]); z = _safe_float(tokens[idx + 2]); idx += 3
    # yaw (ry)
    yaw = _normalize_angle(_safe_float(tokens[idx])); idx += 1

    score = 1.0
    if idx < len(tokens):
        score = _safe_float(tokens[idx], 1.0)

    return {
        "frame": frame,
        "track_id": track_id,
        "obj_type": obj_type,
        "class_label": int(cls),
        "score": float(score),
        "box": (x, y, z, w, l, h, yaw)  # store minimal
    }


def _load_tracking_seq_file(path: str, class_map: Dict[str, int]) -> Dict[int, List[Dict[str, Any]]]:
    """
    tracking mode: one txt per sequence, containing multiple frames.
    """
    out: Dict[int, List[Dict[str, Any]]] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            det = _parse_kitti_line(line.split(), is_tracking_file=True, class_map=class_map)
            if det is None:
                continue
            frame = int(det["frame"])
            out.setdefault(frame, []).append(det)
    return out


def _load_detection_frame_folder(folder: str, class_map: Dict[str, int]) -> Dict[int, List[Dict[str, Any]]]:
    """
    detection mode: per-seq folder, each frame is one txt (e.g., 000123.txt).
    """
    out: Dict[int, List[Dict[str, Any]]] = {}
    txts = sorted(glob.glob(os.path.join(folder, "*.txt")))
    for p in txts:
        base = os.path.splitext(os.path.basename(p))[0]
        frame = _safe_int(base)
        dets: List[Dict[str, Any]] = []
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                det = _parse_kitti_line(line.split(), is_tracking_file=False, class_map=class_map)
                if det is None:
                    continue
                det["frame"] = frame
                dets.append(det)
        out[frame] = dets
    return out


class VODLoader:
    """
    VoD dataloader aligned with NuScenesloader.__getitem__ output keys and np_dets layout.

    Returned dict:
        {
          'is_first_frame': bool
          'timestamp': int                       # global index
          'sample_token': str                    # f"{seq:04d}_{frame:06d}"
          'seq_id': int
          'frame_id': int
          'has_velo': bool
          'np_dets': np.ndarray [N,14]           # (x,y,z,w,l,h,vx,vy,qw,qx,qy,qz,score,label)
          'np_dets_bottom_corners': np.ndarray [N,4,2]
          'box_dets': np.ndarray[...] or list    # depends on your arraydet2box implementation
          'no_dets': bool
          'det_num': int
        }
    """
    def __init__(self,
                 detection_path: str,
                 config: Dict,
                 mode: str = "tracking",
                 seq_ids: Optional[List[int]] = None,
                 class_map: Optional[Dict[str, int]] = None):
        assert mode in ("tracking", "detection")
        self.mode = mode
        self.config = config

        pp = config.get("preprocessing", {})
        self.SF_thre = pp.get("SF_thre", {})                 # dict by class label or name
        self.NMS_thre = float(pp.get("NMS_thre", 0.1))
        self.NMS_type = pp.get("NMS_type", "blend_nms")      # usually "blend_nms"
        self.NMS_metric = pp.get("NMS_metric", "iou_bev")    # iou_bev / iou_3d / giou_bev / giou_3d / d_eucl

        self.has_velo = bool(config.get("basic", {}).get("has_velo", False))

        # VoD/KITTI-style class mapping (adjust as needed)
        self.class_map = class_map or {
            "Car": 0, "Van": 1, "Truck": 2,
            "Pedestrian": 3, "Person_sitting": 4,
            "Cyclist": 5, "Tram": 6, "Misc": 7,
        }

        # unified chronological index: (seq, frame, token)
        self.index: List[Tuple[int, int, str]] = []
        self.det_by_seq_frame: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}

        if self.mode == "tracking":
            seq_files = sorted(glob.glob(os.path.join(detection_path, "*.txt")))
            if seq_ids is not None:
                wanted = set(seq_ids)
                seq_files = [p for p in seq_files if _safe_int(os.path.splitext(os.path.basename(p))[0]) in wanted]

            for p in seq_files:
                seq = _safe_int(os.path.splitext(os.path.basename(p))[0])
                frame_to_dets = _load_tracking_seq_file(p, self.class_map)
                for frame, dets in frame_to_dets.items():
                    token = f"{seq:04d}_{frame:06d}"
                    self.det_by_seq_frame[(seq, frame)] = dets
                    self.index.append((seq, frame, token))
        else:
            seq_folders = sorted([p for p in glob.glob(os.path.join(detection_path, "*")) if os.path.isdir(p)])
            if seq_ids is not None:
                wanted = set(seq_ids)
                seq_folders = [p for p in seq_folders if _safe_int(os.path.basename(p)) in wanted]

            for folder in seq_folders:
                seq = _safe_int(os.path.basename(folder))
                frame_to_dets = _load_detection_frame_folder(folder, self.class_map)
                for frame, dets in frame_to_dets.items():
                    token = f"{seq:04d}_{frame:06d}"
                    self.det_by_seq_frame[(seq, frame)] = dets
                    self.index.append((seq, frame, token))

        self.index.sort(key=lambda x: (x[0], x[1]))

        # counters like NuScenesloader
        self.seq_id = 0
        self.frame_id = 0
        self._last_seq_seen: Optional[int] = None

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        seq, frame, token = self.index[item]
        ori_dets = self.det_by_seq_frame.get((seq, frame), [])

        # assign seq and frame id counters (like NuScenesloader)
        is_first_frame = (self._last_seq_seen is None) or (seq != self._last_seq_seen)
        if is_first_frame:
            self.seq_id += 1
            self.frame_id = 1
            self._last_seq_seen = seq
        else:
            self.frame_id += 1

        # Build list_dets (rows) in NuScenes layout (14 dims)
        list_rows: List[List[float]] = []
        for d in ori_dets:
            score = float(d["score"])
            lbl = int(d["class_label"])

            # Score Filter (support by obj_type or label id)
            thr = None
            if isinstance(self.SF_thre, dict) and len(self.SF_thre) > 0:
                if d["obj_type"] in self.SF_thre:
                    thr = float(self.SF_thre[d["obj_type"]])
                elif lbl in self.SF_thre:
                    thr = float(self.SF_thre[lbl])
            if thr is not None and score < thr:
                continue

            x, y, z, w, l, h, yaw = d["box"]
            vx, vy = 0.0, 0.0  # VoD/KITTI detector outputs often omit velocity
            qw, qx, qy, qz = _yaw_to_quat_wxyz(yaw)

            # np_dets: [x,y,z,w,l,h,vx,vy,qw,qx,qy,qz,score,label]
            list_rows.append([x, y, z, w, l, h, vx, vy, qw, qx, qy, qz, score, float(lbl)])

        np_dets = np.array(list_rows, dtype=np.float32) if len(list_rows) > 0 else np.zeros(0, dtype=np.float32)

        # NMS using your existing blend_nms (same pipeline as NuScenesloader)
        if np_dets.size != 0:
            box_dets, np_dets_bottom_corners = arraydet2box(np_dets)
            assert len(np_dets) == len(box_dets) == len(np_dets_bottom_corners)

            tmp_infos = {"np_dets": np_dets, "np_dets_bottom_corners": np_dets_bottom_corners}
            keep = globals()[self.NMS_type](box_infos=tmp_infos, metrics=self.NMS_metric, thre=self.NMS_thre)
            keep_num = len(keep)
        else:
            keep = 0
            keep_num = 0
            box_dets = np.zeros(0)
            np_dets_bottom_corners = np.zeros(0)

        # Logging (optional, can comment out)
        # print(f"\n[VOD] filtered={len(ori_dets)-keep_num}, left={keep_num}, seq_id={self.seq_id}, frame_id={self.frame_id}")

        data_info = {
            "is_first_frame": is_first_frame,
            "timestamp": item,
            "sample_token": token,
            "seq_id": self.seq_id,
            "frame_id": self.frame_id,
            "has_velo": self.has_velo,
            "np_dets": np_dets[keep] if keep_num != 0 else np.zeros(0),
            "np_dets_bottom_corners": np_dets_bottom_corners[keep] if keep_num != 0 else np.zeros(0),
            "box_dets": box_dets[keep] if keep_num != 0 else np.zeros(0),
            "no_dets": keep_num == 0,
            "det_num": keep_num,
        }
        return data_info
