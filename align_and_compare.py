#!/usr/bin/env python3
"""
align_and_compare.py

Aligns an Intel RealSense D455 (color + depth) with an iniVation DVXplorer Micro event
stream recorded in a ROS 2 bag, then compares three kinds of events on the event
camera's pixel grid:

  1. Real events from the DVXplorer.
  2. Synthetic delta-log-luminance events from consecutive RealSense color frames.
  3. Synthetic delta-log-depth events from consecutive RealSense depth frames.

Geometry, per RealSense frame:
  depth pixel -> undistorted ray -> 3D point (depth frame) -> T_ev_depth -> event camera
  -> project with event intrinsics + distortion -> z-buffer          (depth in event frame)
  event pixel + depth -> 3D point (event frame) -> T_rgb_ev -> project into color image
  -> bilinear sample                                                  (color in event frame)

  Original code written by Claude Opus 5.5. Modified code for project written by Logan H.

Inputs it can use:
  * Topics: found automatically by message type and name (override with --*-topic).
  * Intrinsics: CameraInfo topics in the bag, or --event-calib (ROS camera_info YAML),
    or --kalibr (Kalibr camchain YAML).
  * Extrinsics: /tf_static in the bag, or --extrinsics (4x4 YAML), or --kalibr.

Requirements: numpy, opencv-python, matplotlib, pyyaml.
For .mcap bags (ROS 2 Jazzy default) also: pip install rosbags

Examples:
  python3 align_and_compare.py my_bag --list
  python3 align_and_compare.py my_bag --extrinsics extrinsics.yaml --duration 10
  python3 align_and_compare.py my_bag --kalibr camchain.yaml --kalibr-event cam1 --kalibr-rgb cam0 \
      --estimate-offset --video
"""
import argparse
import csv
import sqlite3
import struct
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

NS = 1_000_000_000


# =============================================================================== CDR decoding
class Cdr:
    """Minimal little-endian CDR reader (the ROS 2 default serialization)."""

    def __init__(self, buf):
        if not buf[1] & 1:
            raise ValueError("big-endian CDR is not supported")
        self.b, self.o = buf, 4  # skip the 4-byte encapsulation header

    def align(self, n):
        r = (self.o - 4) % n
        if r:
            self.o += n - r

    def _read(self, fmt, size):
        self.align(size)
        v = struct.unpack_from(fmt, self.b, self.o)[0]
        self.o += size
        return v

    def u8(self):
        v = self.b[self.o]
        self.o += 1
        return v

    def u32(self):
        return self._read("<I", 4)

    def i32(self):
        return self._read("<i", 4)

    def f64(self):
        return self._read("<d", 8)

    def string(self):
        n = self.u32()
        s = bytes(self.b[self.o:self.o + max(n - 1, 0)]).decode(errors="replace")
        self.o += n
        return s

    def header(self):
        sec, nsec = self.i32(), self.u32()
        return sec * NS + nsec, self.string().lstrip("/")


def image_array(raw, h, w, enc, step, big_endian):
    e = enc.lower()
    rows = raw[:h * step].reshape(h, step)
    if e in ("rgb8", "bgr8", "8uc3"):
        a = rows[:, :w * 3].reshape(h, w, 3)
        return np.ascontiguousarray(a[..., ::-1] if e == "rgb8" else a)
    if e in ("rgba8", "bgra8", "8uc4"):
        a = rows[:, :w * 4].reshape(h, w, 4)[..., :3]
        return np.ascontiguousarray(a[..., ::-1] if e == "rgba8" else a)
    if e in ("mono8", "8uc1"):
        return rows[:, :w].copy()
    if e in ("16uc1", "mono16"):
        return rows[:, :w * 2].copy().view(">u2" if big_endian else "<u2").astype(np.uint16)
    if e == "32fc1":
        return rows[:, :w * 4].copy().view(">f4" if big_endian else "<f4").astype(np.float32)
    if e in ("yuv422_yuy2", "yuyv"):
        return cv2.cvtColor(rows[:, :w * 2].reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUY2)
    if e in ("yuv422", "uyvy"):
        return cv2.cvtColor(rows[:, :w * 2].reshape(h, w, 2), cv2.COLOR_YUV2BGR_UYVY)
    raise ValueError(f"unsupported image encoding '{enc}'")


def decode_image(buf):
    """sensor_msgs/Image -> (stamp_ns, frame_id, array). Color comes back as BGR."""
    r = Cdr(buf)
    t, frame = r.header()
    h, w = r.u32(), r.u32()
    enc = r.string()
    big = r.u8()
    step = r.u32()
    n = r.u32()
    return t, frame, image_array(np.frombuffer(buf, np.uint8, n, r.o), h, w, enc, step, big)


def decode_compressed_image(buf):
    """sensor_msgs/CompressedImage (jpeg/png, or compressedDepth png)."""
    r = Cdr(buf)
    t, frame = r.header()
    fmt = r.string()
    n = r.u32()
    data = np.frombuffer(buf, np.uint8, n, r.o)
    if "compresseddepth" in fmt.lower():
        if "rvl" in fmt.lower():
            raise ValueError("compressedDepth RVL is not supported; record raw depth or png")
        data = data[12:]  # compressedDepth prepends a 12-byte config header
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"could not decode compressed image with format '{fmt}'")
    return t, frame, img


def decode_camera_info(buf):
    r = Cdr(buf)
    _, frame = r.header()
    h, w = r.u32(), r.u32()
    model = r.string()
    d = [r.f64() for _ in range(r.u32())]
    k = [r.f64() for _ in range(9)]
    return CameraModel(np.array(k).reshape(3, 3), d, (w, h), model), frame


def decode_tf(buf):
    """tf2_msgs/TFMessage -> list of (parent, child, T_parent_child)."""
    r = Cdr(buf)
    out = []
    for _ in range(r.u32()):
        _, parent = r.header()
        child = r.string().lstrip("/")
        t = [r.f64() for _ in range(3)]
        q = [r.f64() for _ in range(4)]  # x, y, z, w
        out.append((parent, child, make_T(t, q)))
    return out


EVENT_DTYPE = np.dtype([("x", "<u2"), ("y", "<u2"), ("_pad0", "<u2"), ("sec", "<i4"),
                        ("nsec", "<u4"), ("pol", "u1"), ("_pad1", "u1")])


def _event_sequence(buf, o, n):
    """Event = {uint16 x, uint16 y, builtin_interfaces/Time ts, bool polarity}.
    The first element is 14 bytes; CDR alignment makes every later one a 16-byte stride."""
    x0, y0, s0, ns0, p0 = struct.unpack_from("<HHiIB", buf, o)
    t = np.empty(n, np.int64)
    x = np.empty(n, np.uint16)
    y = np.empty(n, np.uint16)
    p = np.empty(n, np.int8)
    t[0], x[0], y[0], p[0] = s0 * NS + ns0, x0, y0, 1 if p0 else -1
    if n > 1:
        rest = np.frombuffer(bytes(buf) + b"\x00", EVENT_DTYPE, n - 1, o + 14)
        t[1:] = rest["sec"].astype(np.int64) * NS + rest["nsec"]
        x[1:], y[1:] = rest["x"], rest["y"]
        p[1:] = np.where(rest["pol"] > 0, 1, -1)
    return t, x, y, p


def decode_events(buf):
    """dv_ros2_msgs / dv_processing_driver / dvs_msgs-style EventArray -> (t_ns, x, y, polarity).
    Tries the layout with height/width fields first, then without, and checks the byte count."""
    for has_size in (True, False):
        try:
            r = Cdr(buf)
            r.header()
            w = h = 0
            if has_size:
                h, w = r.u32(), r.u32()
            n = r.u32()
            if n == 0:
                if len(buf) - r.o <= 3:
                    return (np.empty(0, np.int64), np.empty(0, np.uint16),
                            np.empty(0, np.uint16), np.empty(0, np.int8))
                continue
            end = r.o + 13 + 16 * (n - 1)
            if not end <= len(buf) <= end + 3:
                continue
            t, x, y, p = _event_sequence(buf, r.o, n)
            if has_size and w and h and (x.max() >= w or y.max() >= h):
                continue
            return t, x, y, p
        except (struct.error, ValueError, IndexError):
            continue
    raise ValueError("Unrecognized event message layout. Run `ros2 interface show <type>` on the "
                     "event message type and compare with Event{x, y, ts, polarity}.")


# =============================================================================== bag access
class Bag:
    """Reads raw CDR messages from a rosbag2 directory (.db3 built in; .mcap via `rosbags`)."""

    def __init__(self, path):
        p = Path(path)
        self.dir = p if p.is_dir() else p.parent
        self.db3 = sorted(self.dir.glob("*.db3")) if p.is_dir() else (
            [p] if p.suffix == ".db3" else [])
        mcap = list(self.dir.glob("*.mcap"))
        self.use_rosbags = not self.db3
        if self.use_rosbags:
            if not mcap:
                raise FileNotFoundError(f"no .db3 or .mcap files found in {self.dir}")
            try:
                import rosbags.rosbag2  # noqa: F401
            except ImportError:
                sys.exit("This bag is .mcap; install the reader with: pip install rosbags")

    def topics(self):
        out = {}
        if self.use_rosbags:
            from rosbags.rosbag2 import Reader
            with Reader(self.dir) as r:
                for c in r.connections:
                    ty, n = out.get(c.topic, (c.msgtype, 0))
                    out[c.topic] = (ty, n + c.msgcount)
            return out
        for f in self.db3:
            db = sqlite3.connect(f)
            counts = dict(db.execute("select topic_id, count(*) from messages group by topic_id"))
            for tid, name, ty in db.execute("select id, name, type from topics"):
                old = out.get(name, (ty, 0))
                out[name] = (ty, old[1] + counts.get(tid, 0))
            db.close()
        return out

    def time_range(self):
        if self.use_rosbags:
            from rosbags.rosbag2 import Reader
            with Reader(self.dir) as r:
                return r.start_time, r.end_time
        lo, hi = None, None
        for f in self.db3:
            db = sqlite3.connect(f)
            a, b = db.execute("select min(timestamp), max(timestamp) from messages").fetchone()
            db.close()
            if a is not None:
                lo = a if lo is None else min(lo, a)
                hi = b if hi is None else max(hi, b)
        return lo, hi

    def messages(self, topic, start=None, stop=None):
        """Yields (bag_timestamp_ns, raw_cdr_bytes) in bag-time order."""
        if self.use_rosbags:
            from rosbags.rosbag2 import Reader
            with Reader(self.dir) as r:
                conns = [c for c in r.connections if c.topic == topic]
                for _, ts, raw in r.messages(connections=conns, start=start, stop=stop):
                    yield ts, bytes(raw)
            return
        for f in self.db3:
            db = sqlite3.connect(f)
            row = db.execute("select id from topics where name=?", (topic,)).fetchone()
            if row is None:
                db.close()
                continue
            q = "select timestamp, data from messages where topic_id=?"
            params = [row[0]]
            if start is not None:
                q += " and timestamp>=?"
                params.append(int(start))
            if stop is not None:
                q += " and timestamp<?"
                params.append(int(stop))
            for ts, data in db.execute(q + " order by timestamp", params):
                yield ts, data
            db.close()

    def first(self, topic):
        return next(self.messages(topic), (None, None))[1]


# =============================================================================== calibration
def make_T(t, q):
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = t
    return T


class CameraModel:
    def __init__(self, K, D, size, model="plumb_bob"):
        self.K = np.asarray(K, float).reshape(3, 3)
        self.size = (int(size[0]), int(size[1]))  # (width, height)
        m = (model or "plumb_bob").lower()
        self.fisheye = m in ("equidistant", "fisheye", "kannala_brandt")
        D = list(D or [])
        n = 4 if self.fisheye else next(k for k in (5, 8, 12, 14) if k >= len(D))
        self.D = np.zeros(n)
        self.D[:min(n, len(D))] = D[:n]

    @property
    def valid(self):
        return self.K[0, 0] > 0 and self.size[0] > 0 and self.size[1] > 0

    def project(self, P):
        """3D points in this camera's optical frame (N x 3) -> distorted pixels (N x 2)."""
        if len(P) == 0:
            return np.empty((0, 2))
        P = np.ascontiguousarray(P, np.float64).reshape(-1, 1, 3)
        z = np.zeros(3)
        if self.fisheye:
            px, _ = cv2.fisheye.projectPoints(P, z, z, self.K, self.D)
        else:
            px, _ = cv2.projectPoints(P, z, z, self.K, self.D)
        return px.reshape(-1, 2)

    def rays(self):
        """Undistorted ray (x', y', 1) for every pixel, row-major: shape (H*W, 3)."""
        W, H = self.size
        g = np.stack(np.meshgrid(np.arange(W), np.arange(H)), -1).reshape(-1, 1, 2).astype(np.float64)
        if self.fisheye:
            u = cv2.fisheye.undistortPoints(g, self.K, self.D)
        else:
            crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-9)
            try:
                u = cv2.undistortPointsIter(g, self.K, self.D, np.eye(3), np.eye(3), crit)
            except (AttributeError, cv2.error):
                u = cv2.undistortPoints(g, self.K, self.D)
        u = u.reshape(-1, 2)
        return np.column_stack([u, np.ones(len(u))])

    def scaled_to(self, size):
        """Same camera at a different resolution (e.g. a calibration done at another size)."""
        sx, sy = size[0] / self.size[0], size[1] / self.size[1]
        K = self.K.copy()
        K[0] *= sx
        K[1] *= sy
        return CameraModel(K, self.D, size, "equidistant" if self.fisheye else "plumb_bob")


def load_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def camera_from_ros_yaml(path):
    y = load_yaml(path)
    return CameraModel(np.array(y["camera_matrix"]["data"]).reshape(3, 3),
                       y["distortion_coefficients"]["data"],
                       (y["image_width"], y["image_height"]), y.get("distortion_model"))


def camera_from_kalibr(c):
    if c.get("camera_model", "pinhole") != "pinhole":
        raise ValueError(f"Kalibr camera model '{c['camera_model']}' is not supported (pinhole only)")
    fu, fv, pu, pv = c["intrinsics"]
    K = [[fu, 0, pu], [0, fv, pv], [0, 0, 1]]
    dm = c.get("distortion_model", "radtan")
    if dm not in ("radtan", "equidistant", "none"):
        raise ValueError(f"Kalibr distortion model '{dm}' is not supported")
    return CameraModel(K, c.get("distortion_coeffs", []), c["resolution"],
                       "equidistant" if dm == "equidistant" else "plumb_bob")


def load_kalibr(path, ev_name, rgb_name):
    """Kalibr camchain: T_cn_cnm1 maps points from cam(n-1) into cam(n)."""
    y = load_yaml(path)
    names = sorted(y, key=lambda s: int("".join(ch for ch in s if ch.isdigit()) or 0))
    T = {names[0]: np.eye(4)}
    for prev, cur in zip(names, names[1:]):
        T[cur] = np.array(y[cur]["T_cn_cnm1"], float) @ T[prev]  # T_cur_cam0
    for n in (ev_name, rgb_name):
        if n not in y:
            raise KeyError(f"'{n}' not in Kalibr file (has {names})")
    return T[ev_name] @ np.linalg.inv(T[rgb_name]), camera_from_kalibr(y[ev_name])


class StaticTF:
    def __init__(self):
        self.edges = {}

    def add(self, parent, child, T_parent_child):
        self.edges.setdefault(parent, []).append((child, T_parent_child))
        self.edges.setdefault(child, []).append((parent, np.linalg.inv(T_parent_child)))

    def lookup(self, target, source):
        """T_target_source: maps points expressed in `source` into `target`, or None."""
        acc, todo = {target: np.eye(4)}, deque([target])
        while todo:
            node = todo.popleft()
            for nb, T in self.edges.get(node, []):
                if nb not in acc:
                    acc[nb] = acc[node] @ T
                    todo.append(nb)
        return acc.get(source)


# =============================================================================== geometry
class Aligner:
    """Brings RealSense depth and color into the event camera's pixel grid."""

    def __init__(self, depth_cam, rgb_cam, ev_cam, T_ev_depth, T_rgb_ev,
                 stride=1, zmin=0.2, zmax=8.0, fill_iters=1):
        self.depth_cam, self.rgb_cam, self.ev_cam = depth_cam, rgb_cam, ev_cam
        self.T_ev_depth, self.T_rgb_ev = T_ev_depth, T_rgb_ev
        self.stride, self.zmin, self.zmax, self.fill = stride, zmin, zmax, fill_iters
        Wd, Hd = depth_cam.size
        self.depth_rays = depth_cam.rays().reshape(Hd, Wd, 3)[::stride, ::stride].reshape(-1, 3)
        self.ev_rays = ev_cam.rays()

    def depth_to_event(self, depth_m):
        """Depth image (meters, 0/NaN = invalid) -> depth in the event grid (NaN = no data)."""
        W, H = self.ev_cam.size
        z = depth_m[::self.stride, ::self.stride].reshape(-1)
        ok = np.isfinite(z) & (z > self.zmin) & (z < self.zmax)
        P = self.depth_rays[ok] * z[ok, None]
        Pe = P @ self.T_ev_depth[:3, :3].T + self.T_ev_depth[:3, 3]
        Pe = Pe[Pe[:, 2] > 0.05]
        px = np.rint(self.ev_cam.project(Pe)).astype(np.int64)
        inb = (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
        flat = np.full(W * H, np.inf, np.float32)
        np.minimum.at(flat, px[inb, 1] * W + px[inb, 0], Pe[inb, 2].astype(np.float32))  # z-buffer
        d = flat.reshape(H, W)
        for _ in range(self.fill):  # close one-pixel cracks left by the forward splat
            missing = ~np.isfinite(d)
            nb = cv2.erode(np.where(missing, 1e9, d).astype(np.float32), np.ones((3, 3), np.uint8))
            d = np.where(missing & (nb < 1e8), nb, d)
        d[~np.isfinite(d)] = np.nan
        return d

    def color_maps(self, depth_ev):
        """For every event pixel with depth, where to sample the color image (-1 = nowhere)."""
        W, H = self.ev_cam.size
        Wc, Hc = self.rgb_cam.size
        idx = np.flatnonzero(np.isfinite(depth_ev))
        Pe = self.ev_rays[idx] * depth_ev.reshape(-1)[idx, None]
        Pr = Pe @ self.T_rgb_ev[:3, :3].T + self.T_rgb_ev[:3, 3]
        front = Pr[:, 2] > 0.05
        idx, Pr = idx[front], Pr[front]
        px = self.rgb_cam.project(Pr)
        inside = (px[:, 0] >= 0) & (px[:, 0] <= Wc - 1) & (px[:, 1] >= 0) & (px[:, 1] <= Hc - 1)
        mx = np.full(W * H, -1, np.float32)
        my = np.full(W * H, -1, np.float32)
        mx[idx[inside]], my[idx[inside]] = px[inside, 0], px[inside, 1]
        return mx.reshape(H, W), my.reshape(H, W)


def remap(img, mx, my):
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def linear_luminance(img, linearize=True):
    """Relative linear luminance in [0, 1]. RealSense color output is sRGB-encoded,
    while event pixels respond to (log) linear irradiance, so undo the gamma by default."""
    x = img.astype(np.float32) / (65535.0 if img.dtype == np.uint16 else 255.0)
    if linearize:
        x = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    if x.ndim == 3:
        x = 0.0722 * x[..., 0] + 0.7152 * x[..., 1] + 0.2126 * x[..., 2]  # BGR order
    return x.astype(np.float32)


def to_meters(depth, scale):
    if depth.dtype == np.uint16:
        return depth.astype(np.float32) * scale
    return depth.astype(np.float32)


# =============================================================================== events
class EventStream:
    """Pulls event messages lazily and answers time-window queries in increasing time."""

    def __init__(self, gen):
        self.gen, self.chunks, self.done, self.last_t = gen, deque(), False, -np.inf

    def window(self, t0, t1):
        while not self.done and self.last_t < t1:
            try:
                _, buf = next(self.gen)
            except StopIteration:
                self.done = True
                break
            ch = decode_events(buf)
            if len(ch[0]):
                self.chunks.append(ch)
                self.last_t = max(self.last_t, int(ch[0][-1]))
        while self.chunks and self.chunks[0][0][-1] < t0 - NS // 2:
            self.chunks.popleft()
        parts = []
        for t, x, y, p in self.chunks:
            if t[-1] < t0 or t[0] >= t1:
                continue
            m = (t >= t0) & (t < t1)
            parts.append((t[m], x[m], y[m], p[m]))
        if not parts:
            return np.empty(0, np.int64), np.empty(0, np.uint16), np.empty(0, np.uint16), np.empty(0, np.int8)
        return tuple(np.concatenate(c) for c in zip(*parts))


def real_event_image(ev, size):
    W, H = size
    img = np.zeros(H * W, np.float32)
    _, x, y, p = ev
    ok = (x < W) & (y < H)
    np.add.at(img, y[ok].astype(np.int64) * W + x[ok], p[ok].astype(np.float32))
    return img.reshape(H, W)


def delta_log_events(a0, a1, c, valid, eps=0.0, normalize=False):
    """Signed count of ideal events per pixel: trunc(delta log / C). NaN where not valid."""
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.log(a1 + eps) - np.log(a0 + eps)
    if normalize and valid.any():
        d = d - np.median(d[valid])  # remove a global gain step (e.g. auto-exposure)
    out = np.trunc(d / c).astype(np.float32)
    out[~valid] = np.nan
    return out


def compare(real, syn):
    """Pearson correlation of blurred signed maps, and polarity agreement where both fire."""
    m = np.isfinite(syn)
    if m.sum() < 200:
        return np.nan, np.nan
    a = cv2.GaussianBlur(real, (5, 5), 0)[m]
    b = cv2.GaussianBlur(np.nan_to_num(syn), (5, 5), 0)[m]
    corr = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan
    both = m & (real != 0) & (np.nan_to_num(syn) != 0)
    agree = np.mean(np.sign(real[both]) == np.sign(syn[both])) if both.any() else np.nan
    return corr, agree


# =============================================================================== rendering
GRAY = 205


def colorize_signed(ev, vmax=None):
    """White = no events, red = positive (ON), blue = negative (OFF), gray = no data."""
    e = np.nan_to_num(ev)
    if vmax is None:
        nz = np.abs(e[e != 0])
        vmax = max(1.0, float(np.percentile(nz, 98))) if nz.size else 1.0
    v = np.clip(e / vmax, -1, 1)
    out = np.full(ev.shape + (3,), 255, np.uint8)
    pos, neg = v > 0, v < 0
    out[pos, 0] = out[pos, 1] = (255 * (1 - v[pos])).astype(np.uint8)
    out[neg, 1] = out[neg, 2] = (255 * (1 + v[neg])).astype(np.uint8)
    out[~np.isfinite(ev)] = GRAY
    return out


def colorize_depth(d, zmin, zmax):
    inv = (1 / np.clip(np.nan_to_num(d, nan=zmax), zmin, zmax) - 1 / zmax) / (1 / zmin - 1 / zmax)
    out = cv2.applyColorMap((255 * np.clip(inv, 0, 1)).astype(np.uint8), cv2.COLORMAP_TURBO)
    out[~np.isfinite(d)] = GRAY
    return out


def color_panel(bgr_ev, valid):
    img = bgr_ev if bgr_ev.ndim == 3 else cv2.cvtColor(bgr_ev, cv2.COLOR_GRAY2BGR)
    img = img.copy()
    img[~valid] = GRAY
    return img


def overlay_panel(color, real):
    out = (0.6 * color + 0.4 * 255).astype(np.uint8)  # lighten so events stand out
    out[real > 0] = (0, 0, 255)
    out[real < 0] = (255, 0, 0)
    return out


PANEL_TITLES = ["Color (in event frame)", "Depth (in event frame)", "Real events over color",
                "Real events", "Δ log luminance events", "Δ log depth events"]


def build_panels(pair, args):
    color = color_panel(pair["color"], pair["valid"])
    return [color, colorize_depth(pair["depth"], args.min_depth, args.max_depth),
            overlay_panel(color, pair["real"]), colorize_signed(pair["real"]),
            colorize_signed(pair["lum"]), colorize_signed(pair["dep"])]


def video_frame(panels, pair, scale):
    tiles = []
    for img, title in zip(panels, PANEL_TITLES):
        t = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(t, (0, 0), (t.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(t, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    grid = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
    cv2.putText(grid, f"t = {pair['t_rel']:.3f} s", (8, grid.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return grid


def save_figure(pairs, args, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(pairs), 6, figsize=(22, 3.4 * len(pairs)), squeeze=False)
    for row, pair in enumerate(pairs):
        counts = [None, None, None, pair["n_real"], pair["n_lum"], pair["n_dep"]]
        for col, img in enumerate(build_panels(pair, args)):
            ax = axes[row, col]
            ax.imshow(img[..., ::-1])
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(PANEL_TITLES[col], fontsize=12)
            if col == 0:
                ax.set_ylabel(f"t = {pair['t_rel']:.3f} s", fontsize=11)
            if counts[col] is not None:
                ax.text(6, img.shape[0] - 12, f"{counts[col]:,} events", fontsize=9,
                        bbox=dict(facecolor="white", alpha=0.85, lw=0))
    fig.text(0.5, 0.004, "Red = positive (ON), blue = negative (OFF), gray = no depth/color for "
             "that event pixel (outside the RealSense view, occluded, or invalid depth).",
             ha="center", fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=100)
    plt.close(fig)


# =============================================================================== setup helpers
def pick(cands, prefer=(), avoid=()):
    def score(t):
        s = t.lower()
        return (sum(p in s for p in prefer) * 10 - sum(a in s for a in avoid) * 100, -len(t))
    return max(cands, key=score) if cands else None


def common_prefix(a, b):
    n = 0
    for x, y in zip(a.strip("/").split("/"), b.strip("/").split("/")):
        if x != y:
            break
        n += 1
    return n


def info_for(image_topic, info_topics, min_common=1):
    if not image_topic or not info_topics:
        return None
    best = max(info_topics, key=lambda t: (common_prefix(image_topic, t), -len(t)))
    return best if common_prefix(image_topic, best) >= min_common else None


def detect_topics(topics, args):
    by_type = lambda *ts: [n for n, (ty, _) in topics.items() if ty in ts]
    images = by_type("sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage")
    infos = by_type("sensor_msgs/msg/CameraInfo")
    events = [n for n, (ty, _) in topics.items() if ty.endswith("/EventArray")]
    sel = dict(
        events=args.event_topic or pick(events),
        color=args.color_topic or pick([t for t in images if "depth" not in t.lower()],
                                       prefer=("color", "image_raw"),
                                       avoid=("dvs", "event", "infra", "compresseddepth")),
        depth=args.depth_topic or pick([t for t in images if "depth" in t.lower()],
                                       prefer=("aligned_depth_to_color", "image_raw")))
    sel["color_info"] = args.color_info_topic or info_for(sel["color"], infos, 2)
    sel["depth_info"] = args.depth_info_topic or info_for(sel["depth"], infos, 2)
    sel["event_info"] = args.event_info_topic or info_for(
        sel["events"], [t for t in infos if t not in (sel["color_info"], sel["depth_info"])], 1)
    return sel


def read_color(bag, topic, ty, start, stop):
    dec = decode_compressed_image if ty.endswith("CompressedImage") else decode_image
    for _, buf in bag.messages(topic, start, stop):
        yield dec(buf)


def read_depth(bag, topic, ty, start, stop, scale, median):
    dec = decode_compressed_image if ty.endswith("CompressedImage") else decode_image
    for _, buf in bag.messages(topic, start, stop):
        t, frame, d = dec(buf)
        if median > 1 and d.dtype == np.uint16:
            d = cv2.medianBlur(d, median if median in (3, 5) else 5)
        yield t, frame, to_meters(d, scale)


class DepthMatcher:
    def __init__(self, gen, tol_ns):
        self.gen, self.buf, self.tol, self.done = gen, deque(), tol_ns, False

    def get(self, t):
        while not self.done and (not self.buf or self.buf[-1][0] < t + self.tol):
            try:
                self.buf.append(next(self.gen))
            except StopIteration:
                self.done = True
        while len(self.buf) > 1 and self.buf[0][0] < t - self.tol:
            self.buf.popleft()
        if not self.buf:
            return None
        best = min(self.buf, key=lambda e: abs(e[0] - t))
        return best if abs(best[0] - t) <= self.tol else None


def estimate_offset(bag, sel, types, start, stop, max_ms):
    """Finds the shift that best correlates event counts with frame-to-frame brightness change."""
    stamps, change, prev = [], [], None
    for t, _, img in read_color(bag, sel["color"], types[sel["color"]], start, stop):
        L = np.log(cv2.resize(linear_luminance(img), (160, 120), interpolation=cv2.INTER_AREA) + 0.01)
        if prev is not None:
            stamps.append((prev[0], t))
            change.append(float(np.mean(np.abs(L - prev[1]))))
        prev = (t, L)
    ts = []
    for _, buf in bag.messages(sel["events"], start - 2 * NS, stop + 2 * NS):
        ts.append(decode_events(buf)[0])
    if len(change) < 10 or not ts:
        return 0, np.nan, np.nan
    ts = np.sort(np.concatenate(ts))
    s = np.array(stamps)
    change = np.array(change)
    best = (-2.0, 0)
    corr0 = np.nan
    for off_ms in range(-max_ms, max_ms + 1):
        off = off_ms * 1_000_000
        cnt = np.searchsorted(ts, s[:, 1] + off) - np.searchsorted(ts, s[:, 0] + off)
        c = np.corrcoef(cnt, change)[0, 1] if cnt.std() > 0 else -1
        if off_ms == 0:
            corr0 = c
        best = max(best, (c, off))
    return best[1], best[0], corr0


# =============================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", help="rosbag2 directory (or a .db3 file)")
    ap.add_argument("--list", action="store_true", help="print topics and the auto-detected choice, then exit")
    g = ap.add_argument_group("topics (auto-detected when omitted)")
    for name in ("event", "color", "depth", "event-info", "color-info", "depth-info"):
        g.add_argument(f"--{name}-topic")
    g = ap.add_argument_group("calibration")
    g.add_argument("--event-calib", help="ROS camera_info YAML for the event camera")
    g.add_argument("--extrinsics", help="YAML with a 4x4 T_ev_rgb (or T_rgb_ev); optional T_rgb_depth")
    g.add_argument("--invert-extrinsics", action="store_true", help="use the inverse of the given extrinsics")
    g.add_argument("--kalibr", help="Kalibr camchain YAML (event intrinsics + extrinsics)")
    g.add_argument("--kalibr-event", default="cam1")
    g.add_argument("--kalibr-rgb", default="cam0")
    g.add_argument("--event-frame", help="event camera optical frame id for TF lookups")
    g.add_argument("--tf-topic", default="/tf_static")
    g.add_argument("--ignore-rgb-distortion", action="store_true",
                   help="treat the color camera as distortion-free")
    g = ap.add_argument_group("time")
    g.add_argument("--start", type=float, default=0.0, help="seconds after the bag start")
    g.add_argument("--duration", type=float, default=10.0, help="seconds to process")
    g.add_argument("--time-offset-ms", type=float, default=0.0,
                   help="event_time = realsense_time + offset")
    g.add_argument("--estimate-offset", action="store_true", help="estimate the offset from the data")
    g.add_argument("--max-offset-ms", type=int, default=100)
    g.add_argument("--sync-tol-ms", type=float, default=15.0, help="color/depth pairing tolerance")
    g = ap.add_argument_group("depth")
    g.add_argument("--depth-scale", type=float, default=0.001, help="meters per unit for 16-bit depth")
    g.add_argument("--min-depth", type=float, default=0.3)
    g.add_argument("--max-depth", type=float, default=8.0)
    g.add_argument("--depth-median", type=int, default=5, help="median filter on raw depth (0 = off)")
    g.add_argument("--depth-stride", type=int, default=1)
    g.add_argument("--fill-holes", type=int, default=1, help="crack-filling iterations in the event grid")
    g = ap.add_argument_group("synthetic events")
    g.add_argument("--c-lum", type=float, default=0.2, help="log-luminance contrast threshold")
    g.add_argument("--eps", type=float, default=0.01, help="offset inside log(luminance)")
    g.add_argument("--no-linearize", action="store_true", help="do not undo the sRGB gamma")
    g.add_argument("--normalize-exposure", action="store_true",
                   help="remove global brightness steps (auto-exposure) before thresholding")
    g.add_argument("--c-depth", type=float, default=0.02, help="log-depth contrast threshold")
    g = ap.add_argument_group("output")
    g.add_argument("--out", default="comparison_out")
    g.add_argument("--panels", type=int, default=4, help="number of frame pairs in the figure")
    g.add_argument("--video", action="store_true", help="also write an mp4 of every frame pair")
    g.add_argument("--video-scale", type=float, default=0.5)
    g.add_argument("--video-fps", type=float, default=10.0)
    args = ap.parse_args()

    bag = Bag(args.bag)
    topics = bag.topics()
    sel = detect_topics(topics, args)
    types = {t: ty for t, (ty, _) in topics.items()}
    if args.list:
        for t, (ty, n) in sorted(topics.items()):
            print(f"  {t:60s} {ty:45s} {n:8d}")
        print("\nauto-detected:")
        for k, v in sel.items():
            print(f"  {k:11s} {v}")
        return
    for key in ("events", "color", "depth"):
        if not sel[key]:
            sys.exit(f"Could not find the {key} topic; pass --{key.rstrip('s')}-topic. Use --list to see topics.")

    # ---- calibration ------------------------------------------------------------------
    first_color = (decode_compressed_image if types[sel["color"]].endswith("CompressedImage")
                   else decode_image)(bag.first(sel["color"]))
    first_depth = (decode_compressed_image if types[sel["depth"]].endswith("CompressedImage")
                   else decode_image)(bag.first(sel["depth"]))
    rgb_frame, depth_frame = first_color[1], first_depth[1]
    rgb_size = first_color[2].shape[1::-1]
    depth_size = first_depth[2].shape[1::-1]

    if not sel["color_info"]:
        sys.exit("No CameraInfo for the color topic; pass --color-info-topic.")
    rgb_cam, _ = decode_camera_info(bag.first(sel["color_info"]))
    if args.ignore_rgb_distortion:
        rgb_cam = CameraModel(rgb_cam.K, [], rgb_cam.size)
    if sel["depth_info"]:
        depth_cam, _ = decode_camera_info(bag.first(sel["depth_info"]))
    elif depth_frame == rgb_frame and depth_size == rgb_size:
        depth_cam = rgb_cam  # aligned depth shares the color intrinsics
    else:
        sys.exit("No CameraInfo for the depth topic; pass --depth-info-topic.")
    for name, cam, size in (("color", rgb_cam, rgb_size), ("depth", depth_cam, depth_size)):
        if cam.size != tuple(size):
            sys.exit(f"{name} CameraInfo is {cam.size} but the images are {tuple(size)}.")

    ev_frame = args.event_frame
    ev_cam = None
    T_ev_rgb = None
    if args.kalibr:
        T_ev_rgb, ev_cam = load_kalibr(args.kalibr, args.kalibr_event, args.kalibr_rgb)
    if args.event_calib:
        ev_cam = camera_from_ros_yaml(args.event_calib)
    if ev_cam is None and sel["event_info"]:
        cam, frame = decode_camera_info(bag.first(sel["event_info"]))
        if cam.valid:
            ev_cam = cam
            ev_frame = ev_frame or frame
    if ev_cam is None or not ev_cam.valid:
        sys.exit("No usable event camera intrinsics; pass --event-calib or --kalibr.")
    if not ev_frame:
        ev_frame = Cdr(bag.first(sel["events"])).header()[1]

    ext = load_yaml(args.extrinsics) if args.extrinsics else {}
    if "T_ev_rgb" in ext:
        T_ev_rgb = np.array(ext["T_ev_rgb"], float)
    elif "T_rgb_ev" in ext:
        T_ev_rgb = np.linalg.inv(np.array(ext["T_rgb_ev"], float))

    tf = StaticTF()
    if args.tf_topic in topics:
        for _, buf in bag.messages(args.tf_topic):
            for parent, child, T in decode_tf(buf):
                tf.add(parent, child, T)

    if T_ev_rgb is not None:
        if args.invert_extrinsics:
            T_ev_rgb = np.linalg.inv(T_ev_rgb)
        if depth_frame == rgb_frame:
            T_rgb_depth = np.eye(4)
        elif "T_rgb_depth" in ext:
            T_rgb_depth = np.array(ext["T_rgb_depth"], float)
        else:
            T_rgb_depth = tf.lookup(rgb_frame, depth_frame)
            if T_rgb_depth is None:
                sys.exit(f"Depth frame '{depth_frame}' differs from color frame '{rgb_frame}' and no "
                         "TF connects them. Record aligned depth, or add T_rgb_depth to --extrinsics.")
        T_ev_depth = T_ev_rgb @ T_rgb_depth
        source = "Kalibr" if args.kalibr and not ext else "extrinsics file"
    else:
        T_ev_depth = tf.lookup(ev_frame, depth_frame)
        T_ev_rgb = tf.lookup(ev_frame, rgb_frame)
        if T_ev_depth is None or T_ev_rgb is None:
            sys.exit(f"No transform between '{ev_frame}' and '{rgb_frame}' in {args.tf_topic}. "
                     "Pass --extrinsics or --kalibr.")
        if args.invert_extrinsics:
            sys.exit("--invert-extrinsics only applies to --extrinsics / --kalibr.")
        source = args.tf_topic
    T_rgb_ev = np.linalg.inv(T_ev_rgb)

    # ---- report what will be used ----------------------------------------------------
    print("Topics:")
    for k in ("events", "color", "depth", "color_info", "depth_info", "event_info"):
        print(f"  {k:11s} {sel[k]}")
    print(f"Frames: color '{rgb_frame}', depth '{depth_frame}', event '{ev_frame}'")
    print(f"Event camera: {ev_cam.size[0]}x{ev_cam.size[1]}, fx={ev_cam.K[0, 0]:.1f}, "
          f"{'fisheye' if ev_cam.fisheye else 'plumb_bob'}")
    t_ev = T_ev_rgb[:3, 3]
    ang = np.degrees(np.arccos(np.clip((np.trace(T_ev_rgb[:3, :3]) - 1) / 2, -1, 1)))
    print(f"Extrinsics from {source}: color origin in event frame = "
          f"({t_ev[0]:+.4f}, {t_ev[1]:+.4f}, {t_ev[2]:+.4f}) m, rotation {ang:.2f} deg")

    bag_t0, bag_t1 = bag.time_range()
    start = bag_t0 + int(args.start * NS)
    stop = min(bag_t1 + 1, start + int(args.duration * NS))

    print(f"Bag file going from {start} to {stop}")

    offset = int(args.time_offset_ms * 1e6)
    if args.estimate_offset:
        off, c, c0 = estimate_offset(bag, sel, types, start, stop, args.max_offset_ms)
        print(f"Estimated time offset: {off / 1e6:+.1f} ms (correlation {c:.3f}; at 0 ms {c0:.3f})")
        if not np.isfinite(c) or c < 0.3:
            print("  Weak correlation: the scene may not have enough motion variation; "
                  "check the offset by eye in the overlay panels.")
        offset = off

    aligner = Aligner(depth_cam, rgb_cam, ev_cam, T_ev_depth, T_rgb_ev,
                      args.depth_stride, args.min_depth, args.max_depth, args.fill_holes)
    events = EventStream(bag.messages(sel["events"], start - NS, stop + NS))
    depths = DepthMatcher(read_depth(bag, sel["depth"], types[sel["depth"]], start - NS, stop + NS,
                                     args.depth_scale, args.depth_median),
                          int(args.sync_tol_ms * 1e6))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    writer = None
    rows, fig_pairs = [], []
    targets, t_first, prev = None, None, None
    n_frames = n_no_depth = 0

    for t, _, img in read_color(bag, sel["color"], types[sel["color"]], start, stop):
        n_frames += 1
        d = depths.get(t)
        if d is None:
            n_no_depth += 1
            prev = None
            continue
        depth_ev = aligner.depth_to_event(d[2])
        mx, my = aligner.color_maps(depth_ev)
        valid = mx >= 0
        cur = dict(t=t, depth=depth_ev, valid=valid, color=remap(img, mx, my),
                   lum=remap(linear_luminance(img, not args.no_linearize), mx, my))
        if t_first is None:
            t_first = t
            targets = deque(t + (i + 0.5) * (stop - start) / args.panels for i in range(args.panels))
        if prev is not None:
            ev = events.window(prev["t"] + offset, t + offset)
            real = real_event_image(ev, ev_cam.size)
            both = prev["valid"] & valid
            lum = delta_log_events(prev["lum"], cur["lum"], args.c_lum, both, args.eps,
                                   args.normalize_exposure)
            dvalid = np.isfinite(prev["depth"]) & np.isfinite(cur["depth"])
            dep = delta_log_events(prev["depth"], cur["depth"], args.c_depth, dvalid)
            c_lum, a_lum = compare(real, lum)
            c_dep, a_dep = compare(real, dep)
            pair = dict(t_rel=(prev["t"] - t_first) / NS, color=prev["color"], valid=prev["valid"],
                        depth=prev["depth"], real=real, lum=lum, dep=dep,
                        n_real=int(len(ev[0])), n_lum=int(np.nansum(np.abs(lum))),
                        n_dep=int(np.nansum(np.abs(dep))))
            rows.append(dict(t_rel=round(pair["t_rel"], 4), n_real=pair["n_real"],
                             n_real_in_overlap=int(np.abs(real[both]).sum()),
                             n_lum=pair["n_lum"], n_depth=pair["n_dep"],
                             overlap_fraction=round(float(both.mean()), 4),
                             corr_real_lum=c_lum, polarity_agree_real_lum=a_lum,
                             corr_real_depth=c_dep, polarity_agree_real_depth=a_dep))
            if targets and prev["t"] >= targets[0]:
                targets.popleft()
                fig_pairs.append(pair)
            if args.video:
                frame = video_frame(build_panels(pair, args), pair, args.video_scale)
                if writer is None:
                    writer = cv2.VideoWriter(str(out / "comparison.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                             args.video_fps, frame.shape[1::-1])
                writer.write(frame)
            if len(rows) % 25 == 0:
                print(f"  processed {len(rows)} frame pairs (t = {pair['t_rel']:.2f} s)")
        prev = cur

    if writer is not None:
        writer.release()
    if not rows:
        sys.exit(f"No frame pairs processed ({n_frames} color frames, {n_no_depth} without matching "
                 "depth). Check --sync-tol-ms, --start and --duration.")

    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if fig_pairs:
        save_figure(fig_pairs, args, out / "comparison.png")

    col = lambda k: np.array([r[k] for r in rows], float)
    print(f"\nProcessed {len(rows)} frame pairs ({n_no_depth} color frames had no matching depth).")
    print(f"Event pixels covered by RealSense data: {np.nanmean(col('overlap_fraction')) * 100:.1f}% on average")
    print(f"Mean events per pair: real {np.mean(col('n_real')):,.0f} "
          f"(in overlap {np.mean(col('n_real_in_overlap')):,.0f}), "
          f"Δlog luminance {np.mean(col('n_lum')):,.0f}, Δlog depth {np.mean(col('n_depth')):,.0f}")
    print(f"Real vs Δlog luminance: correlation {np.nanmean(col('corr_real_lum')):.3f}, "
          f"polarity agreement {np.nanmean(col('polarity_agree_real_lum')) * 100:.1f}%")
    print(f"Real vs Δlog depth:     correlation {np.nanmean(col('corr_real_depth')):.3f}, "
          f"polarity agreement {np.nanmean(col('polarity_agree_real_depth')) * 100:.1f}%")
    if np.mean(col("n_real")) == 0:
        print("WARNING: no events fell in any frame interval. The event and RealSense clocks probably "
              "differ; try --estimate-offset or --time-offset-ms.")
    print(f"Wrote {out / 'metrics.csv'}" + (f", {out / 'comparison.png'}" if fig_pairs else "")
          + (f", {out / 'comparison.mp4'}" if writer is not None else ""))


if __name__ == "__main__":
    main()
