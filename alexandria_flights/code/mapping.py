"""Q3.3: the map.

You fill in the block marked TODO. Everything else already runs.
Check your work with:  python check.py 3.3
"""
import numpy as np, cv2


def map_canvas(Ts, frame_size, scale=1.0, margin=8):
    """The canvas that holds every frame, and the transform S (a scale and a shift) that puts frame 0 onto it.

    Args:
        Ts: list of (3, 3) transforms, frame k -> frame 0.
        frame_size: (width, height) of the frames.
        scale: size of the map relative to frame 0's pixels.
        margin: empty border in map pixels.

    Returns:
        S: (3, 3) transform taking frame-0 pixels to map pixels (a scale and a shift).
        canvas_size: (width, height) of the map, multiples of 32 for the blender.
    """
    w, h = frame_size
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
    pts = np.vstack([cv2.perspectiveTransform(corners[None], np.asarray(T, np.float64))[0] for T in Ts]) * scale
    x0, y0 = pts.min(0) - margin
    x1, y1 = pts.max(0) + margin
    W, H = int(np.ceil((x1 - x0) / 32) * 32), int(np.ceil((y1 - y0) / 32) * 32)
    S = np.array([[scale, 0, -x0], [0, scale, -y0], [0, 0, 1.0]])
    return S, (W, H)


def ownership(Ts, S, frame_size, canvas_size):
    """Which frame each map pixel comes from: the frame in which it lies farthest from the border.

    Args:
        Ts: list of (3, 3) transforms, frame k -> frame 0.
        S: (3, 3) frame 0 -> map transform from map_canvas().
        frame_size: (width, height) of the frames.
        canvas_size: (width, height) of the map from map_canvas().

    Returns:
        (H, W) int32 array: the index k of the frame that owns each map pixel, or -1 where no frame lands.
    """
    w, h = frame_size
    W, H = canvas_size
    yy, xx = np.mgrid[0:h, 0:w]
    inside = np.minimum.reduce([xx, w - 1 - xx, yy, h - 1 - yy]).astype(np.float32) + 1   # distance to the border
    best = np.zeros((H, W), np.float32)
    label = np.full((H, W), -1, np.int32)
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
    for k, T in enumerate(Ts):
        M = S @ T                                        # warp only the box the frame lands in, for speed
        c = cv2.perspectiveTransform(corners[None], M)[0]
        x0, y0 = np.maximum(np.floor(c.min(0)).astype(int) - 1, 0)
        x1, y1 = np.minimum(np.ceil(c.max(0)).astype(int) + 1, [W, H])
        if x1 <= x0 or y1 <= y0:
            continue
        shift = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1.0]])
        d = cv2.warpPerspective(inside, shift @ M, (int(x1 - x0), int(y1 - y0)), flags=cv2.INTER_LINEAR)
        b, l = best[y0:y1, x0:x1], label[y0:y1, x0:x1]
        win = d > b
        b[win] = d[win]
        l[win] = k
    return label


def make_map(imgs, Ts, scale=1.0):
    """Stitch every frame into one map in frame 0's coordinates, blended with OpenCV's multiband blender.

    Args:
        imgs: the frames, in color (load_frames(..., color=True)).
        Ts: list of (3, 3) transforms taking frame-k pixels to frame-0 pixels (from chain() or correct()).
        scale: size of the map relative to frame 0's pixels.

    Returns:
        map_img: the map, a uint8 color image, white where no frame lands.
        S: (3, 3) transform taking frame-0 pixels to map pixels.
    """
    h, w = imgs[0].shape[:2]
    S, (W, H) = map_canvas(Ts, (w, h), scale)
    label = ownership(Ts, S, (w, h), (W, H))
    blender = cv2.detail_MultiBandBlender(0, 5)
    blender.prepare((0, 0, W, H))
    for k, (img, T) in enumerate(zip(imgs, Ts)):
        mask = (label == k).astype(np.uint8) * 255      # the map pixels frame k owns
        if not mask.any():
            continue
        # ------------------------------------------------------------ TODO 3.3 -----
        # Warp frame k onto the (W, H) canvas with cv2.warpPerspective. T takes it to frame 0 and S takes
        # frame 0 to the map. Then hand it to the blender: blender.feed(warped.astype(np.int16), mask, (0, 0)).
        raise NotImplementedError('TODO 3.3 in mapping.py, make_map(): warp each frame onto the canvas')
        # ------------------------------------------------------------ end TODO -----
    out, _ = blender.blend(None, None)
    out = np.clip(out, 0, 255).astype(np.uint8)
    out[label < 0] = 255
    return out, S
