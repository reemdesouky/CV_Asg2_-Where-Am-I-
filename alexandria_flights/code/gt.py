"""Ground-truth helpers for testing and scoring. Already written.

Use the ground truth to check your code and to score your results, never inside your pipeline.
"""
import os, csv
import numpy as np, cv2
from utils import apply


def load_gt(flight_dir):
    """Ground-truth homographies of a synthetic flight.

    Args:
        flight_dir: path to a synthetic flight folder (corniche, loop or lost_at_sea).

    Returns:
        (N, 3, 3) array; element k takes frame-k pixels to ground-map pixels.
    """
    return np.load(os.path.join(flight_dir, 'gt', 'H_frame_to_map.npy'))


def gt_H(flight_dir, i, j):
    """Ground-truth transform from frame i to frame j: inv(H[j]) @ H[i].

    Args:
        flight_dir: path to a synthetic flight folder.
        i: index of the first frame.
        j: index of the second frame.

    Returns:
        (3, 3) homography taking frame-i pixels to frame-j pixels.
    """
    H = load_gt(flight_dir)
    G = np.linalg.inv(H[j]) @ H[i]
    return G / G[2, 2]


def gt_path(flight_dir):
    """Ground-truth transforms of every frame into frame 0, the same convention as chain().

    Args:
        flight_dir: path to a synthetic flight folder.

    Returns:
        List of N (3, 3) transforms taking frame-k pixels to frame-0 pixels.
    """
    H = load_gt(flight_dir)
    return [np.linalg.inv(H[0]) @ h for h in H]


def m_per_px(flight_dir):
    """Meters on the ground per pixel of frame 0 (the scale path_errors() uses).

    Args:
        flight_dir: path to a synthetic flight folder.

    Returns:
        A float, about 10.
    """
    with open(os.path.join(flight_dir, 'gt', 'poses.csv')) as f:
        return float(next(csv.DictReader(f))['alt_m']) / 600.0


def rotate(img, angle, scale=1.0):
    """Rotate (and optionally resize) an image about its center, keeping its size.

    Args:
        img: (H, W) or (H, W, 3) image.
        angle: rotation in degrees, counterclockwise.
        scale: resize factor.

    Returns:
        rotated: the rotated image, same size as img (corners that leave the frame are cut off).
        M: (3, 3) transform taking pixels of img to pixels of rotated.
    """
    h, w = img.shape[:2]
    A = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    return cv2.warpAffine(img, A, (w, h)), np.vstack([A, [0, 0, 1]])


def match_errors(p, q, H):
    """The error of each match under a transform: the distance, in pixels, from H(p) to q.

    Args:
        p: (N, 2) points in the first image.
        q: (N, 2) matched points in the second image.
        H: (3, 3) transform taking the first image to the second, usually the ground truth.

    Returns:
        (N,) array of errors.
    """
    return np.linalg.norm(apply(H, p) - q, axis=1)


def count_correct(p, q, H, thresh=3.0):
    """Which matches a transform confirms.

    Args:
        p: (N, 2) points in the first image.
        q: (N, 2) matched points in the second image.
        H: (3, 3) transform taking the first image to the second.
        thresh: largest error, in pixels, of a correct match.

    Returns:
        (N,) boolean array, True for the correct matches.
    """
    return match_errors(p, q, H) < thresh


def reprojection_error(T, H_true, frame_size=(640, 480)):
    """How far an estimated transform is from the truth: the mean distance between where the two put the
    four corners and the center of the frame.

    Args:
        T: (3, 3) estimated transform.
        H_true: (3, 3) true transform.
        frame_size: (width, height) of the frame.

    Returns:
        Mean error in pixels.
    """
    w, h = frame_size
    P = np.array([[0, 0], [w, 0], [w, h], [0, h], [w / 2, h / 2]], float)
    return float(np.linalg.norm(apply(T, P) - apply(H_true, P), axis=1).mean())


def path_errors(flight_dir, Ts, frame_size=(640, 480)):
    """How far each frame's center lands from the truth, in meters (the score of a chained flight).

    Args:
        flight_dir: path to a synthetic flight folder.
        Ts: list of N (3, 3) transforms, frame k -> frame 0 (from chain() or correct()).
        frame_size: (width, height) of the frames.

    Returns:
        (N,) array of errors in meters; report its mean and its last value. With fewer transforms than
        frames, the first len(Ts) frames are scored.
    """
    c = np.array([[frame_size[0] / 2, frame_size[1] / 2]])
    est = np.array([apply(T, c)[0] for T in Ts])
    true = np.array([apply(T, c)[0] for T in gt_path(flight_dir)[:len(est)]])
    return np.linalg.norm(est - true, axis=1) * m_per_px(flight_dir)
