"""Plots for the report. Already written: every figure the handout asks for, from your own outputs.

Every function draws one figure and saves it to `path` (a .png or .jpg) when one is given.
"""
import os
import numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import apply
import gt as _gt

GREEN, RED, GRAY, ORANGE = '#1F9E6E', '#E5483B', '#8A8A8A', '#D08C00'


def _save(fig, path):
    if path:
        fig.savefig(path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _rgb(img):
    return img if img.ndim == 2 else img[..., ::-1]


def _center(T, frame_size):
    w, h = frame_size
    return apply(T, np.array([[w / 2, h / 2]]))[0]


def show_corners(img, kps, R=None, path=None):
    """1.1: the frame, its Harris response and the kept corners.

    Args:
        img: (H, W) or (H, W, 3) frame.
        kps: (N, 2) corners from detect().
        R: (H, W) Harris response from harris_response(), or None to leave that panel out.
        path: where to save the figure, or None.
    """
    n = 3 if R is not None else 2
    fig, ax = plt.subplots(1, n, figsize=(5 * n, 3.9))
    ax[0].imshow(_rgb(img), cmap='gray'); ax[0].set_title('Frame')
    if R is not None:
        ax[1].imshow(np.sign(R) * np.log1p(np.abs(R) / (np.abs(R).max() * 1e-6 + 1e-12)), cmap='gray')
        ax[1].set_title('Harris response R (signed log scale)')
    ax[-1].imshow(_rgb(img), cmap='gray')
    ax[-1].scatter(kps[:, 0], kps[:, 1], s=12, facecolors='none', edgecolors='#FFD400', lw=1.0)
    ax[-1].set_title(f'{len(kps)} kept corners')
    for a in ax:
        a.axis('off')
    _save(fig, path)


def show_matches(img1, img2, p, q, correct=None, max_lines=150, title='', path=None):
    """1.2: matches between two frames side by side, green if correct and red if wrong when the truth is known.

    Args:
        img1, img2: the two frames.
        p: (N, 2) matched points in img1.
        q: (N, 2) the points they matched in img2.
        correct: (N,) boolean from gt.count_correct(), or None to draw every match in one color.
        max_lines: draw at most this many matches (all the wrong ones are always drawn).
        title: figure title.
        path: where to save the figure, or None.
    """
    h = max(img1.shape[0], img2.shape[0]); gap = 20
    pad = lambda im: cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=255)
    a, b = pad(img1), pad(img2)
    if a.ndim != b.ndim:
        a, b = [x if x.ndim == 3 else cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) for x in (a, b)]
    sep = np.full((h, gap) + a.shape[2:], 255, np.uint8)
    canvas = np.hstack([a, sep, b]); off = img1.shape[1] + gap
    rng = np.random.default_rng(0)
    if correct is None:
        idx = rng.permutation(len(p))[:max_lines]
    else:
        idx = np.r_[np.where(~correct)[0], rng.permutation(np.where(correct)[0])[:max_lines]]
    fig, ax = plt.subplots(figsize=(13, 13 * h / canvas.shape[1] + 0.6))
    ax.imshow(_rgb(canvas), cmap='gray'); ax.axis('off')
    for i in idx:
        col = GREEN if correct is None or correct[i] else RED
        ax.plot([p[i, 0], q[i, 0] + off], [p[i, 1], q[i, 1]], color=col, lw=0.8, alpha=.85)
    if not title and correct is not None:
        title = f'{len(p)} matches, {int((~correct).sum())} wrong (red)'
    ax.set_title(title)
    _save(fig, path)


def plot_inliers(counts, path=None, title='Inliers per step'):
    """3.1: number of inliers of every step.

    Args:
        counts: list of inlier counts, one per step (0 for a lost step).
        path: where to save the figure, or None.
        title: figure title.
    """
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(range(len(counts)), counts, color=GREEN, width=1)
    ax.set_xlabel('step'); ax.set_ylabel('inliers'); ax.set_title(title); ax.grid(alpha=.3, axis='y')
    _save(fig, path)


def plot_paths(flight_dir, runs, path=None, frame_size=(640, 480)):
    """3.1, 3.2, 4.2: estimated paths against the truth, and their error against distance flown.

    Args:
        flight_dir: path to a synthetic flight folder (for its ground truth).
        runs: dict {label: list of N (3, 3) transforms, frame k -> frame 0}; several runs share one figure.
        path: where to save the figure, or None.
        frame_size: (width, height) of the frames.
    """
    gtT = _gt.gt_path(flight_dir); mpp = _gt.m_per_px(flight_dir)
    g = np.array([_center(T, frame_size) for T in gtT]) * mpp / 1000
    dist = np.r_[0, np.cumsum(np.linalg.norm(np.diff(g, axis=0), axis=1))]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    ax[0].plot(g[:, 0], -g[:, 1], 'k-', lw=2.6, label='ground truth')
    for label, Ts in runs.items():
        e = np.array([_center(T, frame_size) for T in Ts]) * mpp / 1000
        err = np.linalg.norm(e - g, axis=1) * 1000
        line, = ax[0].plot(e[:, 0], -e[:, 1], '-', lw=1.6, label=f'{label} (mean {err.mean():.0f} m)')
        ax[1].plot(dist, err, '-', color=line.get_color(), label=label)
    ax[0].set_aspect('equal'); ax[0].set_xlabel('km'); ax[0].set_ylabel('km'); ax[0].grid(alpha=.3)
    ax[0].legend(fontsize=8); ax[0].set_title(f'Path, {os.path.basename(flight_dir)}')
    ax[1].set_xlabel('distance flown (km)'); ax[1].set_ylabel('error (m)'); ax[1].grid(alpha=.3)
    ax[1].set_title('Error against distance flown')
    _save(fig, path)


def plot_error_vs_radius(pairs, path=None, frame_size=(640, 480), max_error=15.0):
    """3.2: each match's error under a transform against its distance from the image center.

    Args:
        pairs: list of (label, p, q, H): matched points p -> q and the transform to test them against,
            usually the ground truth from gt.gt_H().
        path: where to save the figure, or None.
        frame_size: (width, height) of the frames.
        max_error: leave out matches with a larger error (they are simply wrong).
    """
    c = np.array(frame_size) / 2.0
    fig, ax = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 3.8), sharey=True, squeeze=False)
    for a, (label, p, q, H) in zip(ax[0], pairs):
        e = _gt.match_errors(p, q, H); ok = e < max_error
        r = np.linalg.norm(p[ok] - c, axis=1); e = e[ok]
        a.scatter(r, e, s=6, alpha=.4, color=GREEN)
        bins = np.arange(0, r.max() + 50, 50)
        med = [np.median(e[(r >= b) & (r < b + 50)]) if ((r >= b) & (r < b + 50)).sum() > 5 else np.nan for b in bins]
        a.plot(bins + 25, med, '-o', color='k', ms=3, label='median')
        a.set_title(label); a.set_xlabel('distance from image center (px)'); a.grid(alpha=.3)
    ax[0, 0].set_ylabel('error (px)'); ax[0, 0].legend()
    _save(fig, path)


def plot_revisits(Ts, revs, path=None, flight_dir=None, frame_size=(640, 480)):
    """4.1: accepted revisits as lines between the two frames' centers, on the chained path.

    Args:
        Ts: list of N (3, 3) transforms, frame k -> frame 0 (the chained flight).
        revs: revisits from revisits(), [(i, j, H, n_inliers), ...].
        path: where to save the figure, or None.
        flight_dir: a synthetic flight folder, to plot in km; None plots in frame-0 pixels.
        frame_size: (width, height) of the frames.
    """
    s = _gt.m_per_px(flight_dir) / 1000 if flight_dir else 1.0
    c = np.array([_center(T, frame_size) for T in Ts]) * s
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.plot(c[:, 0], -c[:, 1], '-', color=GRAY, lw=2, label='chained path')
    for i, j, _, _ in revs:
        ax.plot([c[i, 0], c[j, 0]], [-c[i, 1], -c[j, 1]], color=ORANGE, lw=1.2)
    ax.plot([], [], color=ORANGE, label=f'accepted revisits ({len(revs)})')
    ax.set_aspect('equal'); ax.legend(fontsize=9); ax.grid(alpha=.3)
    ax.set_xlabel('km' if flight_dir else 'px'); ax.set_ylabel('km' if flight_dir else 'px')
    ax.set_title(f'Revisits, {os.path.basename(flight_dir)}' if flight_dir else 'Revisits')
    _save(fig, path)


def closeup(map_img, S, T, size=600, path=None, frame_size=(640, 480)):
    """3.3, 4.2: a square crop of a map around one frame.

    Args:
        map_img: the map from make_map().
        S: (3, 3) frame 0 -> map transform from make_map().
        T: (3, 3) transform of the frame to center on, frame -> frame 0.
        size: side of the crop in map pixels.
        path: where to save the crop, or None.
        frame_size: (width, height) of the frames.

    Returns:
        The crop as an image.
    """
    x, y = _center(S @ T, frame_size).astype(int)
    h, w = map_img.shape[:2]
    x0, y0 = int(np.clip(x - size // 2, 0, max(0, w - size))), int(np.clip(y - size // 2, 0, max(0, h - size)))
    crop = map_img[y0:y0 + size, x0:x0 + size]
    if path:
        cv2.imwrite(path, crop)
    return crop
