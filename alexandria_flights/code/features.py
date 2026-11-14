"""Part 1: Harris corners (Q1.1), patch descriptors and matching (Q1.2).

You fill in the blocks marked TODO. Everything else already runs.
Check your work with:  python check.py 1.1   and   python check.py 1.2
"""
import numpy as np, cv2


# ===================================================================== Q1.1 Harris corners
def harris_response(img, sd=1.0, si=2.0, k=0.05):
    """Harris corner response of every pixel.

    Args:
        img: (H, W) image.
        sd: sigma of the derivative-of-Gaussian filters.
        si: sigma of the Gaussian window that sums Ix^2, Iy^2 and IxIy.
        k: the Harris constant, between 0.04 and 0.06.

    Returns:
        (H, W) float32 array R = det(M) - k * trace(M)^2.
    """
    img = img.astype(np.float32)
    # ---------------------------------------------------------------- TODO 1.1a ----
    # 1. Ix, Iy: blur with a Gaussian of sigma sd, then differentiate (cv2.GaussianBlur, cv2.Sobel).
    # 2. The entries of M: Ix*Ix, Iy*Iy and Ix*Iy, each blurred with a Gaussian of sigma si.
    # 3. R = det(M) - k * trace(M)^2, for every pixel at once.
    #raise NotImplementedError('TODO 1.1a in features.py, harris_response(): compute R')

    blurred = cv2.GaussianBlur(img, (0, 0), sd)
    Ix = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)

    Sxx = cv2.GaussianBlur(Ix * Ix, (0, 0), si)
    Syy = cv2.GaussianBlur(Iy * Iy, (0, 0), si)
    Sxy = cv2.GaussianBlur(Ix * Iy, (0, 0), si)

    det = Sxx * Syy - Sxy * Sxy
    trace = Sxx + Syy
    R = det - k * trace * trace
    return R.astype(np.float32)
    # ---------------------------------------------------------------- end TODO -----


def detect(img, total=1000, grid=(6, 8), rel_thresh=1e-3):
    """Harris corners spread over the whole frame.

    Args:
        img: (H, W) image.
        total: about how many corners to keep.
        grid: (rows, columns) of the grid the corners are spread over.
        rel_thresh: keep only maxima with R above rel_thresh * max(R).

    Returns:
        (N, 2) corner positions (x, y).
    """
    R = harris_response(img)
    # ---------------------------------------------------------------- TODO 1.1b ----
    # Find the candidate corners: pixels whose R is the largest in their 3 x 3 neighborhood (cv2.dilate helps)
    # and above rel_thresh * R.max(). Set xs, ys (their columns and rows) and scores (their R).
    #raise NotImplementedError('TODO 1.1b in features.py, detect(): find the local maxima of R')
    local_max = cv2.dilate(R, np.ones((3, 3), np.uint8))
    is_max = (R == local_max)

    is_strong = R > rel_thresh * R.max()

    ys, xs = np.nonzero(is_max & is_strong)
    scores = R[ys, xs]
    # ---------------------------------------------------------------- end TODO -----
    return spread_corners(xs, ys, scores, img.shape[:2], total, grid)


def spread_corners(xs, ys, scores, shape, total=1000, grid=(6, 8)):
    """Keep the strongest corners of each grid cell, so the corners cover the whole frame.

    Args:
        xs: (M,) x of the candidate corners.
        ys: (M,) y of the candidate corners.
        scores: (M,) their Harris response R.
        shape: (height, width) of the frame.
        total: about how many corners to keep in all.
        grid: (rows, columns) of the grid.

    Returns:
        (N, 2) corner positions (x, y), N <= total.
    """
    xs, ys, scores = np.asarray(xs), np.asarray(ys), np.asarray(scores)
    H, W = shape
    gy, gx = grid
    cell = np.minimum(ys * gy // H, gy - 1).astype(int) * gx + np.minimum(xs * gx // W, gx - 1).astype(int)
    per_cell = max(1, total // (gy * gx))
    keep = []
    for c in np.unique(cell):
        sel = np.where(cell == c)[0]
        keep += list(sel[np.argsort(-scores[sel])][:per_cell])
    return np.c_[xs[keep], ys[keep]].astype(float).reshape(-1, 2)


# ===================================================================== Q1.2 descriptors and matching
def describe(img, kps, oriented=True, sigma_ori=4.5, spacing=5, n=8):
    """Patch descriptors around corners: an n x n grid of samples, `spacing` pixels apart (a 40 x 40 window),
    normalized to zero mean and unit standard deviation. With oriented=True the grid is turned to the direction
    of the smoothed gradient at the corner, as in the oriented patches of MOPS (here at a single scale).

    Args:
        img: (H, W) image.
        kps: (N, 2) corner positions from detect().
        oriented: turn each grid to the gradient direction at its corner.
        sigma_ori: sigma of the Gaussian that smooths the gradients used for that direction.
        spacing: pixels between samples.
        n: samples per side.

    Returns:
        keypoints: (M, 2) the corners kept, M <= N (corners whose window leaves the image are dropped).
        descriptors: (M, n * n) float32, in the same order as keypoints.
    """
    b = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 2.0)                 # the image the patches sample
    gx = cv2.GaussianBlur(cv2.Sobel(b, cv2.CV_32F, 1, 0), (0, 0), sigma_ori)  # smoothed gradients, which give
    gy = cv2.GaussianBlur(cv2.Sobel(b, cv2.CV_32F, 0, 1), (0, 0), sigma_ori)  # each corner its direction
    u = (np.arange(n) - (n - 1) / 2) * spacing
    U, V = np.meshgrid(u, u)               # the sample offsets of an upright grid centered on (0, 0)
    h, w = b.shape
    r = spacing * n / 2 * 1.42             # half-diagonal of the window: keep corners at least this far in
    keep, desc = [], []
    for x, y in kps:
        if x < r or y < r or x >= w - r or y >= h - r:
            continue
        # ------------------------------------------------------------ TODO 1.2a ----
        # theta: the angle of the smoothed gradient (gx, gy) at the corner if oriented, else 0. x and y are
        # floats: convert them to int to index gx and gy.
        # X, Y: (n, n) float32 sample positions: the offsets (U, V) rotated by theta, plus (x, y).
        #raise NotImplementedError('TODO 1.2a in features.py, describe(): rotate the sample grid')
        if oriented:
            theta = np.arctan2(gy[int(y), int(x)], gx[int(y), int(x)])
        else:
            theta = 0.0
        c, s = np.cos(theta), np.sin(theta)
        X = (c * U - s * V + x).astype(np.float32)
        Y = (s * U + c * V + y).astype(np.float32)
        # ------------------------------------------------------------ end TODO -----
        patch = cv2.remap(b, X, Y, cv2.INTER_LINEAR).ravel()               # bilinear samples at (X, Y)
        # ------------------------------------------------------------ TODO 1.2b ----
        # Normalize the patch to zero mean and unit standard deviation (guard against a std of 0).
        #raise NotImplementedError('TODO 1.2b in features.py, describe(): normalize the patch')
        patch = (patch - patch.mean()) / (patch.std() + 1e-8)
        # ------------------------------------------------------------ end TODO -----
        keep.append((x, y)); desc.append(patch)
    return np.array(keep, float).reshape(-1, 2), np.array(desc, np.float32).reshape(-1, n * n)


def features(img, oriented=True):
    """Your front end, used by every part script: detect(), then describe(). From Part 3 on, features_sift()
    may take its place.

    Args:
        img: (H, W) image.
        oriented: passed on to describe().

    Returns:
        (keypoints (M, 2), descriptors (M, 64)).
    """
    return describe(img, detect(img), oriented)


def pairwise_distances(d1, d2):
    """Euclidean distance between every descriptor of d1 and every descriptor of d2, all at once.

    Args:
        d1: (N1, D) descriptors.
        d2: (N2, D) descriptors.

    Returns:
        (N1, N2) array; element (i, j) is the distance between d1[i] and d2[j].
    """
    d = (d1 ** 2).sum(1)[:, None] + (d2 ** 2).sum(1)[None, :] - 2 * d1 @ d2.T
    return np.sqrt(np.maximum(d, 0))


def match(d1, d2, ratio=0.8):
    """Nearest-neighbor matching with Lowe's ratio test.

    Args:
        d1: (N1, D) descriptors of the first frame.
        d2: (N2, D) descriptors of the second frame.
        ratio: keep a match only if its distance is below ratio times the second-nearest distance.

    Returns:
        (K, 2) integer array of index pairs (i into d1, j into d2).
    """
    if len(d1) < 2 or len(d2) < 2:
        return np.zeros((0, 2), int)
    d = pairwise_distances(d1, d2)
    # ---------------------------------------------------------------- TODO 1.2c ----
    # For every row i of d, find its nearest and second-nearest columns. Keep (i, nearest) when the nearest distance
    # is below ratio times the second-nearest. Return them as a (K, 2) integer array.
    #raise NotImplementedError('TODO 1.2c in features.py, match(): nearest neighbor and ratio test')
    two = np.argpartition(d, 1, axis=1)[:, :2]
    rows = np.arange(len(d))
    d_a = d[rows, two[:, 0]]
    d_b = d[rows, two[:, 1]]
    swap = d_b < d_a
    nearest = np.where(swap, two[:, 1], two[:, 0])
    d_near = np.where(swap, d_b, d_a)
    d_second = np.where(swap, d_a, d_b)

    # Lowe's ratio test.
    ok = d_near < ratio * d_second
    return np.stack([rows[ok], nearest[ok]], axis=1).astype(int)
    # ---------------------------------------------------------------- end TODO -----


_SIFT = {}


def features_sift(img, n=1000):
    """OpenCV SIFT as a drop-in replacement for features(): the fallback allowed from Part 3 on.

    Args:
        img: (H, W) uint8 image.
        n: the maximum number of keypoints to keep. Always set a limit: without one, the revisit search of
            Q4.1 is very slow.

    Returns:
        (keypoints (M, 2), descriptors (M, 128) float32).
    """
    if n not in _SIFT:
        _SIFT[n] = cv2.SIFT_create(nfeatures=n)
    k, d = _SIFT[n].detectAndCompute(img, None)
    return (np.array([p.pt for p in k], float).reshape(-1, 2),
            (d if d is not None else np.zeros((0, 128))).astype(np.float32))
