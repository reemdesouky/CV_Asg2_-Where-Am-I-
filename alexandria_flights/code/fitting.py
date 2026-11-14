"""Part 2: fitting transforms (Q2.1) and RANSAC (Q2.2).

You fill in the blocks marked TODO. Everything else already runs.
Check your work with:  python check.py 2.1   and   python check.py 2.2
"""
import numpy as np, cv2
from utils import apply
from features import match


# ===================================================================== Q2.1 fitting
def fit_similarity(p, q):
    """Least-squares similarity (rotation, uniform scale, translation) taking p to q:
    x' = a x - b y + tx,  y' = b x + a y + ty.

    Args:
        p: (N, 2) points, N >= 2.
        q: (N, 2) matching points.

    Returns:
        (3, 3) transform [[a, -b, tx], [b, a, ty], [0, 0, 1]].
    """
    n = len(p)
    A, rhs = np.zeros((2 * n, 4)), np.zeros(2 * n)       # unknowns (a, b, tx, ty)
    # ---------------------------------------------------------------- TODO 2.1a ----
    # Fill A and rhs: two rows per match, one for x' and one for y'.
    raise NotImplementedError('TODO 2.1a in fitting.py, fit_similarity(): build A and rhs')
    # ---------------------------------------------------------------- end TODO -----
    a, b, tx, ty = np.linalg.lstsq(A, rhs, rcond=None)[0]
    return np.array([[a, -b, tx], [b, a, ty], [0, 0, 1.0]])


def fit_homography(p, q, normalize=True):
    """Homography taking p to q by the direct linear transform (DLT), solved with the SVD.

    Args:
        p: (N, 2) points, N >= 4.
        q: (N, 2) matching points.
        normalize: apply Hartley normalization to both point sets first.

    Returns:
        (3, 3) homography with H[2, 2] = 1.
    """
    Tp, Tq = np.eye(3), np.eye(3)
    if normalize:
        # ------------------------------------------------------------ TODO 2.1b ----
        # Tp, Tq: (3, 3) similarities that move each point set to zero mean and an average distance of
        # sqrt(2) from the origin (Hartley, 1997).
        raise NotImplementedError('TODO 2.1b in fitting.py, fit_homography(): Hartley normalization')
        # ------------------------------------------------------------ end TODO -----
    pn, qn = apply(Tp, p), apply(Tq, q)
    # ---------------------------------------------------------------- TODO 2.1c ----
    # Hn: the homography taking pn to qn. Build the 2N x 9 DLT matrix (two rows per match), take the right
    # singular vector of its smallest singular value (np.linalg.svd), and reshape it to 3 x 3.
    raise NotImplementedError('TODO 2.1c in fitting.py, fit_homography(): the DLT')
    # ---------------------------------------------------------------- end TODO -----
    H = np.linalg.inv(Tq) @ Hn @ Tp                       # undo the normalization
    return H / H[2, 2]


# ===================================================================== Q2.2 RANSAC
def plausible(T, homography):
    """Whether a camera flying over the ground could produce this transform.

    Args:
        T: (3, 3) transform.
        homography: True to also check the perspective terms.

    Returns:
        False if T is not finite, if its scale (sqrt|det| of the top-left 2 x 2 block) is outside 0.6 to 1.6,
        or, for a homography, if a perspective term |T[2, 0]| or |T[2, 1]| is above 0.002. True otherwise.
    """
    if not np.all(np.isfinite(T)) or abs(T[2, 2]) < 1e-12:
        return False
    T = T / T[2, 2]
    s = np.sqrt(abs(np.linalg.det(T[:2, :2])))
    if not 0.6 < s < 1.6:
        return False
    return not homography or np.abs(T[2, :2]).max() <= 2e-3


def ransac(p, q, fit, sample, thresh=3.0, p_success=0.99, max_iter=5000, seed=0):
    """RANSAC for any model.

    Args:
        p: (N, 2) points.
        q: (N, 2) matching points.
        fit: fit_similarity or fit_homography.
        sample: points per sample, 2 for a similarity and 4 for a homography.
        thresh: inlier threshold in pixels.
        p_success: probability of drawing at least one all-inlier sample.
        max_iter: upper bound on the iterations.
        seed: seed of the random generator.

    Returns:
        T: (3, 3) transform refitted on all its inliers, or None if no plausible fit was found.
        inliers: (N,) boolean mask, or None.
        iterations: how many iterations ran.
    """
    rng = np.random.default_rng(seed)
    n = len(p)
    if n < sample + 2:
        return None, None, 0
    best_inliers, needed, it = None, max_iter, 0
    while it < min(needed, max_iter):
        it += 1
        idx = rng.choice(n, sample, replace=False)
        try:
            T = fit(p[idx], q[idx])
        except np.linalg.LinAlgError:
            continue
        if not plausible(T, homography=(sample == 4)):
            continue
        # ------------------------------------------------------------ TODO 2.2 -----
        # inliers: the matches i for which T maps p[i] to within thresh pixels of q[i].
        # If there are more of them than in best_inliers (and at least `sample`), keep them in best_inliers,
        # and set `needed`, the total number of iterations, with the formula of Lecture 8 for the new inlier
        # ratio and p_success.
        raise NotImplementedError('TODO 2.2 in fitting.py, ransac(): score the hypothesis')
        # ------------------------------------------------------------ end TODO -----
    if best_inliers is None:
        return None, None, it
    T = fit(p[best_inliers], q[best_inliers])                   # refit on all the inliers
    if not plausible(T, homography=(sample == 4)):
        return None, None, it
    return T / T[2, 2], best_inliers, it


def estimate(F1, F2, model='similarity', thresh=3.0):
    """Transform between two frames from their features: match(), then ransac().

    Args:
        F1: (keypoints, descriptors) of the first frame.
        F2: (keypoints, descriptors) of the second frame.
        model: 'similarity' or 'homography'.
        thresh: RANSAC inlier threshold in pixels.

    Returns:
        T: (3, 3) transform taking pixels of the first frame to pixels of the second, or None.
        n_inliers: number of inliers, 0 if T is None.
        inlier_points: (p, q), the inlier matches as two (n_inliers, 2) arrays, or None.
    """
    (k1, d1), (k2, d2) = F1, F2
    mm = match(d1, d2)
    if len(mm) < 8:
        return None, 0, None
    p, q = k1[mm[:, 0]], k2[mm[:, 1]]
    fit, s = (fit_similarity, 2) if model == 'similarity' else (fit_homography, 4)
    T, inl, _ = ransac(p, q, fit, s, thresh)
    if T is None:
        return None, 0, None
    return T, int(inl.sum()), (p[inl], q[inl])
