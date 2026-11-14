"""Part 4: finding revisits (Q4.1) and correcting the whole flight (Q4.2).

You fill in the blocks marked TODO. Everything else already runs.
Check your work with:  python check.py 4.1   and   python check.py 4.2
"""
import numpy as np, cv2
from utils import apply
from fitting import estimate


# ===================================================================== Q4.1 revisits
def revisits(F, every=3, gap=30, min_inliers=40):
    """Find pairs of keyframes, far apart in time, that see the same ground.

    Args:
        F: features of every frame, [(keypoints, descriptors), ...].
        every: spacing of the keyframes, which are frames 0, `every`, 2 * `every`, ...
        gap: the two frames of a revisit are at least this many frames apart.
        min_inliers: a revisit needs at least this many inliers.

    Returns:
        List of revisits (i, j, H, n_inliers), with i < j and H the (3, 3) similarity taking frame i to frame j.
    """
    keyframes = list(range(0, len(F), every))
    found = []
    for j in keyframes:
        best = None
        # ------------------------------------------------------------ TODO 4.1 -----
        # Try every keyframe i at least `gap` frames before j: estimate(F[i], F[j], 'similarity').
        # Accept the pair when estimate() returns a transform with at least min_inliers inliers, and keep in
        # `best` the accepted (i, j, H, n_inliers) with the most inliers.
        raise NotImplementedError('TODO 4.1 in loop_closure.py, revisits(): the best partner of keyframe j')
        # ------------------------------------------------------------ end TODO -----
        if best is not None:
            found.append(best)
    return found


# ===================================================================== Q4.2 the pose graph
def measurement_rows(k, l, H, N, corners):
    """The equations that one measured transform adds to the pose graph.

    The unknowns are stacked in one vector z of length 4N: frame m owns z[4m:4m + 4] = (a_m, b_m, tx_m, ty_m),
    and its similarity is T_m(p) = [[a_m, -b_m], [b_m, a_m]] p + [tx_m, ty_m]. The measurement says
    T_k(p) = T_l(H(p)) for every corner p of frame k: two equations per corner, linear in z, each of the form
    row . z = 0.

    Args:
        k: index of the first frame.
        l: index of the second frame.
        H: (3, 3) measured transform taking frame-k pixels to frame-l pixels.
        N: number of frames.
        corners: (4, 2) the corners of a frame.

    Returns:
        (8, 4N) array, one row per equation: rows 2c and 2c + 1 are the x and y equations of corner c.
    """
    rows = np.zeros((8, 4 * N))
    for c, ((x, y), (u, v)) in enumerate(zip(corners, apply(H, corners))):    # p = (x, y), H(p) = (u, v)
        rx, ry = rows[2 * c], rows[2 * c + 1]
        # ------------------------------------------------------------ TODO 4.2 -----
        # Write T_k(p) = T_l(H(p)) at p = (x, y) as two equations, one for each coordinate. Move everything
        # to one side, so that each reads row . z = 0, and write the coefficients into rx and ry, in the
        # columns of frames k and l.
        raise NotImplementedError('TODO 4.2 in loop_closure.py, measurement_rows(): the coefficients')
        # ------------------------------------------------------------ end TODO -----
    return rows


def correct(N, steps, revs, weak=0.02, frame_size=(640, 480)):
    """Correct the whole flight with one linear least-squares solve.

    Every step and every revisit adds its measurement_rows(). A lost step has no measured H, so it adds the
    rows of the last good step's H, multiplied by `weak` so that they count much less. Frame 0 is fixed to
    the identity (a = 1, b = 0, tx = ty = 0): its four columns move to the right-hand side, and
    np.linalg.lstsq solves for every other frame.

    Args:
        N: number of frames.
        steps: the steps of chain() with model='similarity'; steps[k] is None for a lost step.
        revs: the revisits from revisits().
        weak: weight of the equations across lost steps.
        frame_size: (width, height) of the frames.

    Returns:
        List of N (3, 3) transforms taking frame-k pixels to frame-0 pixels.
    """
    w, h = frame_size
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
    blocks, last = [], np.eye(3)
    for k, H in enumerate(steps):                       # step k: frame k -> frame k + 1
        if H is not None:
            blocks.append(measurement_rows(k, k + 1, H, N, corners)); last = H
        else:
            blocks.append(weak * measurement_rows(k, k + 1, last, N, corners))
    for i, j, H, _ in revs:
        blocks.append(measurement_rows(i, j, H, N, corners))
    A = np.vstack(blocks)
    rhs = -A[:, :4] @ np.array([1.0, 0, 0, 0])           # frame 0 is the identity: a=1, b=0, tx=0, ty=0
    x = np.r_[1.0, 0, 0, 0, np.linalg.lstsq(A[:, 4:], rhs, rcond=None)[0]]
    return [np.array([[a, -b, tx], [b, a, ty], [0, 0, 1.0]]) for a, b, tx, ty in x.reshape(N, 4)]
