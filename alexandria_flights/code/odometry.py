"""Q3.1: chaining the flight.

You fill in the block marked TODO. Everything else already runs.
Check your work with:  python check.py 3.1
"""
import numpy as np, cv2
from fitting import estimate


def chain(F, model='similarity', min_inliers=15):
    """Chain the transforms between consecutive frames into one path.

    Args:
        F: features of every frame, [(keypoints, descriptors), ...].
        model: 'similarity' or 'homography'.
        min_inliers: a step with fewer inliers than this is lost.

    Returns:
        T: list of N (3, 3) transforms; T[k] takes frame-k pixels to frame-0 pixels, and T[0] is the identity.
        steps: list of N - 1 transforms; steps[k] takes frame k to frame k + 1, or is None if that step was lost.
        lost: list of the indices k of the lost steps.
        n_inliers: list of N - 1 inlier counts, 0 for a lost step.
    """
    T, steps, lost, n_inliers = [np.eye(3)], [], [], []
    last = np.eye(3)                                    # the motion of the last step that was not lost
    for k in range(1, len(F)):
        H, n, _ = estimate(F[k - 1], F[k], model)       # H takes frame k - 1 to frame k
        # ------------------------------------------------------------ TODO 3.1 -----
        # A step is lost when H is None or n < min_inliers: record k - 1 in `lost`, None in `steps` and 0 in
        # `n_inliers`, and use `last` as its motion. Otherwise record H and n, and remember H in `last`.
        # Then append T_k = T_{k-1} M^-1 to T, where M is the step's motion: H, or `last` for a lost step.
        raise NotImplementedError('TODO 3.1 in odometry.py, chain(): one step of the chain')
        # ------------------------------------------------------------ end TODO -----
    return T, steps, lost, n_inliers
