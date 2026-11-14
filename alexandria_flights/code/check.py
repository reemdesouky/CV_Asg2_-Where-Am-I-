"""Sanity checks for each question, against the synthetic flights' ground truth.

    python check.py 1.1           one question (1.1, 1.2, 2.1, 2.2, 3.1, 3.3, 4.1 or 4.2)
    python check.py               every question, in order
    python check.py 3.1 --sift    use OpenCV's SIFT instead of your Part 1 features (in 2.2, 3.1 and 4.1)

Run a question's check whenever you change its code. The checks catch the usual mistakes early: wrong
shapes, a transform the wrong way round, a ratio test that keeps everything. Passing does not prove your
code is right: the TAs also read your code and rerun the part scripts.
"""
import sys, time, traceback
import numpy as np, cv2
import utils, gt
from utils import apply

COR, LOOP = utils.flight('corniche'), utils.flight('loop')
SIFT = False                    # set by --sift
_results = []


def report(name, ok, detail=''):
    _results.append(bool(ok))
    print(f'  {"PASS" if ok else "FAIL"}  {name:55s} {detail}')


_cache = {}
def frames(flight, color=False):
    if (flight, color) not in _cache:
        _cache[flight, color] = utils.load_frames(flight, color)
    return _cache[flight, color]


def front_end():
    """The features the checks of 2.2, 3.1 and 4.1 use: yours, or OpenCV's SIFT with --sift."""
    from features import features, features_sift
    return (lambda img, oriented=True: features_sift(img)) if SIFT else features


def random_similarity(rng):
    a, s = rng.uniform(0, 2 * np.pi), rng.uniform(0.8, 1.2)
    return np.array([[s * np.cos(a), -s * np.sin(a), rng.uniform(-50, 50)],
                     [s * np.sin(a), s * np.cos(a), rng.uniform(-50, 50)], [0, 0, 1.0]])


# ------------------------------------------------------------------ Part 1
def check_1_1():
    from features import harris_response, detect
    sq = np.zeros((101, 101), np.float32); sq[30:71, 30:71] = 255          # a white square on black
    R = harris_response(sq)
    report('harris_response: shape', R.shape == sq.shape, str(R.shape))
    report('harris_response: corner > 0 > edge', R[30, 30] > 0 > R[50, 30] and R[30, 30] > abs(R[50, 50]),
           f'corner {R[30, 30]:.3g}, edge {R[50, 30]:.3g}, flat {R[50, 50]:.3g}')
    R2 = harris_response(2 * sq)
    rc = R2[30, 30] / R[30, 30] if R[30, 30] else 0
    re = R2[50, 30] / R[50, 30] if R[50, 30] else 0
    report('harris_response: twice the contrast, 16 times R', abs(rc - 16) < 0.5 and abs(re - 16) < 0.5,
           f'corner x{rc:.1f}, edge x{re:.1f} (det and trace^2 both grow with contrast^4)')
    img = frames(COR)[20]
    kps = detect(img)
    t = time.time(); detect(img); dt = time.time() - t
    cells = len({(int(y * 6 // img.shape[0]), int(x * 8 // img.shape[1])) for x, y in kps})
    report('detect: between 100 and 1200 corners', 100 <= len(kps) <= 1200, f'{len(kps)} corners')
    report('detect: corners all over the land', cells >= 18, f'{cells} of 48 grid cells used')
    R = harris_response(img.astype(np.float32))
    xi, yi = kps[:, 0].astype(int), kps[:, 1].astype(int)
    local_max = R[yi, xi] >= cv2.dilate(R, np.ones((3, 3), np.uint8))[yi, xi]
    report('detect: every corner is a 3 x 3 maximum of R', local_max.all(), f'{(~local_max).sum()} are not')
    dup = len(kps) - len({(int(x), int(y)) for x, y in kps})
    report('detect: no corner twice', dup == 0, f'{dup} repeated')
    report('detect: under 0.3 s per frame', dt < 0.3, f'{dt:.2f} s (part3.py runs it 482 times)')


def check_1_2():
    from features import describe, detect, features, match
    img = frames(COR)[20]
    k, d = describe(img, detect(img))
    report('describe: shapes', k.ndim == 2 and k.shape[1] == 2 and d.shape == (len(k), 64),
           f'keypoints {k.shape}, descriptors {d.shape}')
    report('describe: zero mean, unit std', len(d) and np.allclose(d.mean(1), 0, atol=1e-3) and np.allclose(d.std(1), 1, atol=1e-2),
           f'largest |mean| {np.abs(d.mean(1)).max():.2g}, std {d.std(1).min():.3f} to {d.std(1).max():.3f}' if len(d) else 'no descriptors')
    mm = match(d, d + 0.01)
    report('match: a frame against itself', len(mm) >= 0.9 * len(d) and np.all(mm[:, 0] == mm[:, 1]),
           f'{len(mm)} of {len(d)} matched to themselves')
    rng = np.random.default_rng(0)
    r = rng.normal(size=(200, 64)).astype(np.float32)
    mm = match(r, rng.normal(size=(300, 64)).astype(np.float32))
    report('match: the ratio test rejects random descriptors', len(mm) < 20, f'{len(mm)} of 200 kept')
    for oriented, ang, need in [(False, 0, 0.9), (True, 0, 0.9), (True, 90, 0.8), (True, 150, 0.8)]:
        rot, M = gt.rotate(img, ang)
        (k1, d1), (k2, d2) = features(img, oriented), features(rot, oriented)
        mm = match(d1, d2)
        prec = gt.count_correct(k1[mm[:, 0]], k2[mm[:, 1]], M).mean() if len(mm) else 0
        name = f'{"oriented" if oriented else "raw"} patch, frame rotated {ang} deg'
        report(f'match: {name}', prec >= need and len(mm) >= 50, f'{len(mm)} matches, {prec:.0%} correct')


# ------------------------------------------------------------------ Part 2
def check_2_1():
    from fitting import fit_similarity, fit_homography
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 640, (50, 2))
    S = random_similarity(rng)
    e = gt.reprojection_error(fit_similarity(p, apply(S, p)), S)
    report('fit_similarity: exact on clean points', e < 1e-6, f'{e:.2g} px')
    H = S.copy(); H[2, :2] = [3e-4, -2e-4]
    for norm in (True, False):
        e = gt.reprojection_error(fit_homography(p, apply(H, p), normalize=norm), H)
        report(f'fit_homography: exact on clean points, {"normalized" if norm else "not normalized"}', e < 1e-2, f'{e:.2g} px')
    p4 = p[:4]
    e = gt.reprojection_error(fit_homography(p4, apply(H, p4)), H)
    report('fit_homography: from exactly 4 points', e < 1e-2, f'{e:.2g} px')
    rng = np.random.default_rng(3)                          # a frame-sized step with 0.5 px of noise
    a = np.radians(2)
    Hn = np.array([[np.cos(a), -np.sin(a), 20], [np.sin(a), np.cos(a), -60], [1e-5, -2e-5, 1]])
    pn = np.round(rng.uniform(0, 1, (100, 2)) * [640, 480])
    qn = apply(Hn, pn) + rng.normal(0, 0.5, pn.shape)
    e1 = gt.reprojection_error(fit_homography(pn, qn), Hn)
    report('fit_homography: 100 noisy points, normalized', e1 < 0.6, f'{e1:.2f} px')


def check_2_2():
    from fitting import fit_similarity, ransac, estimate
    features = front_end()
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 640, (60, 2)); S = random_similarity(rng)
    q = apply(S, p) + rng.normal(0, 0.5, p.shape)
    bad = rng.random(len(p)) < 0.5
    q[bad] = rng.uniform(0, 640, (bad.sum(), 2))
    T, inl, it = ransac(p, q, fit_similarity, 2)
    ok = T is not None and inl is not None
    report('ransac: similarity at 50% outliers', ok and gt.reprojection_error(T, S) < 2 and (inl == ~bad).mean() > 0.95,
           f'{gt.reprojection_error(T, S):.2f} px, {(inl == ~bad).mean():.0%} labeled correctly' if ok else 'returned None')
    report('ransac: stops early at 50% outliers', ok and 5 <= it < 200, f'{it} iterations' if ok else '')
    q2 = apply(S, p) + rng.normal(0, 0.5, p.shape)
    bad2 = rng.random(len(p)) < 0.25
    q2[bad2] = rng.uniform(0, 640, (bad2.sum(), 2))
    T2, inl2, it2 = ransac(p, q2, fit_similarity, 2)
    report('ransac: stops sooner at 25% outliers', T2 is not None and it2 < 30,
           f'{it2} iterations (the formula needs about 6)')
    img = frames(COR)
    F1, F2 = features(img[20]), features(img[21])
    G = gt.gt_H(COR, 20, 21)
    for model in ('similarity', 'homography'):
        T, n, _ = estimate(F1, F2, model)
        e = gt.reprojection_error(T, G) if T is not None else np.inf
        report(f'estimate: corniche 20 -> 21, {model}', e < 2 and n > 50, f'{e:.2f} px, {n} inliers')


# ------------------------------------------------------------------ Part 3
def check_3_1():
    from odometry import chain
    features = front_end()
    F = [features(i) for i in frames(COR)[:20]]
    T, steps, lost, n_inl = chain(F)
    report('chain: output lengths', len(T) == 20 and len(steps) == 19 and len(n_inl) == 19,
           f'{len(T)} transforms, {len(steps)} steps, {len(n_inl)} inlier counts')
    report('chain: T[0] is the identity', np.allclose(T[0], np.eye(3)), f'largest difference {np.abs(T[0] - np.eye(3)).max():.2g}')
    unit = lambda M: M / M[2, 2]
    diff = max([np.abs(unit(T[k + 1]) - unit(T[k] @ np.linalg.inv(steps[k]))).max()
                for k in range(len(steps)) if steps[k] is not None] or [np.inf])
    report('chain: T[k + 1] = T[k] @ inv(steps[k])', diff < 1e-6, f'largest difference {diff:.2g}')
    e = gt.path_errors(COR, T).mean()
    report('chain: first 20 frames of corniche', e < 150, f'{e:.0f} m mean error')
    empty = (np.zeros((0, 2)), np.zeros((0, F[0][1].shape[1]), np.float32))
    T2, steps2, lost2, n2 = chain(F[:5] + [empty] + F[6:10])
    report('chain: a frame without features is lost', lost2 == [4, 5] and steps2[4] is None and len(T2) == 10,
           f'lost steps {lost2}')
    m4, m3 = np.linalg.inv(T2[4]) @ T2[5], np.linalg.inv(T2[3]) @ T2[4]
    report('chain: a lost step reuses the last motion', np.allclose(m4, m3), f'largest difference {np.abs(m4 - m3).max():.2g}')


def check_3_3():
    from mapping import make_map
    col = frames(COR, color=True)[:8]
    out, S = make_map(col, gt.gt_path(COR)[:8])
    report('make_map: a color uint8 image', out.dtype == np.uint8 and out.ndim == 3, f'{out.shape}')
    h, w = col[0].shape[:2]
    x, y = apply(S, [[w / 2, h / 2]])[0].astype(int)
    a = cv2.GaussianBlur(cv2.cvtColor(out[y - 40:y + 40, x - 40:x + 40], cv2.COLOR_BGR2GRAY).astype(float), (0, 0), 2)
    b = cv2.GaussianBlur(cv2.cvtColor(col[0][h // 2 - 40:h // 2 + 40, w // 2 - 40:w // 2 + 40], cv2.COLOR_BGR2GRAY).astype(float), (0, 0), 2)
    ncc = ((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std() + 1e-9) if a.shape == b.shape else 0
    report('make_map: frame 0 lands where S says', ncc > 0.8, f'correlation {ncc:.2f} with frame 0')
    report('make_map: white where no frame lands', np.all(out[0, 0] >= 250), f'corner pixel {out[0, 0]}')


# ------------------------------------------------------------------ Part 4
def check_4_1():
    from loop_closure import revisits
    features = front_end()
    img = frames(LOOP)
    idx = list(range(0, 18)) + list(range(120, 138))        # the start and the end of loop overlap
    F = [features(img[k]) for k in idx]
    L = revisits(F, every=3, gap=12)
    good = [np.linalg.norm(apply(H, [[320, 240]]) - apply(gt.gt_H(LOOP, idx[i], idx[j]), [[320, 240]])) < 20
            for i, j, H, n in L]
    report('revisits: the end of loop finds its start', len(L) >= 4, f'{len(L)} revisits')
    report('revisits: j - i >= gap, H from frame i to frame j', len(L) > 0 and all(good) and all(j - i >= 12 for i, j, _, _ in L),
           f'{sum(good)} of {len(L)} within 20 px of the truth')
    report('revisits: one partner per keyframe j', len({j for _, j, _, _ in L}) == len(L),
           f'{len(L)} revisits, {len({j for _, j, _, _ in L})} different j')


def check_4_2():
    from loop_closure import measurement_rows, correct
    rng = np.random.default_rng(2)
    corners = np.array([[0, 0], [640, 0], [640, 480], [0, 480]], float)
    N = 6
    Ts = [np.eye(3)] + [random_similarity(rng) for _ in range(N - 1)]
    x = np.concatenate([[T[0, 0], T[1, 0], T[0, 2], T[1, 2]] for T in Ts])
    H = np.linalg.inv(Ts[3]) @ Ts[1]                        # frame 1 -> frame 3, consistent with Ts
    rows = measurement_rows(1, 3, H, N, corners)
    report('measurement_rows: shape', rows.shape == (8, 4 * N), f'{rows.shape}')
    report('measurement_rows: the true poses satisfy them', np.abs(rows @ x).max() < 1e-6, f'max |row . z| = {np.abs(rows @ x).max():.2g}')
    report('measurement_rows: only frames 1 and 3 appear', np.all(rows[:, :4] == 0) and np.all(rows[:, 8:12] == 0),
           f'{np.count_nonzero(rows[:, :4]) + np.count_nonzero(rows[:, 8:12])} nonzero coefficients in the columns of frames 0 and 2')
    steps = [np.linalg.inv(Ts[k + 1]) @ Ts[k] for k in range(N - 1)]
    Tc = correct(N, steps, [])
    d = max(np.abs(a - b).max() for a, b in zip(Tc, Ts))
    report('correct: exact steps give back the poses', d < 1e-6, f'largest difference {d:.2g}')


QUESTIONS = {'1.1': check_1_1, '1.2': check_1_2, '2.1': check_2_1, '2.2': check_2_2,
             '3.1': check_3_1, '3.3': check_3_3, '4.1': check_4_1, '4.2': check_4_2}

if __name__ == '__main__':
    args = sys.argv[1:]
    SIFT = '--sift' in args
    which = [a for a in args if a != '--sift'] or list(QUESTIONS)
    if SIFT:
        print("Using OpenCV's SIFT instead of your Part 1 features in 2.2, 3.1 and 4.1.")
    not_done = []
    for q in which:
        if q not in QUESTIONS:
            sys.exit(f'There is no check for {q}. The questions with checks are {", ".join(QUESTIONS)}.')
        print(f'\nQ{q}')
        t = time.time()
        try:
            QUESTIONS[q]()
        except NotImplementedError as e:
            print(f'  ----  not done yet: {e}')
            not_done.append(q)
        except Exception:
            _results.append(False)
            print('  FAIL  crashed:\n        ' + traceback.format_exc().strip().replace('\n', '\n        '))
        print(f'        ({time.time() - t:.1f} s)')
    summary = f'\n{sum(_results)} of {len(_results)} checks passed' if _results else '\nNo checks ran'
    print(summary + (f'; not done yet: {", ".join(not_done)}' if not_done else ''))
