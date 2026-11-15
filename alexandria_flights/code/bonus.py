import os
import numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import utils, gt, viz
from utils import apply
from features import features, features_sift, match
from fitting import fit_homography, ransac
from odometry import chain

NORM = 600.0


#  the distortion model
def distort(p, kappa, c):
    """Ideal points -> where they appear: pd = c + (p - c)(1 + kappa |p - c|^2 / 600^2). p: (..., 2)."""
    d = p - c
    return c + d * (1 + kappa * (d ** 2).sum(-1, keepdims=True) / NORM ** 2)


def undistort_points(pd, kappa, c, iters=20):
    """Observed points -> ideal points, by fixed-point iteration on u = d / (1 + kappa |u|^2 / 600^2)."""
    d = pd - c
    u = d.copy()
    for _ in range(iters):
        u = d / (1 + kappa * (u ** 2).sum(-1, keepdims=True) / NORM ** 2)
    return c + u


def undistort_image(img, kappa):
    """The image an ideal camera would have taken. Each ideal pixel p is read from where it appears, distort(p)."""
    h, w = img.shape[:2]
    c = np.array([w / 2, h / 2])
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    src = distort(np.stack([xx, yy], -1), kappa, c).astype(np.float32)
    return cv2.remap(img, src[..., 0], src[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# finding kappa from the matches
def collect_matches(imgs, index_pairs, feat):
    """Matches (ratio test only, outliers included) for pairs of frames of one flight."""
    cache, out = {}, []
    def f(k):
        if k not in cache:
            cache[k] = feat(imgs[k])
        return cache[k]
    for i, j in index_pairs:
        (k1, d1), (k2, d2) = f(i), f(j)
        mm = match(d1, d2)
        if len(mm) >= 20:
            out.append((k1[mm[:, 0]], k2[mm[:, 1]]))
    return out


def select_inliers(pairs, kappa, c, thresh):
    """RANSAC with the homography on the points undistorted with `kappa`; keep each pair's inliers."""
    sel = []
    for p, q in pairs:
        pu, qu = undistort_points(p, kappa, c), undistort_points(q, kappa, c)
        T, inl, _ = ransac(pu, qu, fit_homography, 4, thresh)
        if T is not None and inl.sum() >= 30:
            sel.append((p[inl], q[inl]))
    return sel


def cost(sel, kappa, c):
    """How badly one homography per pair explains the undistorted inliers: truncated mean squared error (px^2)."""
    e = []
    for p, q in sel:
        pu, qu = undistort_points(p, kappa, c), undistort_points(q, kappa, c)
        H = fit_homography(pu, qu)
        e.append(np.minimum(np.linalg.norm(apply(H, pu) - qu, axis=1), 3.0) ** 2)
    return float(np.concatenate(e).mean())


def minimize_kappa(sel, c, lo=-0.4, hi=0.4):
    """Grid search over [lo, hi], then ternary search around the best grid value."""
    grid = np.linspace(lo, hi, 81)
    costs = [cost(sel, k, c) for k in grid]
    b = int(np.argmin(costs))
    a, z = grid[max(b - 1, 0)], grid[min(b + 1, len(grid) - 1)]
    for _ in range(25):
        m1, m2 = a + (z - a) / 3, z - (z - a) / 3
        if cost(sel, m1, c) < cost(sel, m2, c):
            z = m2
        else:
            a = m1
    return (a + z) / 2, grid, np.array(costs)


def fit_kappa(imgs, feat, c, step=6):
    """kappa of a flight from its own matches. Two rounds: inliers chosen loosely at kappa = 0 (the distortion
    bends the matches a few pixels), then again, tightly, at the kappa of the first round."""
    n = len(imgs)
    index_pairs = [(i, i + d) for i in range(0, n - 3, step) for d in (1, 3)]
    pairs = collect_matches(imgs, index_pairs, feat)
    sel = select_inliers(pairs, 0.0, c, 6.0)
    k1, _, _ = minimize_kappa(sel, c)
    sel = select_inliers(pairs, k1, c, 2.0)
    k2, grid, costs = minimize_kappa(sel, c)
    print(f'      round 1 kappa = {k1:+.4f}, round 2 kappa = {k2:+.4f}  ({len(sel)} pairs, '
          f'{sum(len(p) for p, _ in sel)} inlier matches)')
    return k2, len(sel), grid, costs


def main():
    out = utils.results_dir('bonus')
    feat = features
    COR, LOOP = utils.flight('corniche'), utils.flight('loop')
    imgs = {'corniche': utils.load_frames(COR), 'loop': utils.load_frames(LOOP)}
    paths = {'corniche': COR, 'loop': LOOP}
    c = np.array([640 / 2, 480 / 2])

    #  kappa from the matches
    kappa, rows, curves = {}, [], {}
    for fl in ('loop', 'corniche'):
        print(f'... estimating kappa on {fl}', flush=True)
        kappa[fl], npairs, grid, costs = fit_kappa(imgs[fl], feat, c)
        curves[fl] = (grid, costs)
        rows.append([fl, f'{kappa[fl]:+.4f}', npairs])
    print('\nkappa found from the matches')
    utils.table(f'{out}/bonus_kappa.md', ['flight', 'kappa', 'pairs used'], rows)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for fl, (grid, costs) in curves.items():
        ax.plot(grid, costs, label=f'{fl} (kappa = {kappa[fl]:+.3f})')
        ax.axvline(kappa[fl], color=ax.lines[-1].get_color(), ls=':')
    ax.set_xlabel('kappa'); ax.set_ylabel('truncated mean squared error (px$^2$)'); ax.grid(alpha=.3); ax.legend()
    ax.set_title('Does one homography per pair explain the matches?')
    fig.savefig(f'{out}/bonus_kappa_cost.png', dpi=120, bbox_inches='tight', facecolor='white'); plt.close(fig)

    #  undistort loop (corniche is the control: its camera is fine, so its frames stay as they are)
    k = kappa['loop']
    print(f'\n... undistorting loop with kappa = {k:+.4f}', flush=True)
    und = [undistort_image(im, k) for im in imgs['loop']]
    cv2.imwrite(f'{out}/bonus_undistort_example.png',
                np.hstack([imgs['loop'][20], und[20], cv2.absdiff(imgs['loop'][20], und[20])]))

    # Q3.2 again
    F = {('corniche', 'original'): [feat(im) for im in imgs['corniche']],
         ('loop', 'original'): [feat(im) for im in imgs['loop']],
         ('loop', 'undistorted'): [feat(im) for im in und]}
    rows, runs = [], {}
    for (fl, camera), Fl in F.items():
        for model in ('similarity', 'homography'):
            print(f'... chaining {fl}, {camera}, {model}', flush=True)
            T, _, lost, _ = chain(Fl, model)
            e = gt.path_errors(paths[fl], T)
            rows.append([fl, camera, model, len(lost), float(e.mean()), float(e[-1])])
            if fl == 'loop':
                runs[f'{model}, {camera}'] = T
    print('\nQ3.2 repeated: similarity against homography')
    utils.table(f'{out}/bonus_models.md', ['flight', 'frames', 'model', 'lost steps', 'mean error (m)',
                                           'final error (m)'], rows)
    viz.plot_paths(LOOP, runs, f'{out}/bonus_paths_loop.png')

    # match errors against distance from the center, scored with the ground truth (the truth is an ideal camera,
    # so after undistorting the error should no longer grow with the radius)
    for tag, key in (('before', 'original'), ('after', 'undistorted')):
        pairs = []
        for fl, i, j in (('corniche', 20, 21), ('loop', 20, 21), ('loop', 3, 132)):
            Fl = F[fl, key if fl == 'loop' else 'original']
            (k1, d1), (k2, d2) = Fl[i], Fl[j]
            mm = match(d1, d2)
            pairs.append((f'{fl}, frames {i} and {j}, {key if fl == "loop" else "original"}',
                          k1[mm[:, 0]], k2[mm[:, 1]], gt.gt_H(paths[fl], i, j)))
        viz.plot_error_vs_radius(pairs, f'{out}/bonus_error_vs_radius_{tag}.png')
    print(f'\nEverything is in {os.path.relpath(out)}')


if __name__ == '__main__':
    utils.run(main)