"""Part 2: fitting and RANSAC. Run it once Q2.1 and Q2.2 pass their checks:

    python part2.py

Writes to results/part2/:
    q2_1_fits.md      Q2.1: the error of each fit against the ground truth, on the correct matches only
    q2_2_ransac.md    Q2.2: RANSAC on all the matches: error, iterations and inliers per model
"""
import os
import numpy as np
import utils, gt
from features import features, match
from fitting import fit_similarity, fit_homography, ransac


def main():
    out = utils.results_dir('part2')
    COR = utils.flight('corniche')
    cor = utils.load_frames(COR)
    F = {}

    def feat(k):                                         # features of frame k, computed once
        if k not in F:
            F[k] = features(cor[k])
        return F[k]

    def matches(i, j):
        (k1, d1), (k2, d2) = feat(i), feat(j)
        mm = match(d1, d2)
        return k1[mm[:, 0]], k2[mm[:, 1]]

    # Q2.1: fit to the correct matches only (those the ground truth confirms), and compare with the truth.
    # The error of a fit is how far it puts the frame's four corners and center from where the truth puts them.
    rows = []
    for i in (10, 20, 30, 40, 50):
        p, q = matches(i, i + 1)
        G = gt.gt_H(COR, i, i + 1)
        ok = gt.count_correct(p, q, G, 3.0)
        p, q = p[ok], q[ok]
        rows.append([f'{i} -> {i + 1}', len(p),
                     gt.reprojection_error(fit_similarity(p, q), G),
                     gt.reprojection_error(fit_homography(p, q), G),
                     gt.reprojection_error(fit_homography(p, q, normalize=False), G)])
    rows.append(['mean', '', *[float(np.mean([r[c] for r in rows])) for c in (2, 3, 4)]])
    print('Q2.1  error of each fit against the ground truth (px)')
    utils.table(f'{out}/q2_1_fits.md', ['frames', 'correct matches', 'similarity (px)', 'homography (px)',
                                        'homography, not normalized (px)'], rows)

    # Q2.2: RANSAC on every match, outliers included, over 20 pairs
    res = {'similarity': [], 'homography': []}
    for i in range(20):
        p, q = matches(i, i + 1)
        G = gt.gt_H(COR, i, i + 1)
        for model, fit, s in (('similarity', fit_similarity, 2), ('homography', fit_homography, 4)):
            T, inl, it = ransac(p, q, fit, s)
            res[model].append((gt.reprojection_error(T, G) if T is not None else np.nan, it,
                               int(inl.sum()) if inl is not None else 0, len(p)))
    rows = []
    for model, v in res.items():
        v = np.array(v, float)
        rows.append([model, float(np.nanmean(v[:, 0])), float(v[:, 1].mean()), float(v[:, 2].mean()),
                     f'{100 * (1 - v[:, 2].sum() / v[:, 3].sum()):.0f}'])
    print('\nQ2.2  RANSAC over 20 pairs of consecutive corniche frames (means)')
    utils.table(f'{out}/q2_2_ransac.md', ['model', 'error (px)', 'iterations', 'inliers', 'outliers (% of matches)'], rows)
    print(f'\nTables in {os.path.relpath(out)}')


if __name__ == '__main__':
    utils.run(main)
