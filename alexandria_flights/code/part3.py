"""Part 3: odometry and the map. Run it once Q3.1 and Q3.3 pass their checks:

    python part3.py            (add --sift to use the SIFT fallback instead of your Part 1 features)

Writes to results/part3/:
    q3_1_errors.md                         Q3.1: mean and final error of every synthetic flight
    q3_1_path_<flight>.png                 Q3.1: the path against the truth, and the error against distance
    q3_1_inliers_<flight>.png              Q3.1: the inliers of every step
    q3_2_models.md                         Q3.2: similarity against homography on corniche and loop
    q3_2_paths_loop.png                    Q3.2: both paths on loop, against the truth
    q3_2_error_vs_radius.png               Q3.2: match errors against distance from the image center
    q3_3_map_<flight>.jpg                  Q3.3: the map of every flight
    q3_3_closeup_{loop,lost_at_sea}.jpg    Q3.3: where loop returns to its start, and where lost_at_sea finds the coast
    T_<flight>_<model>.npy                 the chained transforms, frame k -> frame 0
"""
import os, sys
import numpy as np, cv2
import utils, gt, viz
from features import features, features_sift
from odometry import chain
from mapping import make_map


def main():
    out = utils.results_dir('part3')
    feat = features_sift if '--sift' in sys.argv else features
    flights = ['corniche', 'loop', 'lost_at_sea']
    F, T, imgs = {}, {}, {}

    # Q3.1
    rows = []
    for fl in flights:
        print(f'... chaining {fl}', flush=True)
        imgs[fl] = utils.load_frames(utils.flight(fl))
        F[fl] = [feat(im) for im in imgs[fl]]
        T[fl], steps, lost, n_inl = chain(F[fl])
        np.save(f'{out}/T_{fl}_similarity.npy', np.array(T[fl]))
        e = gt.path_errors(utils.flight(fl), T[fl])
        rows.append([fl, len(F[fl]), len(lost), float(e.mean()), float(e[-1])])
        viz.plot_paths(utils.flight(fl), {'similarity': T[fl]}, f'{out}/q3_1_path_{fl}.png')
        viz.plot_inliers(n_inl, f'{out}/q3_1_inliers_{fl}.png', title=f'Inliers per step, {fl}')
    print('Q3.1  chained with similarities')
    utils.table(f'{out}/q3_1_errors.md', ['flight', 'frames', 'lost steps', 'mean error (m)', 'final error (m)'],
                rows)

    # Q3.2: corniche and loop chained with the homography too
    rows, Th = [], {}
    for fl in ('corniche', 'loop'):
        print(f'... chaining {fl} with the homography', flush=True)
        Th[fl], _, _, _ = chain(F[fl], 'homography')
        np.save(f'{out}/T_{fl}_homography.npy', np.array(Th[fl]))
        for model, Ts in (('similarity', T[fl]), ('homography', Th[fl])):
            e = gt.path_errors(utils.flight(fl), Ts)
            rows.append([fl, model, float(e.mean()), float(e[-1])])
    print('\nQ3.2  similarity against homography')
    utils.table(f'{out}/q3_2_models.md', ['flight', 'model', 'mean error (m)', 'final error (m)'], rows)
    viz.plot_paths(utils.flight('loop'), {'similarity': T['loop'], 'homography': Th['loop']},
                   f'{out}/q3_2_paths_loop.png')
    from features import match
    pairs = []
    for fl, i, j in (('corniche', 20, 21), ('loop', 20, 21), ('loop', 3, 132)):
        (k1, d1), (k2, d2) = F[fl][i], F[fl][j]
        mm = match(d1, d2)
        pairs.append((f'{fl}, frames {i} and {j}', k1[mm[:, 0]], k2[mm[:, 1]], gt.gt_H(utils.flight(fl), i, j)))
    viz.plot_error_vs_radius(pairs, f'{out}/q3_2_error_vs_radius.png')

    # Q3.3
    for fl in flights:
        print(f'... building the map of {fl}', flush=True)
        m, S = make_map(utils.load_frames(utils.flight(fl), color=True), T[fl], scale=0.5)
        cv2.imwrite(f'{out}/q3_3_map_{fl}.jpg', m)
        if fl == 'loop':
            viz.closeup(m, S, T[fl][-3], 450, f'{out}/q3_3_closeup_loop.jpg')
        if fl == 'lost_at_sea':
            viz.closeup(m, S, T[fl][-1], 450, f'{out}/q3_3_closeup_lost_at_sea.jpg')
    print('... chaining vpair and building its map (the slowest step, two or three minutes)', flush=True)
    vp = utils.load_frames(utils.flight('vpair'), color=True)
    Tv, _, lost_v, _ = chain([feat(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)) for im in vp])
    m, _ = make_map(vp, Tv, scale=1 / 3)
    cv2.imwrite(f'{out}/q3_3_map_vpair.jpg', m)
    print(f'\nQ3.3  maps of corniche, loop, lost_at_sea and vpair (vpair: {len(vp)} frames, {len(lost_v)} lost steps)')
    print(f'\nEverything is in {os.path.relpath(out)}')


if __name__ == '__main__':
    utils.run(main)
