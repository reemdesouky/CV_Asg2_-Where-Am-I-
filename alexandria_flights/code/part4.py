"""Part 4: loop closure. Run it once Q4.1 and Q4.2 pass their checks:

    python part4.py            (add --sift to use the SIFT fallback instead of your Part 1 features)

Writes to results/part4/:
    q4_1_revisits_<flight>.png             Q4.1: accepted revisits drawn on the chained path
    q4_2_errors.md                         Q4.2: mean and final error before and after the correction
    q4_2_path_<flight>.png                 Q4.2: the path before and after, against the truth
    q4_2_map_<flight>_{before,after}.jpg   Q4.2: the maps before and after
    q4_2_closeup_<flight>_{before,after}.jpg   Q4.2: where each flight returns to known ground
    T_<flight>_corrected.npy               the corrected transforms, frame k -> frame 0
"""
import os, sys
import numpy as np, cv2
import utils, gt, viz
from features import features, features_sift
from odometry import chain
from loop_closure import revisits, correct
from mapping import make_map


def main():
    out = utils.results_dir('part4')
    feat = features_sift if '--sift' in sys.argv else features
    rows = []
    for fl, back in (('loop', -3), ('lost_at_sea', -1)):       # `back`: a frame where the flight returns
        path = utils.flight(fl)
        print(f'... {fl}: features, chain, revisits, correction, maps', flush=True)
        F = [feat(im) for im in utils.load_frames(path)]
        T, steps, lost, _ = chain(F)

        # Q4.1
        revs = revisits(F)
        viz.plot_revisits(T, revs, f'{out}/q4_1_revisits_{fl}.png', flight_dir=path)
        print(f'Q4.1  {fl}: {len(revs)} revisits accepted')

        # Q4.2
        Tc = correct(len(F), steps, revs)
        np.save(f'{out}/T_{fl}_corrected.npy', np.array(Tc))
        e0, e1 = gt.path_errors(path, T), gt.path_errors(path, Tc)
        rows += [[fl, 'chained', float(e0.mean()), float(e0[-1])], [fl, 'loop closure', float(e1.mean()), float(e1[-1])]]
        viz.plot_paths(path, {'chained': T, 'loop closure': Tc}, f'{out}/q4_2_path_{fl}.png')
        col = utils.load_frames(path, color=True)
        for name, Ts in (('before', T), ('after', Tc)):
            m, S = make_map(col, Ts, scale=0.5)
            cv2.imwrite(f'{out}/q4_2_map_{fl}_{name}.jpg', m)
            viz.closeup(m, S, Ts[back], 450, f'{out}/q4_2_closeup_{fl}_{name}.jpg')
    print('\nQ4.2  before and after loop closure')
    utils.table(f'{out}/q4_2_errors.md', ['flight', 'path', 'mean error (m)', 'final error (m)'], rows)
    print(f'\nEverything is in {os.path.relpath(out)}')


if __name__ == '__main__':
    utils.run(main)
