"""Part 1: features. Run it once Q1.1 and Q1.2 pass their checks:

    python part1.py

Writes to results/part1/:
    q1_1_harris.png                                Q1.1: a corniche frame, its Harris response, the kept corners
    q1_2_corniche_20_21_{raw,oriented}.png         Q1.2: matches between two consecutive corniche frames
    q1_2_loop_3_132_{raw,oriented}.png             Q1.2: matches between frames 3 and 132 of loop
"""
import os
import numpy as np
import utils, gt, viz
from features import harris_response, detect, features, match


def main():
    out = utils.results_dir('part1')
    COR, LOOP = utils.flight('corniche'), utils.flight('loop')
    cor, cor_c = utils.load_frames(COR), utils.load_frames(COR, color=True)
    loop, loop_c = utils.load_frames(LOOP), utils.load_frames(LOOP, color=True)

    # Q1.1
    R = harris_response(cor[20])
    kps = detect(cor[20])
    viz.show_corners(cor_c[20], kps, R, f'{out}/q1_1_harris.png')
    print(f'Q1.1  {len(kps)} corners on corniche frame 20')

    # Q1.2: a match counts as correct if the ground truth puts it within 3 px (corniche) or 30 px (loop 3 -> 132:
    # frames far apart, at a very different heading)
    for fl, fr, fc, flight, i, j, tol in [('corniche', cor, cor_c, COR, 20, 21, 3),
                                          ('loop', loop, loop_c, LOOP, 3, 132, 30)]:
        name = f'{fl}_{i}_{j}'
        for desc, oriented in (('raw', False), ('oriented', True)):
            (k1, d1), (k2, d2) = features(fr[i], oriented), features(fr[j], oriented)
            mm = match(d1, d2)
            p, q = k1[mm[:, 0]], k2[mm[:, 1]]
            ok = gt.count_correct(p, q, gt.gt_H(flight, i, j), tol)
            viz.show_matches(fc[i], fc[j], p, q, ok, title=f'{fl}, frames {i} and {j}, {desc} patch: {len(p)} '
                             f'matches, {int((~ok).sum())} wrong (red)', path=f'{out}/q1_2_{name}_{desc}.png')
            print(f'Q1.2  {fl}, frames {i} and {j}, {desc} patch: {len(p)} matches, {int((~ok).sum())} wrong')
    print(f'\nFigures in {os.path.relpath(out)}')


if __name__ == '__main__':
    utils.run(main)
