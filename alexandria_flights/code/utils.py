"""Helpers used by every part. Already written: use them as they are.

Conventions used in every file:
    points       (N, 2) float arrays of (x, y) pixel coordinates.
    transforms   (3, 3) float arrays acting on homogeneous pixels, normalized so that T[2, 2] = 1.
    F            a list with one (keypoints, descriptors) pair per frame, as returned by features().
"""
import os, glob
import numpy as np, cv2


def load_frames(flight_dir, color=False):
    """Load every frame of a flight, in flight order.

    Args:
        flight_dir: path to a flight folder, the one that contains frames/.
        color: True for BGR color images, False for grayscale.

    Returns:
        A list of uint8 images, (H, W) in grayscale or (H, W, 3) in color.
    """
    files = sorted(glob.glob(os.path.join(flight_dir, 'frames', '*.jpg')) +
                   glob.glob(os.path.join(flight_dir, 'frames', '*.png')))
    return [cv2.imread(f, cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE) for f in files]


def apply(T, p):
    """Apply a transform to points.

    Args:
        T: (3, 3) transform.
        p: (N, 2) points.

    Returns:
        (N, 2) transformed points.
    """
    ph = np.c_[p, np.ones(len(p))] @ T.T
    return ph[:, :2] / ph[:, 2:]


# ------------------------------------------------------------------ for the part scripts
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..'))       # the flights sit next to this code folder


def flight(name):
    """Path of a flight folder.

    Args:
        name: 'corniche', 'loop', 'lost_at_sea' or 'vpair'.

    Returns:
        The path of that flight's folder, next to the code folder.
    """
    return os.path.join(DATA, name)


def results_dir(part):
    """The folder a part script writes its figures and tables to, created if needed.

    Args:
        part: 'part1', 'part2', 'part3' or 'part4'.

    Returns:
        The path of results/<part>/, next to the code folder.
    """
    d = os.path.join(DATA, 'results', part)
    os.makedirs(d, exist_ok=True)
    return d


def table(path, header, rows):
    """Print a table and save it as Markdown, ready to paste into the report.

    Args:
        path: the .md file to write.
        header: list of column names.
        rows: list of rows, each a list of values. Floats below 10 are printed with two decimals, larger
            ones as whole numbers.
    """
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return f'{v:.2f}' if abs(v) < 10 else f'{v:.0f}'
        return str(v)
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    lines += ['| ' + ' | '.join(fmt(v) for v in r) + ' |' for r in rows]
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def run(main):
    """Run a part script, and stop with a message naming the TODO if it reaches one you have not filled in.

    Args:
        main: the script's main() function.
    """
    import sys
    try:
        main()
    except NotImplementedError as e:
        print(f'\nStopped: {e}.')
        print('Fill in that TODO (python check.py tells you when it works), then run this script again.')
        sys.exit(1)
