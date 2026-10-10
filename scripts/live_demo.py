"""CYCLOPS live demo.

    python scripts/live_demo.py                          # interactive window
    python scripts/live_demo.py --scenario dynamic --seed 12
    python scripts/live_demo.py --record demo.mp4        # offline recording (backup for the viva!)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='forest_sar',
                    choices=['forest_sar', 'urban_inspection', 'dynamic', 'pillars'])
    ap.add_argument('--seed', type=int, default=108)
    ap.add_argument('--oracle', action='store_true', help='use ground-truth obstacles instead of vision')
    ap.add_argument('--no-ghost', action='store_true', help='hide the baseline ghost drone')
    ap.add_argument('--speed', type=float, default=1.0, help='simulation speed multiplier')
    ap.add_argument('--record', default=None, help='write an mp4 instead of opening a window')
    ap.add_argument('--duration', type=float, default=60.0, help='max simulated seconds when recording')
    args = ap.parse_args()

    import matplotlib
    if args.record:
        matplotlib.use('Agg')
    from cyclops.config import Config
    from cyclops.demo.live import LiveDemo

    perception = None
    if not args.oracle:
        from cyclops.perception.perception import Perception
        perception = Perception(Config().camera)
    demo = LiveDemo(args.scenario, args.seed, 'oracle' if args.oracle else 'vision', perception,
                    ghost=not args.no_ghost, speed=args.speed, record=args.record, duration=args.duration)
    demo.run()


if __name__ == '__main__':
    main()
