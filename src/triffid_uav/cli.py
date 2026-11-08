"""Command line: `python -m triffid_uav <command>`.

  run       inputs/ + input_topview/  ->  outputs/point_detections.json, outputs/polygon_detections.json
  check     validate inputs, supports and classes; extract the frames; no model is loaded
  bank      support-bank tools (validate / init / add)
  overlays  draw an output JSON on its frames
"""
import argparse
import json
import logging
import sys

from .config import load_config


def _config_options(parser):
    parser.add_argument('--config', default='configs/default.yaml', help='YAML config (default: %(default)s)')
    parser.add_argument('--root', help='Folder that relative config paths resolve against (default: CWD)')
    parser.add_argument('--set', dest='overrides', action='append', default=[], metavar='SECTION.KEY=VALUE',
                        help='Override a config value, e.g. --set model.shots=1 (repeatable)')
    parser.add_argument('--shots', type=int, choices=[1, 5], help='Shortcut for --set model.shots=N')


def _load(args):
    overrides = list(args.overrides)
    if args.shots is not None:
        overrides.append(f'model.shots={args.shots}')
    return load_config(args.config, overrides, root=args.root)


def build_parser():
    parser = argparse.ArgumentParser(prog='triffid_uav', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true')
    commands = parser.add_subparsers(dest='command', required=True)
    _config_options(commands.add_parser('run', help='Run points (orbit frames) and polygons (nadir frame)'))
    _config_options(commands.add_parser('check', help='Validate inputs and extract frames without the model'))

    bank = commands.add_parser('bank', help='Support bank tools')
    bank_commands = bank.add_subparsers(dest='bank_command', required=True)
    validate = bank_commands.add_parser('validate', help='Check every requested class has five valid supports')
    validate.add_argument('--bank', default='supports')
    validate.add_argument('--classes', help='Comma-separated class names/IDs (default: all classes)')
    ready = bank_commands.add_parser('ready', help='List classes usable with N shots')
    ready.add_argument('--bank', default='supports')
    ready.add_argument('--shots', type=int, choices=[1, 5], default=5)
    init = bank_commands.add_parser('init', help='Create an empty managed bank (5 slots per class)')
    init.add_argument('--bank', default='supports')
    add = bank_commands.add_parser('add', help='Add one image/mask example to an empty slot')
    add.add_argument('--bank', default='supports')
    add.add_argument('--class', dest='class_name', required=True)
    add.add_argument('--slot', type=int, choices=range(1, 6), required=True)
    add.add_argument('--image', required=True)
    add.add_argument('--mask', required=True)
    add.add_argument('--foreground-value', type=int)

    overlays = commands.add_parser('overlays', help='Render an output JSON on its frames')
    overlays.add_argument('json')
    overlays.add_argument('--output', help='Overlay folder (default: next to the JSON)')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(levelname)s %(name)s: %(message)s')
    try:
        if args.command == 'run':
            from .pipeline import run
            summary = run(_load(args))
            print(json.dumps({k: summary[k] for k in ('outputs', 'counts', 'timings')}, indent=2))
        elif args.command == 'check':
            from .pipeline import check
            print(json.dumps(check(_load(args)), indent=2))
        elif args.command == 'bank':
            from .ontology import CLASSES
            from .supports import SupportBank, add_example, initialize_bank, resolve_class
            if args.bank_command == 'init':
                print(initialize_bank(args.bank))
            elif args.bank_command == 'add':
                print(add_example(args.bank, args.class_name, args.slot, args.image, args.mask,
                                  args.foreground_value))
            elif args.bank_command == 'ready':
                bank = SupportBank(args.bank)
                print(json.dumps({g: [c.name for c in bank.ready_classes(g, args.shots)]
                                  for g in ('Point', 'Polygon')}, indent=2))
            else:
                classes = tuple(dict.fromkeys(resolve_class(v) for v in args.classes.split(','))) \
                    if args.classes else CLASSES
                report = SupportBank(args.bank).validate(classes)
                print(json.dumps(report, indent=2))
                if not report['valid']:
                    sys.exit(2)
        elif args.command == 'overlays':
            from .visualize import visualize
            print(visualize(args.json, args.output))
    except (ValueError, OSError, ImportError) as error:
        print(f'triffid_uav error: {error}', file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    main()
