import argparse
import json
from os import path

from configuration.options import args, parser
from scheduler.BasicSched import BasicScheduler
from scheduler.DVAEDistillSched import DVAEDistillScheduler
from scheduler.DVAEEnsembleDistillSched import DVAEEnsembleDistillScheduler
from scheduler.DistillSched import DistillScheduler
from scheduler.EnsembleSched import EnsembleScheduler
from utils import *

if __name__ == '__main__':
    with open(path.normpath(args.config_file), 'r') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)

    export_root = setup_train(args)

    # sched = BasicScheduler(args, export_root)
    # sched = EnsembleScheduler(args, export_root)
    sched = DistillScheduler(args, export_root)
    # sched = DVAEDistillScheduler(args, export_root)
    # sched = DVAEEnsembleDistillScheduler(args, export_root)

    sched.run()
