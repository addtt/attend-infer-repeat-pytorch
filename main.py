import warnings

import boilr
import multiobject
from boilr import Trainer

from experiments import AIRExperiment


def _check_version(pkg, pkg_str, version_info):
    def to_str(v):
        return ".".join(str(x) for x in v)
    if pkg.__version_info__[:2] != version_info[:2]:
        msg = "This was last tested with {} {}, but the current version is {}"
        msg = msg.format(pkg_str, to_str(version_info), pkg.__version__)
        warnings.warn(msg)

BOILR_VERSION = (0, 4, 0)
MULTIOBJ_VERSION = (0, 0, 3)
_check_version(boilr, 'boilr', BOILR_VERSION)
_check_version(multiobject, 'multiobject', MULTIOBJ_VERSION)

def main():
    experiment = AIRExperiment()
    trainer = Trainer(experiment)
    trainer.run()

if __name__ == "__main__":
    main()
