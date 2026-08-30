#!/usr/bin/env python3
"""Launcher.  Threading environment must be set before numpy is imported,
which is why this wrapper exists instead of calling `python -m pse.driver`
directly under mpirun."""
import os
import sys

_n = os.environ.get("PSE_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _n)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir))

from pse.driver import main                                        # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
