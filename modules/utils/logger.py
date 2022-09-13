from __future__ import absolute_import, division, print_function

import logging

# disable matplotlib logging
logging.getLogger("matplotlib").setLevel(logging.INFO)
import matplotlib

del matplotlib


def setup_logger() -> logging.Logger:
    head = "%(asctime)-15s %(message)s"
    logging.basicConfig(format=head)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    return logger


logger: logging.Logger = setup_logger()
