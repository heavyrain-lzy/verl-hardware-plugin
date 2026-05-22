# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""verl hardware plugin - Multi-chip platform and engine support.

This package registers hardware platforms (FlagOS, XPU, MLU) and their
corresponding training engines with verl's plugin system.

Loading methods:
    1. VERL_USE_EXTERNAL_MODULES=verl_hardware_plugin
    2. setuptools entry_points (auto-discovered via verl.plugins group)
"""

import logging
import os

from verl_hardware_plugin.platforms import register_all_platforms
from verl_hardware_plugin.engines import register_all_engines

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

register_all_platforms()
register_all_engines()

logger.info("verl-hardware-plugin loaded successfully")
