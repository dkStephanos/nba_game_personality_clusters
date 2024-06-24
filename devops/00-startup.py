# 00-startup.py

print("Loading default imports...")

from models.fpgrowth import run_fpgrowth
from models.fpgrowth import mlxtend_fpgrowth

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Verify ffmpeg path and set it for matplotlib
ffmpeg_path = "/usr/bin/ffmpeg"
plt.rcParams["animation.ffmpeg_path"] = ffmpeg_path
plt.rcParams["animation.writer"] = "ffmpeg"

# Print a message to confirm the script has run
print("Default imports loaded.")
