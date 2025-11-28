import sys
import os
sys.path.append(os.getcwd())
from config.config import Config

print(f"Config.SAME_DIRECTION_LIMIT: {Config.SAME_DIRECTION_LIMIT}")
print(f"Config.MAX_POSITIONS: {Config.MAX_POSITIONS}")
print(f"os.getenv('SAME_DIRECTION_LIMIT'): {os.getenv('SAME_DIRECTION_LIMIT')}")
