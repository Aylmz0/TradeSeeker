import sys

with open("src/core/market_data.py", "r") as f:
    lines = f.readlines()

out = []
in_indicator_funcs = False
in_eff = False

for line in lines:
    if "def calculate_ema_series(" in line:
        in_indicator_funcs = True
    elif "def get_technical_indicators(" in line:
        in_indicator_funcs = False
        
    if "def calculate_efficiency_ratio(" in line:
        in_eff = True
    elif "def get_all_real_prices(" in line:
        in_eff = False
        
    if in_indicator_funcs or in_eff:
        continue
        
    out.append(line)

code = "".join(out)

code = code.replace("self.calculate_ema_series", "calculate_ema_series")
code = code.replace("self.calculate_rsi_series", "calculate_rsi_series")
code = code.replace("self.calculate_macd_series", "calculate_macd_series")
code = code.replace("self.calculate_atr_series", "calculate_atr_series")
code = code.replace("self.calculate_adx", "calculate_adx")
code = code.replace("self.calculate_vwap", "calculate_vwap")
code = code.replace("self.calculate_bollinger_bands", "calculate_bollinger_bands")
code = code.replace("self.calculate_obv", "calculate_obv")
code = code.replace("self.calculate_supertrend", "calculate_supertrend")
code = code.replace("self.calculate_efficiency_ratio", "calculate_efficiency_ratio")
code = code.replace("self._generate_smart_sparkline", "generate_smart_sparkline")
code = code.replace("self._calculate_pivots", "calculate_pivots")
code = code.replace("self._generate_tags", "generate_tags")
code = code.replace("self._extract_semantic_features", "extract_semantic_features")

import_str = """from config.config import Config
from src.utils import RetryManager
from src.core.indicators import (
    calculate_ema_series, calculate_rsi_series, calculate_macd_series,
    calculate_atr_series, calculate_adx, calculate_vwap, calculate_bollinger_bands,
    calculate_obv, calculate_supertrend, calculate_efficiency_ratio,
    extract_semantic_features, generate_smart_sparkline, calculate_pivots, generate_tags
)"""
code = code.replace("from config.config import Config\nfrom src.utils import RetryManager", import_str)

with open("src/core/market_data.py", "w") as f:
    f.write(code)
