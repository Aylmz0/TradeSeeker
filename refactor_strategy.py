
METHODS_TO_EXTRACT = [
    "def check_trend_alignment",
    "def check_momentum_alignment",
    "def enhanced_trend_detection",
    "def calculate_comprehensive_trend_strength",
    "def analyze_rsi_strength",
    "def analyze_macd_strength",
    "def analyze_volume_strength",
    "def analyze_bollinger_bands_strength",
    "def analyze_moving_averages_strength",
    "def determine_trend_direction",
    "def get_confidence_level",
    "def calculate_volume_confidence",
    "def calculate_volume_quality_score",
    "def should_enhance_short_sizing",
    "def generate_advanced_exit_plan",
    "def detect_market_regime",
]

def main():
    with open("src/main.py", encoding="utf-8") as f:
        lines = f.readlines()

    out_main = []
    extracted_methods = []
    
    removing = False
    
    for line in lines:
        if line.startswith("    def "):
            stripped = line.lstrip()
            removing = any(stripped.startswith(m) for m in METHODS_TO_EXTRACT)
            
        if removing:
            extracted_methods.append(line)
        else:
            out_main.append(line)

    # 1. Write StrategyAnalyzer class
    strategy_code = [
        "import copy\n",
        "import json\n",
        "import re\n",
        "from typing import Any\n",
        "from config.config import Config\n",
        "from src.utils import format_num\n\n",
        "HTF_INTERVAL = getattr(Config, 'HTF_INTERVAL', '1h') or '1h'\n",
        "HTF_LABEL = HTF_INTERVAL\n\n",
        "class StrategyAnalyzer:\n",
        "    def __init__(self, market_data):\n",
        "        self.market_data = market_data\n\n",
    ]
    
    strategy_code.extend(extracted_methods)
    
    with open("src/core/strategy_analyzer.py", "w", encoding="utf-8") as f:
        f.writelines(strategy_code)

    # 2. Modify main.py
    main_code = "".join(out_main)
    
    # Import StrategyAnalyzer
    if "from src.core.strategy_analyzer import StrategyAnalyzer" not in main_code:
        import_stmt = "from src.core.portfolio_manager import PortfolioManager\nfrom src.core.strategy_analyzer import StrategyAnalyzer\n"
        main_code = main_code.replace("from src.core.portfolio_manager import PortfolioManager\n", import_stmt)

    # Instantiate StrategyAnalyzer in __init__
    init_stmt = "        self.market_data = RealMarketData()\n        self.strategy_analyzer = StrategyAnalyzer(self.market_data)\n"
    if init_stmt not in main_code:
        main_code = main_code.replace("        self.market_data = RealMarketData()\n", init_stmt)

    # Replace calls
    for m in METHODS_TO_EXTRACT:
        method_name = m.replace("def ", "").strip()
        main_code = main_code.replace(f"self.{method_name}", f"self.strategy_analyzer.{method_name}")

    with open("src/main.py", "w", encoding="utf-8") as f:
        f.write(main_code)

if __name__ == "__main__":
    main()
