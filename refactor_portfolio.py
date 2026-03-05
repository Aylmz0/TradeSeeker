import os
import re

METHODS_TO_EXTRACT = [
    "def _initialize_live_trading",
    "def _build_default_exit_plan",
    "def _ensure_exit_plan",
    "def _merge_live_positions",
    "def sync_live_account",
    "def _calculate_realized_pnl",
    "def execute_live_entry",
    "def execute_live_close",
    "def execute_live_partial_close",
    "def close_position",
    "def check_and_execute_tp_sl",
    "def get_profit_levels_by_notional",
    "def enhanced_exit_strategy",
    "def _evaluate_trailing_stop",
    "def _execute_new_positions_only"
]

INTERNAL_METHODS = [m.replace("def ", "").strip() for m in METHODS_TO_EXTRACT]

def transform_method_body(code):
    # First, change all self. to self.pm.
    code = code.replace("self.", "self.pm.")
    # Now, revert the ones that belong to AccountService
    code = code.replace("self.pm.is_live_trading", "self.is_live_trading")
    code = code.replace("self.pm.order_executor", "self.order_executor")
    for m in INTERNAL_METHODS:
        code = code.replace(f"self.pm.{m}", f"self.{m}")
    return code

def main():
    with open("src/core/portfolio_manager.py", "r", encoding="utf-8") as f:
        lines = f.readlines()

    out_pm = []
    extracted_methods = []
    
    removing = False
    
    for line in lines:
        if line.startswith("    def ") or line.startswith("    @staticmethod"):
            # Check if this or the next line is the method definition
            # (In case of decorators like @staticmethod, the next line is the def)
            pass
            
        # simpler logic:
        stripped = line.lstrip()
        if stripped.startswith("def ") and line.startswith("    def "):
            removing = any(stripped.startswith(m) for m in METHODS_TO_EXTRACT)
            
        elif stripped.startswith("@staticmethod") and line.startswith("    @staticmethod"):
            # The only staticmethod we extract is _calculate_realized_pnl
            # Let's peek the next line
            idx = lines.index(line)
            if idx + 1 < len(lines) and "_calculate_realized_pnl" in lines[idx+1]:
                removing = True
            
        if removing:
            extracted_methods.append(transform_method_body(line))
        else:
            out_pm.append(line)

    # Clean up PM __init__
    pm_code = "".join(out_pm)
    
    # Remove Binance executor init from PortfolioManager
    init_remove = """        self.trading_mode = getattr(Config, "TRADING_MODE", "simulation")
        self.is_live_trading = self.trading_mode == "live"
        self.order_executor: BinanceOrderExecutor | None = None"""
    pm_code = pm_code.replace(init_remove, "")
    
    init_remove2 = """        if self.is_live_trading:
            self._initialize_live_trading()
        elif BINANCE_IMPORT_ERROR:
            print(
                f"[INFO] Binance executor unavailable ({BINANCE_IMPORT_ERROR}). Staying in simulation mode."
            )"""
    pm_code = pm_code.replace(init_remove2, "")

    with open("src/core/portfolio_manager.py", "w", encoding="utf-8") as f:
        f.write(pm_code)

    # 1. Write AccountService class
    account_service_code = [
        "import copy\n",
        "from datetime import datetime\n",
        "from typing import Any\n",
        "from config.config import Config\n",
        "from src.utils import format_num\n",
        "\n",
        "try:\n",
        "    from src.services.binance import BinanceOrderExecutor, BinanceAPIError\n",
        "    BINANCE_IMPORT_ERROR = None\n",
        "except Exception as e:\n",
        "    BinanceOrderExecutor = None\n",
        "    BINANCE_IMPORT_ERROR = str(e)\n",
        "    class BinanceAPIError(Exception): pass\n",
        "\n",
        "HTF_INTERVAL = getattr(Config, 'HTF_INTERVAL', '1h') or '1h'\n",
        "HTF_LABEL = HTF_INTERVAL\n",
        "\n",
        "class AccountService:\n",
        "    def __init__(self, portfolio_manager):\n",
        "        self.pm = portfolio_manager\n",
        "        self.is_live_trading = getattr(Config, 'TRADING_MODE', 'simulation') == 'live'\n",
        "        self.order_executor = None\n",
        "        if self.is_live_trading:\n",
        "            self._initialize_live_trading()\n",
        "        elif BINANCE_IMPORT_ERROR:\n",
        "            print(f'[INFO] Binance executor unavailable ({BINANCE_IMPORT_ERROR}). Staying in simulation mode.')\n\n"
    ]
    
    account_service_code.extend(extracted_methods)
    
    with open("src/core/account_service.py", "w", encoding="utf-8") as f:
        f.writelines(account_service_code)

    # 2. Modify main.py
    with open("src/main.py", "r", encoding="utf-8") as f:
        main_code = f.read()
    
    # Import AccountService
    if "from src.core.account_service import AccountService" not in main_code:
        import_stmt = "from src.core.portfolio_manager import PortfolioManager\nfrom src.core.account_service import AccountService\n"
        main_code = main_code.replace("from src.core.portfolio_manager import PortfolioManager\n", import_stmt)

    # Instantiate AccountService in __init__
    init_stmt = "        self.portfolio = PortfolioManager()\n        self.account_service = AccountService(self.portfolio)\n"
    if init_stmt not in main_code:
        main_code = main_code.replace("        self.portfolio = PortfolioManager()\n", init_stmt)

    # Replace self.portfolio.XXXX with self.account_service.XXXX for the extracted methods
    for m in INTERNAL_METHODS:
        main_code = main_code.replace(f"self.portfolio.{m}", f"self.account_service.{m}")

    with open("src/main.py", "w", encoding="utf-8") as f:
        f.write(main_code)

if __name__ == "__main__":
    main()
