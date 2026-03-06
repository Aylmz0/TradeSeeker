METHODS_TO_EXTRACT = [
    "def format_position_context",
    "def format_market_regime_context",
    "def format_performance_insights",
    "def format_directional_feedback",
    "def format_risk_context",
    "def format_suggestions",
    "def format_trend_reversal_analysis",
    "def format_volume_ratio",
    "def format_list",
    "def generate_alpha_arena_prompt",
    "def generate_alpha_arena_prompt_json",
    "def parse_ai_response",
    "def _clean_ai_decisions",
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

    # 1. Write AIService class
    ai_code = [
        "import copy\n",
        "import json\n",
        "import re\n",
        "from typing import Any\n",
        "from config.config import Config\n",
        "from src.utils import format_num\n\n",
        "HTF_INTERVAL = getattr(Config, 'HTF_INTERVAL', '1h') or '1h'\n",
        "HTF_LABEL = HTF_INTERVAL\n\n",
        "class AIService:\n",
        "    def __init__(self, portfolio, market_data, strategy_analyzer):\n",
        "        self.portfolio = portfolio\n",
        "        self.market_data = market_data\n",
        "        self.strategy_analyzer = strategy_analyzer\n\n",
    ]

    ai_code.extend(extracted_methods)

    # We need to replace `self.` with `self.` carefully. Inside AIService, `self.portfolio` requires `self.portfolio`, etc.
    # What did `generate_alpha_arena_prompt_json` access?
    # self.market_data
    # self.portfolio
    # self.strategy_analyzer (was self.)
    # The safest way is to replace `self.` with `self.` for AIService methods, but `generate_alpha_arena_prompt` accesses `self.market_data.available_coins`.

    aiservice_code_str = "".join(ai_code)
    # the format methods are accessed as self.format_xxx, which remains same in AIService.
    # self.market_data is same. self.portfolio is same.
    # But methods in StrategyAnalyzer? We replaced them in main.py, so they are `self.strategy_analyzer.xxx` which matches AIService.
    # Replace `self.recent_decisions`? Wait, they use `self.portfolio` methods maybe?
    # Let's write the raw class first and check dependencies.

    with open("src/core/ai_service.py", "w", encoding="utf-8") as f:
        f.write(aiservice_code_str)

    # 2. Modify main.py
    main_code = "".join(out_main)

    if "from src.core.ai_service import AIService" not in main_code:
        import_stmt = "from src.core.portfolio_manager import PortfolioManager\nfrom src.core.ai_service import AIService\n"
        main_code = main_code.replace(
            "from src.core.portfolio_manager import PortfolioManager\n", import_stmt
        )

    init_stmt = "        self.strategy_analyzer = StrategyAnalyzer(self.market_data)\n        self.ai_service = AIService(self.portfolio, self.market_data, self.strategy_analyzer)\n"
    if "self.ai_service = AIService(" not in main_code:
        main_code = main_code.replace(
            "        self.strategy_analyzer = StrategyAnalyzer(self.market_data)\n", init_stmt
        )

    # Replace self.generate_alpha_arena_prompt_json with self.ai_service.generate_alpha_arena_prompt_json
    main_code = main_code.replace(
        "self.generate_alpha_arena_prompt_json", "self.ai_service.generate_alpha_arena_prompt_json"
    )
    main_code = main_code.replace(
        "self.generate_alpha_arena_prompt", "self.ai_service.generate_alpha_arena_prompt"
    )
    main_code = main_code.replace("self.parse_ai_response", "self.ai_service.parse_ai_response")
    main_code = main_code.replace("self._clean_ai_decisions", "self.ai_service._clean_ai_decisions")

    with open("src/main.py", "w", encoding="utf-8") as f:
        f.write(main_code)


if __name__ == "__main__":
    main()
