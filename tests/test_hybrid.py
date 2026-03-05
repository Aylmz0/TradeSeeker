import json
import logging
from config.config import Config
from src.core.market_data import RealMarketData
from src.core.portfolio_manager import PortfolioManager
from src.core.performance_monitor import PerformanceMonitor
from src.strategies.strategy_analyzer import StrategyAnalyzer
from src.core.ai_service import AIService

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_hybrid_ai_prompt():
    print("--- Testing Hybrid AI Prompt Generation (XGBoost + LLM) ---")
    
    # Init Core Mock Objects
    market_data = RealMarketData()
    portfolio = PortfolioManager()
    performance_monitor = PerformanceMonitor()
    strategy_analyzer = StrategyAnalyzer(market_data, portfolio, performance_monitor)
    
    # Init AI Service (This now strictly depends on MLService internally)
    ai_service = AIService(portfolio, market_data, strategy_analyzer)
    
    # We will only look at XRP for brevity in this dry run
    market_data.available_coins = ["XRP"]
    
    print("\n[INFO] Triggering generate_alpha_arena_prompt_json()...")
    try:
        prompt = ai_service.generate_alpha_arena_prompt_json()
        print("\n[OK] Prompt generated successfully. Extracting the MARKET DATA block...\n")
        
        # Searching the prompt string for the JSON block we care about
        # It's inside a markdown block: ```json\n[{...}]\n```
        import re
        match = re.search(r'```json\n(.*?)\n```', prompt, re.DOTALL)
        
        if match:
            # We found the first JSON block (which might be Counter Trades or Market Data)
            # Let's just print the whole prompt since we want to see the ML injection.
            print("================== TRUNCATED PROMPT OUTPUT ==================")
            # Prettify the prompt by grabbing the market data section
            market_data_str = prompt.split('MARKET_DATA')[1]
            print(market_data_str[:1500] + "\n... [TRUNCATED ALIVE DATA] ...")
            print("=============================================================")
        else:
            print("[WARN] Could not parse json blocks from prompt.")
            print(prompt[:1000])
            
    except Exception as e:
        print(f"\n[FAIL] Error generating Hybrid Prompt: {e}")

if __name__ == "__main__":
    test_hybrid_ai_prompt()
