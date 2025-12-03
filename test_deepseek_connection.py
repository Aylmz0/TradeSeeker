import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

api_key = os.getenv('DEEPSEEK_API_KEY')
if not api_key:
    print("❌ DEEPSEEK_API_KEY not found in .env")
    exit(1)

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=180.0)

def test_simple_connection():
    print("\n1️⃣  TESTING SIMPLE CONNECTION ('Hello')...")
    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Hello"}],
            stream=False
        )
        duration = time.time() - start_time
        print(f"✅ Success! Response: {response.choices[0].message.content}")
        print(f"⏱️  Duration: {duration:.2f}s")
        return True
    except Exception as e:
        print(f"❌ Simple connection failed: {e}")
        return False

def test_large_payload():
    print("\n2️⃣  TESTING LARGE PAYLOAD (~15k chars)...")
    # Create a dummy large JSON payload
    dummy_data = {
        "market_data": {
            "coins": {
                f"COIN_{i}": {
                    "price": [100 + x for x in range(30)],
                    "rsi": [50 + (x%10) for x in range(30)],
                    "volume": [1000000 for _ in range(30)],
                    "indicators": {"ema": 105.5, "macd": 0.5}
                } for i in range(20) # 20 coins to bulk it up
            },
            "filler": "x" * 5000 # Add bulk text
        }
    }
    payload = json.dumps(dummy_data)
    print(f"📦 Payload Size: {len(payload)} chars")
    
    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a JSON parser. Output JSON."},
                {"role": "user", "content": f"Analyze this JSON and return a summary JSON: {payload}"}
            ],
            response_format={ "type": "json_object" },
            stream=False
        )
        duration = time.time() - start_time
        print(f"✅ Success! Response received.")
        print(f"⏱️  Duration: {duration:.2f}s")
        return True
    except Exception as e:
        print(f"❌ Large payload failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 STARTING DEEPSEEK DIAGNOSTIC...")
    if test_simple_connection():
        time.sleep(2)
        test_large_payload()
    print("\n🏁 DIAGNOSTIC COMPLETE")
