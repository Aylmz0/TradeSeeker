import json
import os


PROJECT_ROOT = "/home/yilmaz/projects/TradeSeeker"
active_cycle_file = os.path.join(PROJECT_ROOT, "data/cycle_history.json")

if os.path.exists(active_cycle_file):
    with open(active_cycle_file) as f:
        try:
            data = json.load(f)
            if data:
                print("Cycle history keys:", data[-1].keys())
                print("\nSample cycle entry details:")
                for k, v in data[-1].items():
                    if k != "chain_of_thoughts" and k != "decisions":
                        print(f" - {k}: {type(v)}")
                if "market_data" in data[-1]:
                    md = data[-1]["market_data"]
                    print("\ntype of market_data:", type(md))
                    if isinstance(md, list) and md:
                        print("Sample market_data item keys:", md[0].keys())
                        print("Sample item:", md[0])
                    elif isinstance(md, dict):
                        print("market_data keys:", md.keys())
                        first_key = list(md.keys())[0]
                        print(f"market_data[{first_key}] keys:", md[first_key].keys())
                        print("Sample item:", md[first_key])
        except Exception as e:
            print("Error parsing cycle history:", e)
