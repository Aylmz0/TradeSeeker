import json
import os
import glob
import re


def analyze_full_er_history():
    base_path = "/home/yilmaz/projects/TradeSeeker/data"
    files = glob.glob(os.path.join(base_path, "**/cycle_history.json"), recursive=True)

    er_data = {}  # coin -> list of ERs
    total_cycles = 0

    # Pattern to find ER in chain_of_thoughts (e.g., "ER 0.04" or "efficiency ratio 0.169")
    er_pattern = re.compile(
        r"(?:ER|efficiency ratio|efficiency_ratio)\s*[:=]?\s*([0-9]\.[0-9]+)", re.IGNORECASE
    )

    for path in files:
        if not os.path.exists(path):
            continue

        with open(path, "r") as f:
            try:
                cycles = json.load(f)
            except:
                continue

            total_cycles += len(cycles)
            for entry in cycles:
                cot = entry.get("chain_of_thoughts", "")
                # Find all matches for ER in CoT
                matches = er_pattern.findall(cot)
                if matches:
                    # Usually multiple ERs (one per coin), but hard to map to coin without complex parsing
                    # Let's just collect all ERs found to get a general market sense
                    for m in matches:
                        er_val = float(m)
                        if "market_wide" not in er_data:
                            er_data["market_wide"] = []
                        er_data["market_wide"].append(er_val)

    if "market_wide" not in er_data:
        print("Could not find ER data in chain_of_thoughts.")
        return

    all_ers = er_data["market_wide"]
    avg_er = sum(all_ers) / len(all_ers)
    max_er = max(all_ers)
    min_er = min(all_ers)

    choppy_count = len([x for x in all_ers if x < 0.3])
    trending_count = len([x for x in all_ers if x >= 0.3])

    print(f"\n--- Statistical Market Analysis (77 Cycles) ---")
    print(f"Total ER Samples Found: {len(all_ers)}")
    print(f"Average Market ER: {avg_er:.4f}")
    print(f"Max ER Seen: {max_er:.4f}")
    print(f"Min ER Seen: {min_er:.4f}")
    print(f"\n--- Regime Distribution ---")
    print(f"CHOPPY Cycles (< 0.3): {choppy_count} ({choppy_count/len(all_ers)*100:.1f}%)")
    print(f"TRENDING Cycles (>= 0.3): {trending_count} ({trending_count/len(all_ers)*100:.1f}%)")


if __name__ == "__main__":
    analyze_full_er_history()
