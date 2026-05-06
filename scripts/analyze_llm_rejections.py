import json
import os
import glob


def analyze_rejections():
    base_path = "/home/yilmaz/projects/TradeSeeker/data"
    # Find all cycle_history.json files
    files = glob.glob(os.path.join(base_path, "**/cycle_history.json"), recursive=True)

    total_proposals = 0
    executed = 0
    blocked_reasons = {}
    total_cycles = 0

    for path in files:
        if not os.path.exists(path):
            continue

        with open(path, "r") as f:
            try:
                cycles = json.load(f)
            except json.JSONDecodeError:
                continue

            total_cycles += len(cycles)

            for entry in cycles:
                decisions = entry.get("decisions", {})
                execution_report = entry.get("metadata", {}).get("execution_report", {})

                # Check LLM proposals
                for coin, dec in decisions.items():
                    signal = dec.get("signal", "")
                    if signal in ["buy_to_enter", "sell_to_enter"]:
                        total_proposals += 1

                        # Check if it was in executed list
                        was_executed = any(
                            item.get("coin") == coin
                            for item in execution_report.get("executed", [])
                        )
                        if was_executed:
                            executed += 1
                        else:
                            # Check why it was blocked
                            block_info = next(
                                (
                                    item
                                    for item in execution_report.get("blocked", [])
                                    if item.get("coin") == coin
                                ),
                                None,
                            )
                            if not block_info:
                                block_info = next(
                                    (
                                        item
                                        for item in execution_report.get("skipped", [])
                                        if item.get("coin") == coin
                                    ),
                                    None,
                                )

                            if block_info:
                                reason = block_info.get("reason", "Unknown")
                                blocked_reasons[reason] = blocked_reasons.get(reason, 0) + 1
                            else:
                                blocked_reasons["Implicitly Filtered / Slot Full"] = (
                                    blocked_reasons.get("Implicitly Filtered / Slot Full", 0) + 1
                                )

    print(f"\n--- LLM Proposal vs System Execution Report (Full History) ---")
    print(f"Total Cycles Analyzed: {total_cycles}")
    print(f"Total LLM Entry Proposals: {total_proposals}")
    print(f"Successfully Executed: {executed}")
    print(f"Total Blocked/Filtered: {total_proposals - executed}")
    print(f"\n--- Rejection Reasons Breakdown ---")
    for reason, count in sorted(blocked_reasons.items(), key=lambda x: x[1], reverse=True):
        print(f"- {reason}: {count}")


if __name__ == "__main__":
    analyze_rejections()
