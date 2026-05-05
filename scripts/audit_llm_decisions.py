import json
import os


def analyze_cycles_deep(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    with open(file_path, "r") as f:
        data = json.load(f)

    print(f"--- DEEP AUDIT: LLM INTENT VS SYSTEM FILTERS ---\n")

    intent_count = 0
    executed_count = 0
    blocked_by_limit = 0

    for cycle_data in data:
        cycle_num = cycle_data.get("cycle")
        decisions = cycle_data.get("decisions", {})
        thoughts = cycle_data.get("thoughts", "")

        # Check for system block notes in thoughts
        system_blocked_note = "[Position Limit" in thoughts or "[Directional Capacity" in thoughts

        for ticker, decision in decisions.items():
            signal = decision.get("signal")
            justification = decision.get("justification", "")

            # Case 1: LLM said BUY/SELL and it remained so
            if signal in ["buy_to_enter", "sell_to_enter"]:
                intent_count += 1
                exec_report = cycle_data.get("metadata", {}).get("execution_report", {})
                executed_list = exec_report.get("executed", [])
                is_executed = any(e.get("coin") == ticker for e in executed_list)

                if is_executed:
                    executed_count += 1
                    print(f"Cycle {cycle_num} | {ticker} | INTENT: {signal} | RESULT: EXECUTED")
                else:
                    print(
                        f"Cycle {cycle_num} | {ticker} | INTENT: {signal} | RESULT: BLOCKED (by Risk/Confidence/Balance)"
                    )

            # Case 2: LLM said BUY/SELL but system converted it to HOLD (Limit reached)
            elif signal == "hold" and "Position limit reached" in justification:
                intent_count += 1
                blocked_by_limit += 1
                print(
                    f"Cycle {cycle_num} | {ticker} | INTENT: BUY/SELL | RESULT: SYSTEM BLOCKED (Position Limit)"
                )

    print(f"\n--- DEEP SUMMARY ---")
    print(f"Total True LLM Entry Intentions: {intent_count}")
    print(f"Actually Executed: {executed_count}")
    print(f"Blocked by Cycle Position Limits: {blocked_by_limit}")
    print(f"Blocked by Risk/Conf/Other: {intent_count - executed_count - blocked_by_limit}")


if __name__ == "__main__":
    analyze_cycles_deep("data/cycle_history.json")
