from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FUSION_FILE = PROJECT_ROOT / "fusion" / "outputs" / "fusion_results.csv"
OUTPUT_DIR = PROJECT_ROOT / "adaptive_firewall" / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "firewall_recommendations.csv"

REQUIRED_COLUMNS = {"user", "final_score", "decision"}

def recommend_action(score, decision):
    decision = str(decision).strip().upper()

    if decision == "HIGH" or score >= 80:
        return "PROPOSE BLOCK", "High risk: review identity and mapped device before blocking"
    elif decision == "MEDIUM" or score >= 60:
        return "ALERT", "Medium risk: investigate the user's activity"
    elif decision == "LOW" or score >= 30:
        return "MONITOR", "Low risk: continue monitoring"
    else:
        return "MONITOR", "Normal risk: no blocking action recommended"

def main():
    if not FUSION_FILE.exists():
        raise FileNotFoundError(f"Fusion results not found: {FUSION_FILE}")

    df = pd.read_csv(FUSION_FILE)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Fusion CSV is missing columns: {sorted(missing)}")

    df["final_score"] = pd.to_numeric(df["final_score"], errors="raise")
    recommendations = df.apply(
        lambda row: recommend_action(row["final_score"], row["decision"]),
        axis=1,
        result_type="expand",
    )
    recommendations.columns = ["firewall_action", "reason"]
    df = pd.concat([df, recommendations], axis=1)
    df["enforcement_status"] = "SIMULATION ONLY"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("Adaptive firewall policy completed.")
    print(f"Input users: {len(df)}")
    print(f"Output file: {OUTPUT_FILE}")
    print("\nRecommended actions:")
    print(df["firewall_action"].value_counts().to_string())
    print("\nSample recommendations:")
    print(
        df[["user", "final_score", "decision", "firewall_action",
            "reason", "enforcement_status"]].head(10).to_string(index=False)
    )

if __name__ == "__main__":
    main()
