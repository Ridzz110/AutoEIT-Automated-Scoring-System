import os
import pandas as pd
from preprocessor import load_all_files
from scorer import score_participant
import glob


def main():
    os.makedirs("output", exist_ok=True)
    FILES = sorted([f for f in glob.glob('data/*.csv') if 'Info' not in f])
    if not FILES:
        print("No CSV files found in data/ folder")
        return
    
    print(f"Found {len(FILES)} participant files:")
    for f in FILES:
        print(f"  {f}")

    print("\nLoading participant files...\n")
    dataframes = load_all_files(FILES)

    all_results = []

    for df in dataframes:
        participant = df['Participant'].iloc[0]
        print(f"\nScoring participant: {participant}")

        scored_df = score_participant(df)
        all_results.append(scored_df)

        output_path = f"output/scored_{participant}.csv"
        scored_df.to_csv(output_path, index=False)
        print(f"  Saved: {output_path}")
        print(f"  Total LLM Score: {scored_df['LLM_Score'].sum()} / {len(scored_df) * 4}")
        print(f"  Avg Semantic Similarity: {scored_df['Semantic_Similarity'].mean():.4f}")
        flagged = scored_df['Divergence_Flag'].sum()
        print(f"  Divergent sentences flagged: {flagged}")

    # Combine and save all
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv("output/AutoEIT_All_Scored.csv", index=False)

    # Print summary
    print("\n===== SCORING SUMMARY =====")
    for participant, group in combined.groupby('Participant'):
        total = group['LLM_Score'].sum()
        maximum = len(group) * 4
        pct = round((total / maximum) * 100, 1)
        avg_sim = group['Semantic_Similarity'].mean()
        flagged = group['Divergence_Flag'].sum()
        print(f"{participant}:")
        print(f"  LLM Score:           {total}/{maximum} ({pct}%)")
        print(f"  Avg Similarity:      {avg_sim:.4f}")
        print(f"  Divergent sentences: {flagged}")

    print("\nDone! Results saved to output/")

if __name__ == "__main__":
    main()
