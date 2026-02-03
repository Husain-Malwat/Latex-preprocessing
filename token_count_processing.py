import os
import json
import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------- PATHS ----------------
json_dir = "/home/husainmalwat/workspace/OCR_Latex/data/token_count_logs_preprocess_1"
csv_output_dir = "/home/husainmalwat/workspace/OCR_Latex/experiments/7_NORMALIZATION_WITH_STY/2new_csv_outputs"
os.makedirs(csv_output_dir, exist_ok=True)

# ---------------- TOKEN RANGES ----------------
intervals = [
    (0, 500, "0–500", "tokens_0-500.csv"),
    (500, 3000, "500–3k", "tokens_500-3k.csv"),
    (3000, 7000, "3k–7k", "tokens_3k-7k.csv"),
    (7000, 15000, "7k–15k", "tokens_7k-15k.csv"),
    (15000, 31000, "15k–31k", "tokens_15k-31k.csv"),
    (31000, 63000, "31k–63k", "tokens_31k-63k.csv"),
    (63000, float("inf"), ">63k", "tokens_63k+.csv"),
]

interval_data = {label: [] for _, _, label, _ in intervals}
all_tokens = []

# ---------------- READ JSON FILES ----------------
for filename in sorted(os.listdir(json_dir)):
    if not filename.endswith(".json"):
        continue

    try:
        with open(os.path.join(json_dir, filename), "r") as f:
            data = json.load(f)

        for entry in data.get("details", []):
            token_count = entry["tokens"]
            file_path = entry["file"]

            all_tokens.append({
                "file": file_path,
                "num_tokens": token_count
            })

            for min_v, max_v, label, _ in intervals:
                if min_v <= token_count < max_v:
                    interval_data[label].append({
                        "file": file_path,
                        "token_count": token_count
                    })
                    break

        print(f"Processed {filename}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

# ---------------- WRITE CSVs ----------------
for _, _, label, csv_name in intervals:
    csv_path = os.path.join(csv_output_dir, csv_name)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "token_count"])
        writer.writeheader()
        writer.writerows(interval_data[label])
    print(f"Created {csv_path} ({len(interval_data[label])} files)")

combined_csv_path = os.path.join(csv_output_dir, "qwen_token_counts_combined.csv")
pd.DataFrame(all_tokens).to_csv(combined_csv_path, index=False)
print(f"\nCreated combined CSV: {combined_csv_path}")

# ---------------- PRINT STATISTICS ----------------
print("\n" + "=" * 60)
print("TOKEN COUNT STATISTICS BY RANGE")
print("=" * 60)

total_files = len(all_tokens)

for min_v, max_v, label, _ in intervals:
    files = interval_data[label]
    count = len(files)
    pct = (count / total_files * 100) if total_files else 0

    print(f"\nRange: {label}")
    print(f"  Files: {count} ({pct:.2f}%)")

    if count > 0:
        values = [x["token_count"] for x in files]
        print(f"  Avg tokens: {sum(values)/count:.0f}")
        print(f"  Min tokens: {min(values)}")
        print(f"  Max tokens: {max(values)}")

print("\nTOTAL FILES:", total_files)
print("=" * 60)

# ---------------- VISUALIZATION ----------------
df = pd.read_csv(combined_csv_path)
df = df[df["num_tokens"] >= 0]

# Assign token ranges
def token_range(x):
    for min_v, max_v, label, _ in intervals:
        if min_v <= x < max_v:
            return label
    return ">63k"

df["token_range"] = df["num_tokens"].apply(token_range)

sns.set_theme(style="whitegrid")
palette = sns.color_palette("crest")

# Prepare ordered labels
ordered_labels = [label for _, _, label, _ in intervals]

# Aggregate stats per range
range_counts = (
    df["token_range"]
    .value_counts()
    .reindex(ordered_labels)
    .fillna(0)
    .astype(int)
)

range_pct = (range_counts / len(df) * 100).round(2) if len(df) else pd.Series(0, index=ordered_labels)
range_stats = {}
for label in ordered_labels:
    files = interval_data.get(label, [])
    count = len(files)
    vals = [x["token_count"] for x in files] if count else []
    range_stats[label] = {
        "count": int(count),
        "percentage": float(range_pct[label]) if label in range_pct else 0.0,
        "avg_tokens": int(sum(vals)/count) if count else None,
        "min_tokens": int(min(vals)) if vals else None,
        "max_tokens": int(max(vals)) if vals else None
    }

# Save stats to JSON
stats_output_path = os.path.join(csv_output_dir, "token_counts_stats.json")
stats_dump = {
    "total_files": int(len(df)),
    "ranges": range_stats
}
with open(stats_output_path, "w", encoding="utf-8") as jf:
    json.dump(stats_dump, jf, indent=2, ensure_ascii=False)
print(f"Saved stats JSON: {stats_output_path}")

# -------- 1. FILE COUNT PER RANGE (Number of Papers vs Token Range) --------
plt.figure(figsize=(12, 6))
ax = sns.barplot(x=range_counts.index, y=range_counts.values, palette=palette)
plt.title("Number of Papers per Token Range", fontsize=16, weight="bold")
plt.xlabel("Token Range")
plt.ylabel("Number of Papers")
plt.xticks(rotation=30)

# Annotate bars with counts
for p in ax.patches:
    height = int(p.get_height())
    if height > 0:
        ax.annotate(f"{height}", (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(csv_output_dir, "papers_per_token_range.png"), dpi=300)
plt.close()

# -------- 2. PERCENTAGE PER RANGE --------
plt.figure(figsize=(12, 6))
ax = sns.barplot(x=range_pct.index, y=range_pct.values, palette=palette)
plt.title("Percentage of Papers per Token Range", fontsize=16, weight="bold")
plt.xlabel("Token Range")
plt.ylabel("Percentage (%)")
plt.xticks(rotation=30)

for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f"{height:.1f}%", (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(csv_output_dir, "percentage_per_token_range.png"), dpi=300)
plt.close()

# -------- 3. LOG-SCALE HISTOGRAM (Number of Papers by token counts) --------
plt.figure(figsize=(14, 6))
sns.histplot(
    df["num_tokens"],
    bins=50,
    log_scale=(True, False),
    color=palette[4],
    alpha=0.85
)
plt.title("Token Count Distribution (Log Scale)", fontsize=16, weight="bold")
plt.xlabel("Number of Tokens (log scale)")
plt.ylabel("Number of Papers")
plt.tight_layout()
plt.savefig(os.path.join(csv_output_dir, "token_distribution_logscale.png"), dpi=300)
plt.close()

# -------- 4. BOXPLOT (OUTLIERS) --------
plt.figure(figsize=(14, 4))
sns.boxplot(x=df["num_tokens"], color=palette[4])
plt.xscale("log")
plt.title("Token Count Boxplot (Log Scale)", fontsize=16, weight="bold")
plt.xlabel("Number of Tokens (log scale)")
plt.tight_layout()
plt.savefig(os.path.join(csv_output_dir, "token_boxplot_logscale.png"), dpi=300)
plt.close()

print("\nAll visualizations and JSON stats generated successfully!")
