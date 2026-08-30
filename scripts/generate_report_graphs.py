"""Generate report-ready figures for the restaurant feasibility project.

The script uses the final entropy dataset, the saved deployment model, and the
recorded model-comparison results from notebook 9a. Outputs are written to
``outputs/report_graphs`` as 300-DPI PNG files and supporting CSV tables.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, learning_curve


RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_final_entropy.csv"
WEIGHT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "entropy_feasibility_weightage.csv"
)
IMPORTANCE_PATH = (
    PROJECT_ROOT / "outputs" / "model_results" / "feature_importance.csv"
)
MODEL_PATH = (
    PROJECT_ROOT
    / "backend"
    / "models"
    / "restaurant_feasibility_pipeline.joblib"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "report_graphs"

COLORS = {
    "Low": "#D95F59",
    "Moderate": "#E5A84B",
    "High": "#2A9D8F",
    "blue": "#3478B8",
    "navy": "#234E70",
    "gray": "#6B7280",
}


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, filename: str) -> None:
    fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def cluster_validation(score: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for k in range(1, 11):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = model.fit_predict(score)
        rows.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette_score": (
                    silhouette_score(score, labels) if k >= 2 else np.nan
                ),
            }
        )

    metrics = pd.DataFrame(rows)
    metrics["inertia_reduction_percent"] = (
        -metrics["inertia"].pct_change() * 100
    )
    metrics.to_csv(OUTPUT_DIR / "cluster_validation_metrics.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))

    axes[0].plot(
        metrics["k"],
        metrics["inertia"],
        marker="o",
        linewidth=2.2,
        color=COLORS["blue"],
    )
    axes[0].scatter([3], metrics.loc[metrics["k"] == 3, "inertia"], s=110,
                    color="#E76F51", zorder=5, label="Selected k = 3")
    axes[0].axvline(3, color="#E76F51", linestyle="--", alpha=0.65)
    axes[0].set(title="Elbow Method", xlabel="Number of clusters (k)",
                ylabel="Within-cluster sum of squares (inertia)")
    axes[0].set_xticks(range(1, 11))
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    axes[0].legend(loc="upper right")

    sil = metrics.dropna(subset=["silhouette_score"])
    axes[1].plot(
        sil["k"],
        sil["silhouette_score"],
        marker="o",
        linewidth=2.2,
        color=COLORS["navy"],
    )
    k3_sil = float(sil.loc[sil["k"] == 3, "silhouette_score"].iloc[0])
    axes[1].scatter([3], [k3_sil], s=110, color="#E76F51", zorder=5,
                    label=f"Selected k = 3 ({k3_sil:.3f})")
    axes[1].axvline(3, color="#E76F51", linestyle="--", alpha=0.65)
    axes[1].set(title="Silhouette Analysis", xlabel="Number of clusters (k)",
                ylabel="Mean silhouette score")
    axes[1].set_xticks(range(2, 11))
    axes[1].set_ylim(0.48, 0.60)
    axes[1].legend(loc="upper right")

    fig.suptitle(
        "Validation of K-Means Feasibility Tiers (n = 4,172)",
        fontsize=15,
        fontweight="bold",
        y=1.03,
    )
    fig.text(
        0.5,
        -0.02,
        "Three clusters retain acceptable separation while producing actionable Low, Moderate, and High tiers.",
        ha="center",
        color="#4B5563",
        fontsize=10,
    )
    fig.tight_layout()
    save_figure(fig, "01_elbow_silhouette.png")
    return metrics


def cluster_profiles(df: pd.DataFrame) -> pd.DataFrame:
    order = ["Low", "Moderate", "High"]
    profile = (
        df.groupby("feasibility_label", observed=True)["feasibility_score"]
        .agg(["count", "min", "mean", "median", "max", "std"])
        .reindex(order)
    )
    profile.to_csv(OUTPUT_DIR / "cluster_profile_summary.csv")

    fig, axes = plt.subplots(
        2, 1, figsize=(11.4, 7.2), gridspec_kw={"height_ratios": [2.1, 1]}
    )
    for label in order:
        values = df.loc[df["feasibility_label"] == label, "feasibility_score"]
        sns.histplot(
            values,
            bins=28,
            stat="density",
            element="step",
            fill=True,
            alpha=0.25,
            color=COLORS[label],
            label=f"{label} (n={len(values):,})",
            ax=axes[0],
        )
        axes[0].axvline(values.mean(), color=COLORS[label], linestyle="--", lw=1.7)

    axes[0].set(
        title="Distribution of Feasibility Scores by Assigned Tier",
        xlabel="Entropy-weighted feasibility score (0–100)",
        ylabel="Density",
    )
    axes[0].legend(ncols=3, loc="upper center")

    sns.boxplot(
        data=df,
        x="feasibility_score",
        y="feasibility_label",
        order=order,
        hue="feasibility_label",
        palette=COLORS,
        legend=False,
        showfliers=False,
        ax=axes[1],
    )
    axes[1].set(
        title="Within-tier Score Ranges",
        xlabel="Entropy-weighted feasibility score (0–100)",
        ylabel="Feasibility tier",
    )
    fig.suptitle("Profile of the Three Feasibility Clusters", fontsize=15,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    save_figure(fig, "02_cluster_profiles.png")
    return profile


def feature_graphs() -> tuple[pd.DataFrame, pd.DataFrame]:
    weights = pd.read_csv(WEIGHT_PATH).sort_values("weight", ascending=True)
    importance = pd.read_csv(IMPORTANCE_PATH).sort_values(
        "Importance Mean", ascending=True
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2))
    axes[0].barh(weights["feature"], weights["weight"] * 100,
                 color="#5B8E7D")
    for i, value in enumerate(weights["weight"] * 100):
        axes[0].text(value + 0.7, i, f"{value:.1f}%", va="center", fontsize=9)
    axes[0].set(
        title="A. Entropy Weights Used to Build the Score",
        xlabel="Weight (%)",
        ylabel="Criterion",
        xlim=(0, 47),
    )

    axes[1].barh(
        importance["Feature"],
        importance["Importance Mean"],
        xerr=importance["Importance Std"],
        color="#4C78A8",
        ecolor="#374151",
        capsize=3,
    )
    for i, value in enumerate(importance["Importance Mean"]):
        axes[1].text(value + 0.012, i, f"{value:.3f}", va="center", fontsize=9)
    axes[1].set(
        title="B. Final Model Permutation Importance",
        xlabel="Mean decrease in macro F1 after permutation",
        ylabel="Model input feature",
        xlim=(0, 0.57),
    )
    fig.suptitle("Feature Set, Score Construction, and Predictive Importance",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.text(
        0.5,
        -0.02,
        "Error bars show ±1 standard deviation. Study area is included as a categorical model feature.",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )
    fig.tight_layout()
    save_figure(fig, "03_feature_weights_importance.png")
    return weights, importance


def final_model_learning_curve(df: pd.DataFrame) -> pd.DataFrame:
    package = joblib.load(MODEL_PATH)
    pipeline = package["pipeline"]
    metadata = package["metadata"]
    features = metadata["model_features"]
    label_to_int = {"Low": 0, "Moderate": 1, "High": 2}
    x = df[features]
    y = df["feasibility_label"].map(label_to_int)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    sizes = np.linspace(0.1, 1.0, 7)
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator=clone(pipeline),
        X=x,
        y=y,
        train_sizes=sizes,
        cv=cv,
        scoring="f1_macro",
        n_jobs=1,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    result = pd.DataFrame(
        {
            "training_samples": train_sizes,
            "training_macro_f1_mean": train_scores.mean(axis=1),
            "training_macro_f1_std": train_scores.std(axis=1),
            "validation_macro_f1_mean": validation_scores.mean(axis=1),
            "validation_macro_f1_std": validation_scores.std(axis=1),
        }
    )
    result.to_csv(OUTPUT_DIR / "learning_curve_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.4, 5.5))
    for prefix, label, color in [
        ("training", "Training macro F1", "#2A9D8F"),
        ("validation", "Cross-validation macro F1", "#E76F51"),
    ]:
        mean = result[f"{prefix}_macro_f1_mean"].to_numpy()
        std = result[f"{prefix}_macro_f1_std"].to_numpy()
        samples = result["training_samples"].to_numpy()
        ax.plot(samples, mean, marker="o", lw=2.2, label=label, color=color)
        ax.fill_between(samples, mean - std, mean + std, color=color, alpha=0.16)

    ax.set(
        title="Learning Curve of the Final Tuned Logistic Regression Model",
        xlabel="Number of training samples",
        ylabel="Macro F1 score (5-fold stratified CV)",
        ylim=(0.94, 1.005),
    )
    ax.legend(loc="lower right")
    final_gap = (
        result.iloc[-1]["training_macro_f1_mean"]
        - result.iloc[-1]["validation_macro_f1_mean"]
    )
    ax.text(
        0.02,
        0.05,
        f"Final generalization gap: {final_gap:.3f}",
        transform=ax.transAxes,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#9CA3AF"},
    )
    fig.tight_layout()
    save_figure(fig, "04_learning_curve.png")
    return result


def model_comparison_graph() -> pd.DataFrame:
    # Recorded outputs from notebooks/9a_feature_prep_model_dataset_final_entropy.ipynb.
    comparison = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "XGBoost",
                "Random Forest",
                "Gradient Boosting",
            ],
            "Test Macro F1": [0.9778, 0.9803, 0.9768, 0.9778],
            "5-fold CV Macro F1": [0.9792, 0.9791, 0.9716, 0.9703],
            "Area-held-out Macro F1": [0.9734, 0.9560, 0.9515, 0.9566],
        }
    )
    comparison.to_csv(OUTPUT_DIR / "model_comparison_metrics.csv", index=False)

    long = comparison.melt(id_vars="Model", var_name="Evaluation",
                           value_name="Macro F1")
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    sns.barplot(
        data=long,
        x="Model",
        y="Macro F1",
        hue="Evaluation",
        palette=["#4C78A8", "#59A14F", "#F28E2B"],
        ax=ax,
    )
    ax.set(
        title="Comparative Generalization Performance of Candidate Models",
        xlabel="Candidate classifier",
        ylabel="Macro F1 score",
        ylim=(0.93, 0.99),
    )
    ax.tick_params(axis="x", rotation=12)
    ax.legend(title=None, ncols=3, loc="upper center")
    ax.text(
        0.02,
        0.06,
        "Logistic Regression achieved the best area-held-out result\n"
        "and was therefore selected for tuning and deployment.",
        transform=ax.transAxes,
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": "#9CA3AF"},
    )
    fig.tight_layout()
    save_figure(fig, "05_model_comparison.png")
    return comparison


def write_report_insert(
    metrics: pd.DataFrame,
    profile: pd.DataFrame,
    learning: pd.DataFrame,
) -> None:
    k2_sil = metrics.loc[metrics["k"] == 2, "silhouette_score"].iloc[0]
    k3_sil = metrics.loc[metrics["k"] == 3, "silhouette_score"].iloc[0]
    k3_inertia = metrics.loc[metrics["k"] == 3, "inertia"].iloc[0]
    k4_inertia = metrics.loc[metrics["k"] == 4, "inertia"].iloc[0]
    final_learning = learning.iloc[-1]

    text = f"""# Report-ready additions

## Cluster-number selection

K-Means clustering was applied to the entropy-weighted feasibility score. The
elbow curve shows a pronounced reduction in within-cluster variation up to
approximately three clusters, followed by progressively smaller practical
gains. At `k = 3`, inertia was {k3_inertia:,.1f}; increasing to four clusters
reduced it to {k4_inertia:,.1f}, but would divide the decision output into an
additional tier that was not required by the Low–Moderate–High interpretation.
The silhouette score for three clusters was {k3_sil:.3f}, indicating reasonable
separation. Although the numerical maximum occurred at two clusters
({k2_sil:.3f}), that solution cannot represent the important intermediate
"Moderate" feasibility category. Therefore, `k = 3` was selected as a balance
between compactness, separation, and decision usefulness.

**Figure caption:** *Elbow and silhouette analysis for K-Means solutions from
one to ten clusters. The selected three-cluster solution provides reasonable
statistical separation and directly maps to Low, Moderate, and High restaurant
location feasibility tiers.*

## Three-cluster profile

The final clusters contained {int(profile.loc['Low', 'count']):,} Low,
{int(profile.loc['Moderate', 'count']):,} Moderate, and
{int(profile.loc['High', 'count']):,} High observations. Their mean feasibility
scores were {profile.loc['Low', 'mean']:.2f},
{profile.loc['Moderate', 'mean']:.2f}, and
{profile.loc['High', 'mean']:.2f}, respectively. The learned score intervals
were approximately {profile.loc['Low', 'min']:.2f}–{profile.loc['Low', 'max']:.2f}
for Low, {profile.loc['Moderate', 'min']:.2f}–{profile.loc['Moderate', 'max']:.2f}
for Moderate, and {profile.loc['High', 'min']:.2f}–{profile.loc['High', 'max']:.2f}
for High.

**Figure caption:** *Distribution and within-tier spread of entropy-weighted
feasibility scores for the three K-Means clusters.*

## Feature list and importance

The final model used four numerical criteria—Demand, Accessibility,
Competition, and Population—and one categorical feature, search area. The
feasibility score was constructed with entropy-derived weights of 41.82% for
Demand, 33.37% for Population, 16.01% for Competition, and 8.79% for
Accessibility. Permutation analysis of the deployed classifier similarly
identified Demand and Population as the most influential predictive inputs.
The search-area feature had near-zero global permutation importance, suggesting
that most predictive information was carried by the four substantive criteria.

**Figure caption:** *Entropy weights used in score construction and permutation
importance of the five inputs to the final feasibility classifier. Error bars
represent one standard deviation over repeated permutations.*

## Learning curve

The final tuned Logistic Regression pipeline was evaluated using a stratified
five-fold learning curve with macro F1 as the scoring measure. At the largest
training size, mean training macro F1 was
{final_learning['training_macro_f1_mean']:.3f}, while mean validation macro F1
was {final_learning['validation_macro_f1_mean']:.3f}. The small gap of
{final_learning['training_macro_f1_mean'] - final_learning['validation_macro_f1_mean']:.3f}
indicates limited overfitting, and the validation curve's stabilization shows
that the model generalizes consistently as more training data are added.

**Figure caption:** *Five-fold stratified learning curve for the final tuned
Logistic Regression model. Shaded bands show ±1 standard deviation across
folds.*

## Comparative analysis

Four classifiers were compared using an untouched test set, stratified
five-fold cross-validation, and area-held-out validation. XGBoost produced the
highest initial test macro F1 (0.9803), but Logistic Regression had nearly
identical cross-validation performance (0.9792) and the strongest area-held-out
macro F1 (0.9734). The area-held-out result was important because it tests
transfer to an unseen study area. Logistic Regression was therefore selected
for tuning; the saved final model achieved a test accuracy of 0.9940 and test
macro F1 of 0.9942.

**Figure caption:** *Macro F1 comparison of candidate models under test-set,
five-fold cross-validation, and area-held-out evaluation. Logistic Regression
showed the strongest spatial generalization and was selected for deployment.*

## Methodological note

The feasibility labels were produced from an entropy-weighted score and then
learned by a supervised classifier. Thus, the reported accuracy measures how
consistently the model reproduces the constructed feasibility tiers; it should
not be interpreted as external proof of restaurant business success. A future
validation study should compare predictions with independent outcomes such as
revenue, survival, rent-adjusted profitability, or observed foot traffic.
"""
    (OUTPUT_DIR / "REPORT_INSERT.md").write_text(text, encoding="utf-8")


def main() -> None:
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()

    df = pd.read_csv(DATA_PATH)
    required = {
        "feasibility_score",
        "feasibility_label",
        "Demand",
        "Accessibility",
        "Competition",
        "Population",
        "search_area",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    metrics = cluster_validation(df[["feasibility_score"]])
    profile = cluster_profiles(df)
    feature_graphs()
    learning = final_model_learning_curve(df)
    model_comparison_graph()
    write_report_insert(metrics, profile, learning)

    print(f"Generated report figures and tables in: {OUTPUT_DIR}")
    for path in sorted(OUTPUT_DIR.iterdir()):
        print(f"- {path.name}")


if __name__ == "__main__":
    main()
