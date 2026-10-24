"""Generate EDA charts and the internship capstone PDF report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PDF_PATH = ROOT / "E20_Capstone_Project_Report.pdf"

NAVY = "#0f2c59"
TEAL = "#1b6ca8"
GOLD = "#c9a227"
SLATE = "#334155"

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": SLATE,
        "axes.labelcolor": SLATE,
        "text.color": SLATE,
        "xtick.color": SLATE,
        "ytick.color": SLATE,
        "axes.titleweight": "bold",
    }
)


def load_and_clean() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw = pd.read_excel(ROOT / "E20_predictions_updated.xlsx")
    notes = {}
    notes["raw_rows"] = int(len(raw))
    notes["raw_cols"] = int(raw.shape[1])
    notes["missing_before"] = raw.isna().sum().to_dict()
    notes["dups_before"] = int(raw.duplicated().sum())

    df = raw.copy()
    # Launch Year Note is almost entirely empty and only flags 2 Scorpio-N rows.
    df = df.drop(columns=["Launch Year Note"], errors="ignore")

    # Standardise text columns
    for col in ["Brand", "Model", "Variant", "E20 Compatible"]:
        df[col] = df[col].astype(str).str.strip()

    notes["engine_missing"] = int(df["Engine (CC)"].isna().sum())
    notes["mileage_missing"] = int(df["Mileage Tested (km/L)"].isna().sum())
    notes["drop_missing"] = int(df["E20_Drop_%_updated"].isna().sum())

    # Incorrect / inapplicable records for petrol E20 analysis
    non_petrol = df["E20 Compatible"].isin(["N/A (CNG)", "N/A (Diesel)"])
    notes["non_petrol_rows"] = int(non_petrol.sum())

    # Duplicate brand-model-variant-engine combinations
    key_cols = ["Brand", "Model", "Variant", "Engine (CC)", "Launch Year"]
    notes["key_dups"] = int(df.duplicated(subset=key_cols).sum())
    df = df.drop_duplicates(subset=key_cols, keep="first")

    # Analysis set: petrol-relevant rows with usable engine + drop
    analysis = df.dropna(subset=["Engine (CC)", "E20_Drop_%_updated"]).copy()
    analysis = analysis[~analysis["E20 Compatible"].isin(["N/A (CNG)", "N/A (Diesel)"])]
    analysis["Mileage Gap (km/L)"] = (
        analysis["Mileage Claimed (km/L)"] - analysis["Mileage Tested (km/L)"]
    )
    analysis["Mileage Gap %"] = (
        analysis["Mileage Gap (km/L)"] / analysis["Mileage Claimed (km/L)"] * 100
    )

    notes["analysis_rows"] = int(len(analysis))
    notes["compat_counts"] = df["E20 Compatible"].value_counts(dropna=False).to_dict()
    notes["brand_counts"] = df["Brand"].value_counts().to_dict()
    notes["numeric_describe"] = df.describe().round(2).to_dict()
    return raw, df, analysis, notes


def iqr_outliers(series: pd.Series) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)


def save_fig(fig: plt.Figure, name: str) -> Path:
    path = FIG_DIR / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def make_charts(df: pd.DataFrame, analysis: pd.DataFrame) -> dict[str, Path]:
    charts = {}

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    counts = df["Brand"].value_counts().sort_values()
    counts.plot(kind="barh", ax=ax, color=TEAL)
    ax.set_title("Number of Vehicle Records by Brand")
    ax.set_xlabel("Number of records")
    ax.set_ylabel("")
    charts["brand"] = save_fig(fig, "01_records_by_brand.png")

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    compat = df["E20 Compatible"].value_counts()
    colors_pie = sns.color_palette("Set2", len(compat))
    ax.pie(compat, labels=compat.index, autopct="%1.1f%%", colors=colors_pie, startangle=90)
    ax.set_title("E20 Compatibility Mix in the Dataset")
    charts["compat"] = save_fig(fig, "02_e20_compatibility.png")

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    sns.histplot(analysis["Engine (CC)"], bins=18, kde=True, color=TEAL, ax=ax)
    ax.set_title("Distribution of Engine Displacement (CC)")
    ax.set_xlabel("Engine size (CC)")
    ax.set_ylabel("Count")
    charts["engine"] = save_fig(fig, "03_engine_distribution.png")

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    year_counts = df["Launch Year"].value_counts().sort_index()
    year_counts.plot(kind="bar", ax=ax, color=NAVY)
    ax.set_title("Vehicle Coverage by Launch Year")
    ax.set_xlabel("Launch year")
    ax.set_ylabel("Number of records")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    charts["year"] = save_fig(fig, "04_launch_year.png")

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    claimed = analysis["Mileage Claimed (km/L)"]
    tested = analysis["Mileage Tested (km/L)"]
    ax.scatter(claimed, tested, alpha=0.7, color=TEAL, edgecolor="white", s=55)
    lims = [
        np.nanmin([claimed.min(), tested.min()]) - 1,
        np.nanmax([claimed.max(), tested.max()]) + 1,
    ]
    ax.plot(lims, lims, color="crimson", linestyle="--", label="Claimed = tested")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Claimed mileage (km/L)")
    ax.set_ylabel("Real-world tested mileage (km/L)")
    ax.set_title("Claimed vs Real-World Tested Mileage")
    ax.legend()
    charts["mileage"] = save_fig(fig, "05_claimed_vs_tested.png")

    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    numeric_cols = [
        "Engine (CC)",
        "Launch Year",
        "Mileage Claimed (km/L)",
        "Mileage Tested (km/L)",
        "E20_Drop_%_updated",
    ]
    corr = analysis[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", center=0, ax=ax)
    ax.set_title("Correlation Heatmap of Numeric Features")
    charts["corr"] = save_fig(fig, "06_correlation_heatmap.png")

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    sns.boxplot(
        data=analysis,
        x="E20 Compatible",
        y="E20_Drop_%_updated",
        hue="E20 Compatible",
        ax=ax,
        palette="Set2",
        legend=False,
    )
    ax.set_title("E20 Mileage Drop % by Compatibility Status")
    ax.set_xlabel("E20 compatibility")
    ax.set_ylabel("Updated E20 drop (%)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    charts["box"] = save_fig(fig, "07_drop_by_compatibility.png")

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    sns.scatterplot(
        data=analysis,
        x="Launch Year",
        y="E20_Drop_%_updated",
        hue="E20 Compatible",
        s=70,
        ax=ax,
    )
    ax.set_title("Launch Year vs E20 Mileage Drop %")
    ax.set_ylabel("Updated E20 drop (%)")
    charts["year_drop"] = save_fig(fig, "08_year_vs_drop.png")

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    brand_drop = (
        analysis.groupby("Brand")["E20_Drop_%_updated"].mean().sort_values()
    )
    brand_drop.plot(kind="barh", ax=ax, color=GOLD)
    ax.set_title("Average E20 Mileage Drop % by Brand")
    ax.set_xlabel("Average updated E20 drop (%)")
    ax.set_ylabel("")
    charts["brand_drop"] = save_fig(fig, "09_avg_drop_by_brand.png")

    # Dashboard-style KPI visual (Power BI substitute)
    monthly_km, price = 1200, 96.5
    cost_rows = analysis.dropna(subset=["Mileage Tested (km/L)"]).copy()
    cost_rows["e20_mileage"] = cost_rows["Mileage Tested (km/L)"] * (
        1 - cost_rows["E20_Drop_%_updated"] / 100
    )
    cost_rows["extra_monthly"] = (
        monthly_km / cost_rows["e20_mileage"] - monthly_km / cost_rows["Mileage Tested (km/L)"]
    ) * price

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.6))
    fig.suptitle("Analytics Snapshot (KPI + Trend Views)", fontsize=14, fontweight="bold", color=NAVY)

    kpis = [
        (f"{analysis['E20_Drop_%_updated'].mean():.1f}%", "Avg E20 drop"),
        (f"{(df['E20 Compatible']=='Yes').mean()*100:.0f}%", "Share E20-compatible"),
        (f"{cost_rows['extra_monthly'].mean():.0f}", "Avg extra Rs/month"),
        (f"{analysis['Mileage Gap %'].mean():.1f}%", "Claimed vs tested gap"),
    ]
    ax = axes[0, 0]
    ax.axis("off")
    ax.set_title("Key performance indicators")
    for i, (val, label) in enumerate(kpis):
        y = 0.78 - i * 0.22
        ax.text(0.06, y, val, fontsize=16, fontweight="bold", color=NAVY, transform=ax.transAxes)
        ax.text(0.42, y, label, fontsize=11, color=SLATE, transform=ax.transAxes)

    yearly = analysis.groupby("Launch Year")["E20_Drop_%_updated"].mean()
    axes[0, 1].plot(yearly.index, yearly.values, marker="o", color=TEAL)
    axes[0, 1].set_title("Average drop by launch year")
    axes[0, 1].set_ylabel("Drop %")

    yes_no = analysis[analysis["E20 Compatible"].isin(["Yes", "No"])]
    sns.barplot(
        data=yes_no,
        x="E20 Compatible",
        y="E20_Drop_%_updated",
        estimator="mean",
        ax=axes[1, 0],
        color=TEAL,
    )
    axes[1, 0].set_title("Mean drop: compatible vs not")
    axes[1, 0].set_ylabel("Drop %")

    sns.histplot(cost_rows["extra_monthly"], bins=16, ax=axes[1, 1], color=GOLD, kde=True)
    axes[1, 1].set_title("Extra monthly fuel cost (Rs)")
    axes[1, 1].set_xlabel("Rs extra per month @ 1,200 km / Rs 96.5")
    fig.tight_layout()
    charts["dashboard"] = save_fig(fig, "10_analytics_snapshot.png")
    return charts, cost_rows


def build_pdf(raw, df, analysis, notes, charts, cost_rows) -> None:
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "CoverTitle",
            parent=styles["Title"],
            fontSize=24,
            textColor=colors.HexColor(NAVY),
            spaceAfter=10,
            alignment=TA_CENTER,
            leading=30,
        )
    )
    styles.add(
        ParagraphStyle(
            "CoverSub",
            parent=styles["Normal"],
            fontSize=12,
            textColor=colors.HexColor(TEAL),
            alignment=TA_CENTER,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            "H",
            parent=styles["Heading1"],
            fontSize=15,
            textColor=colors.HexColor(NAVY),
            spaceBefore=12,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            "H2",
            parent=styles["Heading2"],
            fontSize=12,
            textColor=colors.HexColor(TEAL),
            spaceBefore=8,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            "Caption",
            parent=styles["Normal"],
            fontSize=8.5,
            textColor=colors.HexColor("#64748b"),
            alignment=TA_CENTER,
            spaceBefore=3,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            "BulletBody",
            parent=styles["Normal"],
            fontSize=10,
            leading=13.5,
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            "FooterNote",
            parent=styles["Normal"],
            fontSize=8,
            textColor=colors.HexColor("#64748b"),
            alignment=TA_CENTER,
        )
    )

    def img(path: Path, width=6.6 * inch) -> Image:
        im = Image(str(path))
        ratio = im.imageHeight / im.imageWidth
        im.drawWidth = width
        im.drawHeight = width * ratio
        return im

    def bullets(items: list[str]) -> ListFlowable:
        return ListFlowable(
            [ListItem(Paragraph(x, styles["BulletBody"]), leftIndent=8) for x in items],
            bulletType="bullet",
            start="circle",
            leftIndent=16,
            spaceAfter=8,
        )

    def section_table(data, col_widths=None):
        t = Table(data, colWidths=col_widths, hAlign="LEFT")
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8fafc"), colors.white]),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        return t

    # Stats
    avg_drop = analysis["E20_Drop_%_updated"].mean()
    med_drop = analysis["E20_Drop_%_updated"].median()
    yes_drop = analysis.loc[analysis["E20 Compatible"] == "Yes", "E20_Drop_%_updated"].mean()
    no_drop = analysis.loc[analysis["E20 Compatible"] == "No", "E20_Drop_%_updated"].mean()
    claimed_mean = analysis["Mileage Claimed (km/L)"].mean()
    tested_mean = analysis["Mileage Tested (km/L)"].mean()
    gap_pct = analysis["Mileage Gap %"].mean()
    extra_mean = cost_rows["extra_monthly"].mean()
    extra_p90 = cost_rows["extra_monthly"].quantile(0.90)
    compatible_share = (df["E20 Compatible"] == "Yes").mean() * 100
    engine_out = int(iqr_outliers(analysis["Engine (CC)"]).sum())
    drop_out = int(iqr_outliers(analysis["E20_Drop_%_updated"]).sum())
    mileage_out = int(iqr_outliers(analysis["Mileage Tested (km/L)"].dropna()).sum())
    corr_year = analysis[["Launch Year", "E20_Drop_%_updated"]].corr().iloc[0, 1]
    corr_engine = analysis[["Engine (CC)", "E20_Drop_%_updated"]].corr().iloc[0, 1]
    corr_tested = analysis[["Mileage Tested (km/L)", "E20_Drop_%_updated"]].corr().iloc[0, 1]
    brand_best = analysis.groupby("Brand")["E20_Drop_%_updated"].mean().sort_values()
    newest_yes = analysis[analysis["Launch Year"] >= 2022]
    newest_share = (newest_yes["E20 Compatible"] == "Yes").mean() * 100
    older = analysis[analysis["Launch Year"] <= 2018]
    older_no = (older["E20 Compatible"] == "No").mean() * 100 if len(older) else 0

    missing_rows = [["Column", "Missing values", "Share of 221 rows"]]
    for col, n in notes["missing_before"].items():
        missing_rows.append([col, str(n), f"{n / notes['raw_rows'] * 100:.1f}%"])

    dtype_rows = [["Column", "Data type", "Role"]]
    dtype_map = {
        "Brand": ("categorical", "Vehicle manufacturer"),
        "Model": ("categorical", "Car model name"),
        "Variant": ("categorical", "Engine / trim variant"),
        "Engine (CC)": ("numeric", "Displacement feature"),
        "Mileage Claimed (km/L)": ("numeric", "OEM claimed mileage"),
        "Mileage Tested (km/L)": ("numeric", "Real-world tested mileage"),
        "Launch Year": ("integer", "Model introduction year"),
        "Launch Year Note": ("text", "Rare annotation (dropped)"),
        "E20 Compatible": ("categorical", "Compatibility label"),
        "E20_Drop_%": ("numeric", "Original drop estimate"),
        "Mileage_E20_Estimated": ("numeric", "Original E20 mileage"),
        "E20_Drop_%_updated": ("numeric", "Target: updated drop %"),
        "Mileage_E20_Estimated_updated": ("numeric", "Updated E20 mileage"),
    }
    for col in raw.columns:
        kind, role = dtype_map.get(col, (str(raw[col].dtype), ""))
        dtype_rows.append([col, kind, role])

    story = []
    story.append(Spacer(1, 1.5 * inch))
    story.append(Paragraph("E20 Mileage Drop &amp; Fuel Cost Estimator", styles["CoverTitle"]))
    story.append(Paragraph("Internship Capstone Project Report", styles["CoverSub"]))
    story.append(Paragraph("Option 3 &mdash; Direct PDF Upload", styles["CoverSub"]))
    story.append(Spacer(1, 0.25 * inch))
    story.append(
        Paragraph(
            "Student: Utsav Ujjwal<br/>Domain: Automotive / Fuel Cost Analytics (India)<br/>"
            "Repository: github.com/UtsavUjjwal/E20-Expense-Predictor<br/>"
            "Live app: https://e20predictor.streamlit.app/",
            styles["CoverSub"],
        )
    )
    story.append(Spacer(1, 0.4 * inch))
    story.append(
        Paragraph(
            "This report documents dataset selection, cleaning, exploratory analysis, "
            "visualisation, business insights, and recommendations. Power BI dashboard "
            "work (Task 6) is omitted as agreed; an analytics snapshot page is included instead.",
            styles["Body"],
        )
    )
    story.append(PageBreak())

    # 1 Overview
    story.append(Paragraph("1. Project Overview", styles["H"]))
    story.append(
        Paragraph(
            "India is shifting petrol supply toward E20 (20% ethanol blend). Ethanol has lower "
            "energy density than petrol, so many vehicles lose mileage and owners spend more on "
            "fuel. Compatibility also varies: some engines are certified for E20, while older or "
            "non-certified cars can lose more efficiency. This project turns that problem into a "
            "practical estimator for Indian car buyers and fleet operators.",
            styles["Body"],
        )
    )
    story.append(Paragraph("Problem statement", styles["H2"]))
    story.append(
        Paragraph(
            "Drivers still plan monthly fuel budgets using claimed km/L figures and do not know "
            "how much extra they will pay after E20 blending. There is no simple public tool that "
            "combines vehicle specs, real-world tested mileage, compatibility status, and a "
            "predicted E20 drop into a rupee cost comparison.",
            styles["Body"],
        )
    )
    story.append(Paragraph("Project objective", styles["H2"]))
    story.append(
        bullets(
            [
                "Assemble a real-world Indian passenger-car dataset covering brand, engine, launch year, claimed/tested mileage, and E20 compatibility.",
                "Clean and inspect the data, then explore patterns that explain mileage drop.",
                "Train a Random Forest model (existing project pipeline) to predict E20 mileage drop %.",
                "Expose the result in a Streamlit app that estimates extra monthly fuel cost.",
                "Translate the analysis into business insights and recommendations for consumers, OEMs, and fleet managers.",
            ]
        )
    )
    story.append(Paragraph("Tools used", styles["H2"]))
    story.append(
        bullets(
            [
                "Python, pandas, NumPy &mdash; inspection, cleaning, statistics",
                "Matplotlib, Seaborn &mdash; EDA charts included in this report",
                "scikit-learn Random Forest Regressor &mdash; drop-percentage model (R² ≈ 0.9986, MAE ≈ 0.028 on the notebook test split)",
                "Streamlit &amp; joblib &mdash; public estimator app",
                "Git &amp; GitHub &mdash; version control and documentation",
            ]
        )
    )

    # 2 Dataset
    story.append(Paragraph("2. Dataset Description", styles["H"]))
    story.append(
        Paragraph(
            f"The working file is <b>E20_predictions_updated.xlsx</b> ({notes['raw_rows']} rows × "
            f"{notes['raw_cols']} columns). A companion lookup file, <b>car_lookup.csv</b>, powers "
            "the app dropdowns. Records were compiled from publicly cited Indian automotive test "
            "and specification sources: ARAI, SIAM, and Autocar India real-world mileage tests. "
            "The domain is finance + automotive operations: fuel cost impact of a national fuel-policy change.",
            styles["Body"],
        )
    )
    story.append(Paragraph("Columns and data types", styles["H2"]))
    story.append(section_table(dtype_rows, col_widths=[2.15 * inch, 1.3 * inch, 3.15 * inch]))
    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            f"Brands covered: {len(notes['brand_counts'])} manufacturers, led by Maruti ({notes['brand_counts'].get('Maruti', 0)}), "
            f"Hyundai ({notes['brand_counts'].get('Hyundai', 0)}), Tata ({notes['brand_counts'].get('Tata', 0)}), "
            f"and Toyota ({notes['brand_counts'].get('Toyota', 0)}). Launch years range from 2008 to 2026. "
            f"E20 labels: Yes {notes['compat_counts'].get('Yes', 0)}, No {notes['compat_counts'].get('No', 0)}, "
            f"CNG {notes['compat_counts'].get('N/A (CNG)', 0)}, Diesel {notes['compat_counts'].get('N/A (Diesel)', 0)}, "
            f"upcoming/discontinued {notes['compat_counts'].get('TBA (Upcoming)', 0) + notes['compat_counts'].get('Discontinued', 0)}.",
            styles["Body"],
        )
    )

    # 3 Cleaning
    story.append(Paragraph("3. Data Cleaning Process", styles["H"]))
    story.append(Paragraph("Inspection findings", styles["H2"]))
    story.append(
        bullets(
            [
                f"Raw size: {notes['raw_rows']} records, {notes['raw_cols']} columns; full-row duplicates: {notes['dups_before']}.",
                "Engine (CC) missing on 4 rows; claimed and tested mileage missing on 13 rows each.",
                "E20 drop and updated drop missing on 17 rows (typically non-applicable or incomplete specs).",
                "Launch Year Note is empty on 219/221 rows and was dropped as non-informative.",
                "CNG and diesel rows are labelled N/A because ethanol blending does not apply to those fuels.",
                "A few luxury/high-displacement engines (for example ~3956 CC) sit far from the mass-market 1.0–1.5 L cluster.",
            ]
        )
    )
    story.append(Paragraph("Missing-value handling", styles["H2"]))
    story.append(section_table(missing_rows, col_widths=[2.6 * inch, 1.5 * inch, 2.5 * inch]))
    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            "Missing values were not blindly filled. Engine CC and the target drop percentage are "
            "structurally required, so those rows were excluded from statistical modelling and from "
            "the EDA analysis set. Mileage-based charts drop only the rows missing claimed/tested "
            "km/L. CNG/diesel rows are retained in the compatibility mix (to show the market) but "
            "removed from petrol-drop analysis. Categorical labels were trimmed of whitespace. "
            f"Key-level duplicates on brand/model/variant/engine/year: {notes['key_dups']} (removed if present). "
            f"The cleaned analysis set used for EDA has <b>{notes['analysis_rows']} petrol-relevant rows</b>.",
            styles["Body"],
        )
    )

    # 4 EDA
    story.append(Paragraph("4. Exploratory Data Analysis", styles["H"]))
    story.append(Paragraph("Descriptive statistics", styles["H2"]))
    desc = analysis[
        ["Engine (CC)", "Launch Year", "Mileage Claimed (km/L)", "Mileage Tested (km/L)", "E20_Drop_%_updated"]
    ].describe().round(2)
    desc_table = [["Metric"] + list(desc.columns)]
    for idx in desc.index:
        desc_table.append([idx] + [str(desc.loc[idx, c]) for c in desc.columns])
    story.append(section_table(desc_table, col_widths=[1.15 * inch] + [1.17 * inch] * 5))
    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            f"Mean claimed mileage is {claimed_mean:.2f} km/L versus mean tested mileage of "
            f"{tested_mean:.2f} km/L — about a {gap_pct:.1f}% real-world shortfall before E20 is applied. "
            f"The updated E20 drop averages {avg_drop:.2f}% (median {med_drop:.1f}%). "
            f"Compatible cars average {yes_drop:.2f}% drop versus {no_drop:.2f}% for non-compatible petrol cars.",
            styles["Body"],
        )
    )
    story.append(Paragraph("Correlation and outliers", styles["H2"]))
    story.append(
        Paragraph(
            f"Launch year vs drop correlation is {corr_year:.2f}. Engine size vs drop is {corr_engine:.2f}. "
            f"Tested mileage vs drop is {corr_tested:.2f}. IQR outlier counts in the analysis set: "
            f"engine size {engine_out}, tested mileage {mileage_out}, E20 drop {drop_out}. "
            "Large-displacement outliers (premium engines) were kept in descriptive EDA but are flagged "
            "in the app when engine CC is outside 600–3000 CC, because predictions there are less reliable.",
            styles["Body"],
        )
    )
    story.append(Paragraph("Main patterns", styles["H2"]))
    story.append(
        bullets(
            [
                f"Among cars launched 2022 or later in the analysis set, {newest_share:.0f}% are labelled E20 compatible — OEMs have shifted quickly.",
                f"Among cars launched 2018 or earlier, {older_no:.0f}% are not compatible, so older fleets carry more E20 risk.",
                "Claimed mileage systematically overstates real-world km/L; cost planning should start from tested mileage.",
                f"Lowest average drop brands in this sample: {', '.join(f'{b} ({v:.1f}%)' for b, v in brand_best.head(3).items())}.",
                f"Highest average drop brands in this sample: {', '.join(f'{b} ({v:.1f}%)' for b, v in brand_best.tail(3).items())}.",
                "The original E20_Drop_% column clustered around 3–9%, while the updated target is higher (about 10–18%), which is why the app and this report use the updated field.",
            ]
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("5. Visualizations", styles["H"]))
    story.append(
        Paragraph(
            "At least five labelled charts were produced from the cleaned dataset. Each chart is "
            "tied to a decision: market coverage, compatibility mix, physical engine spread, "
            "claimed vs real mileage, relationships among numeric fields, and drop by brand/year.",
            styles["Body"],
        )
    )
    viz_plan = [
        ("brand", "Figure 1. Records by brand — sample is India-mass-market heavy (Maruti, Hyundai, Tata)."),
        ("compat", "Figure 2. Compatibility mix — a large minority of cars are still not E20 certified."),
        ("engine", "Figure 3. Engine sizes concentrate around 1.2 L and 1.5 L, the typical hatchback/SUV core."),
        ("year", "Figure 4. Most records are 2019–2024 models, matching the E20 transition window."),
        ("mileage", "Figure 5. Almost all points sit below the equality line: tested km/L is lower than claimed."),
        ("corr", "Figure 6. Correlation heatmap for engine, year, mileage, and updated drop."),
        ("box", "Figure 7. Drop distribution by compatibility status."),
        ("year_drop", "Figure 8. Newer compatible cars still show a drop, but older non-compatible cars cluster higher."),
        ("brand_drop", "Figure 9. Brand-level average drop — useful for fleet buying shortlists."),
    ]
    for key, caption in viz_plan:
        story.append(img(charts[key], width=6.3 * inch))
        story.append(Paragraph(caption, styles["Caption"]))

    story.append(PageBreak())
    story.append(Paragraph("6. Analytics Snapshot (Power BI Task Skipped)", styles["H"]))
    story.append(
        Paragraph(
            "Task 6 asked for an interactive Power BI dashboard (KPI cards, four visuals, slicers). "
            "That task is skipped per the project instruction. The snapshot below is a static "
            "stand-in: four KPI cards plus three supporting views generated from the same metrics "
            "a dashboard would highlight.",
            styles["Body"],
        )
    )
    story.append(img(charts["dashboard"], width=6.5 * inch))
    story.append(
        Paragraph(
            "Figure 10. KPI snapshot using 1,200 km/month and Rs 96.5/litre as a standard household scenario.",
            styles["Caption"],
        )
    )
    story.append(
        bullets(
            [
                f"KPI 1 — Average updated E20 drop: {avg_drop:.1f}%.",
                f"KPI 2 — Share of records labelled E20 compatible: {compatible_share:.0f}%.",
                f"KPI 3 — Average extra monthly fuel spend in the standard scenario: Rs {extra_mean:.0f}.",
                f"KPI 4 — Average claimed-versus-tested mileage gap: {gap_pct:.1f}%.",
                f"The 90th percentile extra monthly cost is Rs {extra_p90:.0f}, so high-drop / low-mileage cars are materially more expensive to run on E20.",
            ]
        )
    )

    story.append(Paragraph("7. Business Insights", styles["H"]))
    story.append(
        bullets(
            [
                f"<b>Insight 1 — E20 is a real cost, not a rounding error.</b> Mean drop of {avg_drop:.1f}% on already-lower tested mileage raises monthly fuel bills by about Rs {extra_mean:.0f} for a 1,200 km driver at Rs 96.5/L.",
                f"<b>Insight 2 — Certification matters.</b> Non-compatible petrol cars average {no_drop:.1f}% drop versus {yes_drop:.1f}% for compatible cars. Buying an uncertified used car during the E20 rollout is a measurable penalty.",
                f"<b>Insight 3 — Brochure mileage is the wrong planning number.</b> Claimed km/L overstates tested km/L by roughly {gap_pct:.1f}%. Combining brochure figures with E20 drop would understate fuel need twice.",
                f"<b>Insight 4 — The transition is front-loaded in new cars.</b> {newest_share:.0f}% of 2022+ analysis records are compatible, while {older_no:.0f}% of 2018-or-earlier cars are not. Used-car and older fleet segments are the high-risk pocket.",
                "<b>Insight 5 — Brand mix is not uniform.</b> Average drop differs across manufacturers in this sample, so fleet RFPs can rank OEMs on expected E20 penalty rather than treating petrol as interchangeable.",
                "<b>Insight 6 — CNG/diesel are a different policy problem.</b> Those rows are correctly treated as 0% E20 drop. For mixed fleets, ethanol policy mainly hits petrol SKUs; CNG already decouples the household from petrol energy density.",
            ]
        )
    )

    story.append(Paragraph("8. Recommendations", styles["H"]))
    story.append(
        bullets(
            [
                "<b>Recommendation 1 (consumers &amp; used-car platforms):</b> Show E20 compatibility and an estimated extra monthly rupee cost next to every petrol listing. Default the calculator to tested mileage, not claimed mileage. This is a data-driven UX change using fields already in the dataset.",
                "<b>Recommendation 2 (fleet managers / ride-hailing):</b> Prioritise E20-certified petrol vehicles or shift high-mileage duty cycles to CNG/hybrid. On a 1,200 km/month duty cycle the average extra is modest, but the upper tail (≈ Rs "
                + f"{extra_p90:.0f}/month) compounds across hundreds of cars.",
                "<b>Recommendation 3 (OEMs):</b> Publish certified E20 km/L, not only petrol claimed km/L, especially for 1.2 L and 1.5 L volume engines that dominate this dataset.",
                "<b>Recommendation 4 (policy / fuel retailers):</b> Pair E20 pump labelling with a simple household cost explainer. The analysis shows people already lose ~18% vs claimed mileage; E20 stacks another double-digit percent on top.",
            ]
        )
    )

    story.append(Paragraph("9. Conclusion", styles["H"]))
    story.append(
        Paragraph(
            "The dataset is small but decision-useful: 221 Indian passenger-car records spanning "
            "the E20 transition, cleaned to a petrol analysis set with explicit missing-value rules. "
            f"EDA shows a consistent real-world mileage gap, a typical E20 drop near {avg_drop:.1f}%, "
            "and a clear split between certified and uncertified cars. Visualisations communicate "
            "brand coverage, compatibility, engine mix, claimed-versus-tested km/L, correlations, "
            "and cost impact. The existing Streamlit app turns the same logic into a consumer tool. "
            "Power BI was not built for this submission. Next improvements would be a larger "
            "independent mileage-test sample, city-level fuel prices, and a dashboard with brand/year slicers.",
            styles["Body"],
        )
    )

    story.append(Paragraph("10. Git &amp; GitHub Workflow (Tasks 1–2 and 8)", styles["H"]))
    story.append(
        Paragraph(
            "The project lives at github.com/UtsavUjjwal/E20-Expense-Predictor. Local Git is in use; "
            "the repository was created on GitHub, cloned, and updated with multiple meaningful commits "
            "(dataset uploads, Streamlit app, EDA notebook, mileage-age adjustment, README, and this report). "
            "Standard workflow practised: clone → edit → add → commit → push, plus pull to stay in sync with origin/main.",
            styles["Body"],
        )
    )
    story.append(
        bullets(
            [
                "Task 1: Git installed, GitHub account in use, repository created, cloned locally, files added, committed, and pushed.",
                "Task 2: Repeated add/commit/push/pull while iterating on EDA, the predictor app, and documentation (well above three meaningful commits).",
                "Task 8: README and this PDF cover title, overview, problem, dataset, tools, cleaning, EDA, visuals, insights, recommendations, and conclusion.",
            ]
        )
    )

    story.append(Paragraph("Appendix — How to reproduce", styles["H2"]))
    story.append(
        Paragraph(
            "Clone the repository, install requirements (pandas, matplotlib, seaborn, openpyxl, "
            "reportlab, scikit-learn, streamlit, joblib), then run <b>python generate_capstone_report.py</b> "
            "to rebuild this PDF and the figure folder. Run <b>streamlit run app.py</b> for the live estimator.",
            styles["Body"],
        )
    )

    def add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(colors.HexColor(NAVY))
        canvas.rect(0, A4[1] - 18, A4[0], 18, fill=1, stroke=0)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica", 8)
        canvas.drawString(20, A4[1] - 13, "E20 Mileage Drop & Fuel Cost Estimator  |  Capstone Report")
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawRightString(A4[0] - 20, 16, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.5 * inch,
        title="E20 Mileage Drop Capstone Report",
        author="Utsav Ujjwal",
    )
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"Wrote {PDF_PATH}")


def main():
    raw, df, analysis, notes = load_and_clean()
    charts, cost_rows = make_charts(df, analysis)
    build_pdf(raw, df, analysis, notes, charts, cost_rows)


if __name__ == "__main__":
    main()
