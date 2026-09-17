from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DATA_PATH = Path(__file__).resolve().parent / "crypto_historical_365days.csv"


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["coin_id", "date"]).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def prepare_model_data(df: pd.DataFrame):
    work = df.copy()
    work["market_cap_rank"] = pd.to_numeric(work["market_cap_rank"], errors="coerce")
    work["price"] = pd.to_numeric(work["price"], errors="coerce")
    work["market_cap"] = pd.to_numeric(work["market_cap"], errors="coerce")
    work["volume"] = pd.to_numeric(work["volume"], errors="coerce")
    work["daily_return"] = pd.to_numeric(work["daily_return"], errors="coerce")
    work["price_ma7"] = pd.to_numeric(work["price_ma7"], errors="coerce")
    work["price_ma30"] = pd.to_numeric(work["price_ma30"], errors="coerce")
    work["volatility_7d"] = pd.to_numeric(work["volatility_7d"], errors="coerce")
    work["cumulative_return"] = pd.to_numeric(work["cumulative_return"], errors="coerce")

    work["next_day_price"] = work.groupby("coin_id")["price"].shift(-1)
    work["next_day_return"] = (work["next_day_price"] - work["price"]) / work["price"].replace(0, np.nan)
    work["target"] = (work["next_day_return"] > 0).astype(int)

    work["volume_to_marketcap"] = np.where(
        work["market_cap"].replace(0, np.nan).notna(),
        work["volume"] / work["market_cap"],
        np.nan,
    )
    work["price_vs_ma7"] = (work["price"] - work["price_ma7"]) / work["price_ma7"].replace(0, np.nan)
    work["price_vs_ma30"] = (work["price"] - work["price_ma30"]) / work["price_ma30"].replace(0, np.nan)
    work["day_of_week"] = work["date"].dt.dayofweek
    work["day_of_month"] = work["date"].dt.day
    work["month_num"] = work["date"].dt.month
    work["is_month_end"] = work["date"].dt.is_month_end.astype(int)

    feature_cols = [
        "price",
        "market_cap",
        "volume",
        "daily_return",
        "price_ma7",
        "price_ma30",
        "volatility_7d",
        "cumulative_return",
        "volume_to_marketcap",
        "price_vs_ma7",
        "price_vs_ma30",
        "day_of_week",
        "day_of_month",
        "month_num",
        "is_month_end",
        "market_cap_rank",
    ]

    for col in feature_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
        med = work[col].median()
        work[col] = work[col].fillna(med)

    analysis_df = work.dropna(subset=[*feature_cols, "target"]).copy()
    analysis_df = analysis_df.sort_values(["date", "coin_id"]).reset_index(drop=True)

    split_date = analysis_df["date"].sort_values().iloc[int(len(analysis_df) * 0.8)]
    train_mask = analysis_df["date"] < split_date
    test_mask = analysis_df["date"] >= split_date

    X_train = analysis_df.loc[train_mask, feature_cols]
    y_train = analysis_df.loc[train_mask, "target"]
    X_test = analysis_df.loc[test_mask, feature_cols]
    y_test = analysis_df.loc[test_mask, "target"]

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
    }

    return {
        "df": analysis_df,
        "feature_cols": feature_cols,
        "split_date": split_date,
        "model": model,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": pred,
        "y_proba": proba,
        "metrics": metrics,
    }


@st.cache_data(show_spinner=False)
def get_coin_summary(df: pd.DataFrame):
    summary = (
        df.groupby("coin_id")
        .agg(
            avg_daily_return=("daily_return", "mean"),
            avg_volatility=("volatility_7d", "mean"),
            avg_market_cap=("market_cap", "mean"),
            total_return=("cumulative_return", "last"),
            max_price=("price", "max"),
            min_price=("price", "min"),
        )
        .reset_index()
    )
    summary = summary.sort_values("total_return", ascending=False)
    return summary


@st.cache_data(show_spinner=False)
def get_market_comparison(df: pd.DataFrame):
    market = (
        df.groupby("coin_id")
        .agg(
            avg_return=("daily_return", "mean"),
            avg_volatility=("volatility_7d", "mean"),
            avg_market_cap=("market_cap", "mean"),
            final_price=("price", "last"),
            total_return=("cumulative_return", "last"),
        )
        .reset_index()
    )
    return market


def render_overview(df: pd.DataFrame):
    st.title("CryptoInsight AI Dashboard")
    st.subheader("Overview")

    total_coins = df["coin_id"].nunique()
    date_range = df["date"].min().strftime("%Y-%m-%d") + " to " + df["date"].max().strftime("%Y-%m-%d")
    avg_daily_return = df["daily_return"].mean() * 100
    avg_volatility = df["volatility_7d"].mean()
    top_coin = get_coin_summary(df).iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Coins", f"{total_coins}")
    col2.metric("Date range", date_range)
    col3.metric("Avg daily return", f"{avg_daily_return:.2f}%")
    col4.metric("Top cumulative winner", top_coin["coin_id"])

    summary = get_coin_summary(df)
    best = summary.head(10)

    st.markdown("### Top-performing assets by cumulative return")
    st.dataframe(best[["coin_id", "total_return", "avg_daily_return", "avg_volatility"]], use_container_width=True)

    fig = px.bar(
        best,
        x="coin_id",
        y="total_return",
        title="Top 10 cumulative returns by coin",
        color="total_return",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


def render_eda(df: pd.DataFrame):
    st.subheader("Exploratory Data Analysis")
    coin = st.selectbox("Select coin", sorted(df["coin_id"].unique()))
    coin_df = df[df["coin_id"] == coin].sort_values("date").copy()

    col1, col2 = st.columns(2)
    with col1:
        price_fig = px.line(coin_df, x="date", y="price", title=f"{coin} price history")
        st.plotly_chart(price_fig, use_container_width=True)
    with col2:
        return_fig = px.line(coin_df, x="date", y="daily_return", title=f"{coin} daily returns")
        st.plotly_chart(return_fig, use_container_width=True)

    st.markdown("### Return distribution")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(coin_df["daily_return"].dropna(), bins=40, kde=True, ax=ax)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    st.pyplot(fig)

    st.markdown("### Correlation heatmap (selected numerical columns)")
    corr_cols = [
        "price",
        "market_cap",
        "volume",
        "daily_return",
        "price_ma7",
        "price_ma30",
        "volatility_7d",
        "cumulative_return",
    ]
    corr = coin_df[corr_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)


def render_comparison(df: pd.DataFrame):
    st.subheader("Asset Comparison")
    metric = st.selectbox("Metric", ["avg_daily_return", "avg_volatility", "avg_market_cap", "total_return"])
    top_n = st.slider("Top N coins", 5, 20, 10)

    comparison = get_market_comparison(df)
    chart_df = comparison.sort_values(metric, ascending=False).head(top_n)

    fig = px.bar(
        chart_df,
        x="coin_id",
        y=metric,
        title=f"Top {top_n} coins by {metric}",
        color=metric,
        color_continuous_scale="Turbo",
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(chart_df[["coin_id", "avg_return", "avg_volatility", "avg_market_cap", "total_return"]], use_container_width=True)


def render_prediction(df: pd.DataFrame):
    st.subheader("AI Next-Day Price Movement Prediction")
    prepared = prepare_model_data(df)
    model = prepared["model"]
    feature_cols = prepared["feature_cols"]
    all_data = prepared["df"]

    coin_select = st.selectbox("Choose coin", sorted(all_data["coin_id"].unique()))
    selected_date = st.selectbox(
        "Choose a date",
        sorted(all_data.loc[all_data["coin_id"] == coin_select, "date"].dt.strftime("%Y-%m-%d").unique())[-30:],
    )

    row = all_data[(all_data["coin_id"] == coin_select) & (all_data["date"].dt.strftime("%Y-%m-%d") == selected_date)].iloc[0]
    features = row[feature_cols].to_frame().T
    probability = model.predict_proba(features)[0, 1]
    prediction = int(probability >= 0.5)

    st.markdown(f"### Prediction for {coin_select} on {selected_date}")
    st.info(f"Model probability of next-day price increase: {probability:.2%}")
    if prediction == 1:
        st.success("Prediction: Next day is likely UP")
    else:
        st.warning("Prediction: Next day is likely DOWN")

    st.dataframe(features.round(6), use_container_width=True)


def render_model_evaluation(df: pd.DataFrame):
    st.subheader("Model Evaluation")
    prepared = prepare_model_data(df)
    metrics = prepared["metrics"]
    y_test = prepared["y_test"]
    y_pred = prepared["y_pred"]
    model = prepared["model"]
    feature_cols = prepared["feature_cols"]

    st.markdown("### Evaluation metrics")
    metric_cols = st.columns(5)
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    for idx, metric_name in enumerate(metric_names):
        metric_cols[idx].metric(metric_name, f"{metrics[metric_name]:.3f}")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Down", "Up"], yticklabels=["Down", "Up"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    st.pyplot(fig)

    importances = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    st.markdown("### Feature importance")
    fig = px.bar(
        importances.head(10),
        x="importance",
        y="feature",
        orientation="h",
        title="Top 10 model features by importance",
        color="importance",
        color_continuous_scale="Plasma",
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


def render_data_quality(df: pd.DataFrame):
    st.subheader("Data Quality")
    missing = df.isna().sum().sort_values(ascending=False)
    missing_df = missing[missing > 0].reset_index()
    missing_df.columns = ["column", "missing_values"]

    duplicates = df.duplicated().sum()
    invalid_dates = df["date"].isna().sum()
    const_cols = [col for col in df.columns if df[col].nunique(dropna=True) <= 1]

    col1, col2, col3 = st.columns(3)
    col1.metric("Duplicate rows", int(duplicates))
    col2.metric("Null dates", int(invalid_dates))
    col3.metric("Constant columns", len(const_cols))

    if missing_df.empty:
        st.success("No missing values detected in the dataset.")
    else:
        st.dataframe(missing_df, use_container_width=True)

    st.markdown("### Null value heatmap")
    null_matrix = df.isna()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(null_matrix.astype(int), cmap="magma", cbar=False, ax=ax)
    ax.set_title("Null value map")
    st.pyplot(fig)

    st.markdown("### Duplicate and constant checks")
    st.write({"constant_columns": const_cols})


def main():
    st.set_page_config(page_title="CryptoInsight AI", layout="wide")
    df = load_dataset()

    pages = {
        "Overview": lambda: render_overview(df),
        "EDA": lambda: render_eda(df),
        "Comparison": lambda: render_comparison(df),
        "AI Prediction": lambda: render_prediction(df),
        "Model Evaluation": lambda: render_model_evaluation(df),
        "Data Quality": lambda: render_data_quality(df),
    }

    page = st.sidebar.radio("Navigate", list(pages.keys()))
    pages[page]()


if __name__ == "__main__":
    main()
