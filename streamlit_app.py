"""Interactive road accident severity model and evaluation dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from modeling import TARGET_COLUMN, generate_demo_dataset, run_experiment

st.set_page_config(page_title="Road accident severity", page_icon=":material/traffic:",
                   layout="wide")


@st.cache_resource
def train_cached(data: pd.DataFrame, target_column: str) -> dict:
    return run_experiment(data, target_column)


st.title("Road accident severity prediction")
st.caption("Decision tree, logistic regression, and random forest — selected using training-only cross-validation.")

with st.sidebar:
    st.header("Dataset")
    uploaded_file = st.file_uploader("Upload a CSV dataset", type=["csv"])
    if uploaded_file is not None:
        try:
            dataset = pd.read_csv(uploaded_file)
        except (UnicodeDecodeError, pd.errors.ParserError) as error:
            st.error(f"Could not read the uploaded CSV: {error}")
            st.stop()
        st.caption("Using your uploaded CSV. Choose the column containing accident severity.")
    else:
        dataset = generate_demo_dataset()
        st.warning("Using synthetic demo data. Results do not represent real-world performance.")
        st.download_button(
            "Download demo CSV",
            dataset.to_csv(index=False).encode("utf-8"),
            file_name="road_accidents_demo.csv",
            mime="text/csv",
        )

    target_default = TARGET_COLUMN if TARGET_COLUMN in dataset.columns else dataset.columns[-1]
    target_column = st.selectbox(
        "Severity target column",
        options=list(dataset.columns),
        index=list(dataset.columns).index(target_default),
    )

try:
    with st.spinner("Preparing data and evaluating the three models..."):
        result = train_cached(dataset, target_column)
except (ValueError, TypeError) as error:
    st.error(str(error))
    st.stop()

info = result["data_info"]
metrics = result["metrics"]
st.subheader("Dataset overview")
st.write(
    f"**{info['rows_used']:,}** usable rows · **{len(result['feature_columns'])}** "
    f"features · **{len(info['class_counts'])}** severity classes"
)
st.dataframe(
    pd.DataFrame(
        [{"Severity": name, "Records": count} for name, count in info["class_counts"].items()]
    ),
    hide_index=True,
    alt="Accident severity class counts",
)
with st.expander("Data quality checks"):
    st.write(f"Duplicate rows removed: {info['duplicate_rows']:,}")
    st.write("Missing values by column:")
    st.dataframe(
        pd.DataFrame(
            [{"Column": column, "Missing values": count}
             for column, count in info["missing_values"].items()]
        ),
        hide_index=True,
        alt="Missing values by dataset column",
    )

st.subheader("Model comparison")
st.write(f"**Selected by 5-fold training CV (macro F1): {result['selected_model']}**")
comparison = pd.DataFrame(
    [
        {
            "Model": name,
            "Mean CV macro F1": values["mean_f1_macro"],
            "CV standard deviation": values["std_f1_macro"],
        }
        for name, values in result["cv_results"].items()
    ]
)
st.dataframe(comparison, hide_index=True, alt="Five-fold model comparison")

st.subheader("Held-out test metrics")
with st.container(horizontal=True):
    st.metric("Accuracy", f"{metrics['accuracy']:.3f}", border=True)
    st.metric("Macro precision", f"{metrics['precision_macro']:.3f}", border=True)
    st.metric("Macro recall", f"{metrics['recall_macro']:.3f}", border=True)
    st.metric("Macro F1", f"{metrics['f1_macro']:.3f}", border=True)
    st.metric("ROC-AUC (macro OVR)", f"{metrics['roc_auc_ovr_macro']:.3f}", border=True)
st.caption("The test set is used once for evaluation, not to choose the model.")

st.subheader("Evaluation plots")
chart_descriptions = {
    "class_distribution": "Bar chart of accident severity class counts",
    "correlation_heatmap": "Correlation heatmap of numeric input features",
    "confusion_matrix": "Confusion matrix for the held-out test set",
    "roc_curve": "One-vs-rest ROC curves for the selected classifier",
    "model_comparison": "Cross-validation macro F1 comparison of candidate models",
}
for row_keys in (
    ("class_distribution", "correlation_heatmap"),
    ("confusion_matrix", "roc_curve"),
    ("model_comparison",),
):
    columns = st.columns(len(row_keys))
    for column, key in zip(columns, row_keys):
        with column:
            st.image(result["chart_images"][key], caption=key.replace("_", " ").title(),
                     alt=chart_descriptions[key])

st.subheader("Try a prediction")
st.caption(
    "Enter one example using the training feature columns. This educational model "
    "must not be used for emergency response or individual safety decisions."
)
feature_values = {}
input_columns = st.columns(2)
numeric_columns = info["numeric_columns"]
categorical_columns = info["categorical_columns"]
for index, column_name in enumerate(result["feature_columns"]):
    column = input_columns[index % 2]
    if column_name in numeric_columns:
        values = pd.to_numeric(dataset[column_name], errors="coerce").dropna()
        minimum = float(values.min())
        maximum = float(values.max())
        default = float(values.median())
        if minimum == maximum:
            feature_values[column_name] = minimum
            column.caption(f"{column_name}: fixed at {minimum:g}")
        else:
            step = (maximum - minimum) / 100 or 1.0
            feature_values[column_name] = column.slider(
                column_name.replace("_", " ").title(),
                min_value=minimum,
                max_value=maximum,
                value=min(max(default, minimum), maximum),
                step=step,
                key=f"input_{column_name}",
            )
    else:
        values = dataset[column_name].dropna().astype(str).unique().tolist()
        if not values:
            values = ["Unknown"]
        feature_values[column_name] = column.selectbox(
            column_name.replace("_", " ").title(),
            options=sorted(values),
            key=f"input_{column_name}",
        )

if st.button("Predict severity", type="primary", icon=":material/analytics:"):
    example = pd.DataFrame([feature_values], columns=result["feature_columns"])
    prediction = result["model"].predict(example)[0]
    st.success(f"Predicted severity: **{prediction}**")
    st.caption("Prediction is for demonstration only; this model is not safety validated.")

st.divider()
st.caption("Run `python train_model.py` to save the trained model, plots, and A4 report.")
