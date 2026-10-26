# Road accident severity prediction

An educational machine-learning project comparing logistic regression, a decision
tree, and random forest to predict accident severity. The best candidate is chosen
using stratified five-fold macro-F1 cross-validation on the training set. The held-out
20% test set is used only for final evaluation.

## Data

The project runs out of the box using a **synthetic demonstration dataset**. Its
severity labels are generated from a simple risk-score rule; the reported scores do
not demonstrate real-world performance. For a real study, use an appropriately
documented road-accident dataset and cite its source, collection period, geography,
license, and any filtering decisions.

The Streamlit dashboard accepts a CSV upload. The target can be any column with at
least two classes and at least five records per class. Numeric inputs are median
imputed and scaled; categorical inputs are imputed and one-hot encoded inside each
cross-validation pipeline.

## Install and run

Python 3.9 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_model.py
streamlit run streamlit_app.py
```

To train from a CSV file instead of the demo data:

```powershell
python train_model.py --data .\data\your_accident_data.csv --target Accident_severity
```

The CSV must have a header row. Each target class needs at least five examples to
support the required stratified five-fold cross-validation.

## Outputs

`python train_model.py` creates:

- `data/road_accidents_demo.csv` when no input CSV is specified
- `outputs/plots/` with class distribution, numeric correlation heatmap, confusion
  matrix, ROC curve, and five-fold model-comparison chart
- `outputs/road_accident_severity_model.joblib`, containing the selected fitted
  pipeline and metadata
- `outputs/road_accident_severity_report.docx`, an A4 report with metrics, figures,
  analysis, limitations, and the CO5 mapping

The interactive app provides dataset checks, model comparison, test metrics, plots,
and a prediction form. Its predictions are educational only and must not be used for
emergency response or decisions about individuals.
