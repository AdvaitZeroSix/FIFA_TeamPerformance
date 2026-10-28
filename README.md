# FIFA Team Performance Predictor

A Streamlit app that sorts a FIFA 23 team into a performance tier (Weak, Average, Good or Elite) from its attack, midfield and defence ratings, ages and prestige. It also has a side-by-side comparison of any two teams.

Live app: https://advaitsaxena.streamlit.app/

## How it works

- **Data:** `data/male_teams.csv` from the FIFA 23 Complete Player Dataset. It has 107,596 rows but only 1,499 distinct teams, because each team appears many times across squad snapshots.
- **Label:** the tier is made by cutting the team's `overall` rating into bands: below 75 is Weak, 75 to 79 is Average, 80 to 84 is Good and 85 and above is Elite.
- **Features:** attack, midfield, defence, starting XI age, whole team age, international prestige and domestic prestige.
- **Models:** Random Forest and Decision Tree, trained in `notebooks/fifa_analysis.ipynb` on an 80/20 split.

| Model | Accuracy |
| ----- | -------- |
| Random Forest | 99.2% |
| Decision Tree | 98.7% |

## What the accuracy does and doesn't mean

These scores are high for two reasons, so they are not a measure of real forecasting:

1. The label comes from `overall`, and attack, midfield and defence are the ratings that `overall` is built from. The model mostly learns to rebuild the rating bands.
2. The split is random over rows, so snapshots of the same team land in both train and test.

A fairer test would predict something the ratings don't define, such as league finish, and split by team.

## Run it locally

```bash
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Files

```text
app.py            Streamlit app (prediction, charts, comparison, model summary)
model.pkl         trained Random Forest (scikit-learn 1.9.0)
data/             team ratings CSV
notebooks/        data cleaning, charts and model training
images/           charts shown in the app
```

`model.pkl` was saved with scikit-learn 1.9.0, so keep that version when loading it.
