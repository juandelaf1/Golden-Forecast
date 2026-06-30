# Golden Forecast - DataScope Solutions

## Context

We are **DataScope Solutions**, an international data analytics consultancy. This project applies supervised Machine Learning to historical gold price data to generate explainable predictions.

## Team

| Role | Name |
|------|------|
| Product Owner | María |
| Scrum Master | Juan |
| Development Team | José, Gema, Joel |

## Business Problem

Predict the price movement of gold (GC=F) using historical data. We approach the problem from two angles:

- **Regression**: predict the closing price (continuous numerical value)
- **Classification**: predict whether the price will go up or down next day (binary)

## Dataset

Historical data of **GC=F** (Gold Futures) via Yahoo Finance, from 2015 to present. Main variables: Open, High, Low, Close, Volume. Enriched with SP500 and USD Index as external features.

## Tech Stack

Python · pandas · numpy · scikit-learn · matplotlib · seaborn · yfinance

## Repository Structure

```
golden-forecast/
├── data/                # Raw and processed data
│   └── README.md        # Data lineage
├── notebooks/           # Pipeline notebooks
├── src/                 # Reusable functions
├── models/              # Trained models (.pkl)
├── slides/              # Final presentation
├── docs/
│   ├── project_handbook.md   # Project governance
│   ├── data_dictionary.md    # Variable definitions
│   └── decision_log.md       # Decision registry
├── .github/
│   └── PULL_REQUEST_TEMPLATE.md
├── README.md
├── ROADMAP.md
├── requirements.txt
└── .gitignore
```

## How to Run

1. Clone the repository
2. `pip install -r requirements.txt`
3. Open and run notebooks in numerical order

## Governance

This project follows an Agile (Scrum) methodology with:
- **Sprint planning** and daily syncs
- **GitHub Flow**: feature branches + PRs with peer review
- **Protected `main`** branch (no direct pushes)
- **Decision Log** documenting all key technical choices
- **Data lineage** tracked for full reproducibility

## License

Academic project for non-commercial use.
