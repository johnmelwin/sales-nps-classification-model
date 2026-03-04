# Sales Representative Net Promoter Score (NPS) Prediction

> Binary classification of sales representative NPS scores (>= 9 vs. < 9) using logistic regression and ensemble stacking, with Tableau dashboards for exploratory analysis.

---

## Overview

This project identifies the key drivers of high Net Promoter Scores among college-educated sales representatives in a software product group. The goal is to predict whether a sales rep will achieve an NPS of 9 or higher based on demographics, experience, salary, certifications, feedback, and personality type.

**Problem type**: Binary classification (NPS >= 9 → Positive class)

## Key Findings

| Factor | Impact on NPS |
|--------|--------------|
| **Personality type** | "Diplomat" and "Explorer" types are significantly more likely to achieve NPS >= 9 |
| **Experience level** | Senior reps outperform entry-level and mid-level counterparts |
| **Salary** | Higher compensation correlates with higher NPS scores |
| **Certifications** | More certifications → higher credibility → better NPS |
| **Feedback history** | Reps with excellent positive feedback are most likely to score >= 9 |

## Methodology

```
Data Subsetting → Binary Target Creation → Class Balancing → Model Training → Cross-Validation → Evaluation
```

1. **Data Subsetting** — Filter to Software business group, college-educated employees
2. **Target Engineering** — Binary variable: NPS >= 9 → 1, otherwise 0
3. **Class Balancing** — Up-sampling of minority class (configurable ratio)
4. **Model Training** — Train logistic regression and ensemble stacking models
5. **Validation** — 5-fold cross-validation with accuracy, precision, recall, F1, and AUC-ROC
6. **Prediction** — Generate predictions for new sales representative profiles

## Models

### Logistic Regression (`models/logistic_regression.R`)
- Baseline binary classifier using `glm` (binomial family)
- Manual 5-fold cross-validation with per-fold metrics
- 20% minority up-sampling via `caret`
- **Predictors**: Age, Years, Personality, Salary, Certifications, Feedback, Gender

### Ensemble Stacking (`models/model_stacking_ensemble.R`) — *Selected Model*
- Three base learners trained via `caretList` (5-fold CV, 2 repeats):
  - **Decision Tree** (`rpart`)
  - **Logistic Regression** (`glm`)
  - **Naive Bayes** (`nb`)
- **GLM meta-learner** stacks base predictions (5-fold CV, 3 repeats)
- Full confusion matrix + ROC/AUC curve on held-out test set
- Libraries: `caret`, `caretEnsemble`, `ROSE`, `pROC`

### Additional Models (`models/decision_tree_naive_bayes.zip`)
- Standalone Decision Tree and Naive Bayes implementations (archived)

## Project Structure

```
├── models/
│   ├── model_stacking_ensemble.R       # Ensemble stacking (selected model)
│   ├── logistic_regression.R           # Logistic regression baseline
│   └── decision_tree_naive_bayes.zip   # Additional DT & NB models
│
├── reports/
│   ├── executive_summary.pdf           # Full findings and recommendations
│   ├── sales_dashboard.pdf             # Tableau dashboard (PDF export)
│   ├── sales_dashboard.twb             # Tableau workbook (interactive)
│   └── pivot_table_analysis.xlsx       # Excel pivot table analysis + data
│
└── README.md
```

## Recommendations

Based on the analysis, five actionable strategies to improve NPS:

1. **Recruit for personality fit** — Prioritize "Diplomat" and "Explorer" personality types for sales roles
2. **Competitive compensation** — Higher salaries correlate with improved motivation and NPS outcomes
3. **Training investment** — Develop skills and domain knowledge to enhance performance
4. **Leverage feedback** — Identify and promote reps with consistently excellent positive feedback
5. **Certification programs** — Encourage professional certifications to build credibility and customer trust

## Visualizations

| Dashboard Overview | NPS Distribution |
|---|---|
| ![Dashboard](https://github.com/johnmelwin/SalesNPSAnalysis/assets/42464701/e1f0ed4e-6a8b-4b69-8614-7054d066919d) | ![Analysis](https://github.com/johnmelwin/SalesNPSAnalysis/assets/42464701/0ec04bd8-bc29-4bc5-90e6-15787b9ca390) |

## Tech Stack

- **Language**: R (`caret`, `caretEnsemble`, `pROC`, `ROSE`, `dplyr`)
- **Visualization**: Tableau, Excel
- **Methods**: Logistic regression, decision trees, Naive Bayes, model stacking, cross-validation

## Getting Started

1. Clone the repository
   ```bash
   git clone https://github.com/johnmelwin/SalesNPSAnalysis.git
   cd SalesNPSAnalysis
   ```

2. Open the R scripts in RStudio and load data from `reports/pivot_table_analysis.xlsx`

3. View the interactive dashboard: open `reports/sales_dashboard.twb` in Tableau, or see `reports/sales_dashboard.pdf` for the static version
