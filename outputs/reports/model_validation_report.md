# Model Validation & Comparative Evaluation Report

## Evaluation Strategy & Imbalance Considerations
The IBM HR Analytics dataset contains an inherent **5.2 : 1 class imbalance** (16.12% turnover rate).
When evaluating attrition models in an enterprise human resources context:
- **Cost of False Negatives (Missed Leavers)**: High (unplanned turnover, institutional knowledge loss, recruitment and training costs averaging 1.5x - 2x annual salary).
- **Cost of False Positives (Surplus Retention Focus)**: Low (retention outreach, 1-on-1 check-ins, professional growth conversations).

Consequently, **Accuracy alone is misleading** (a naive stayer baseline attains 83.9% accuracy with 0% attrition recall). Model selection strictly prioritizes **Attrition Recall** and **F1-Score**, supported by **ROC-AUC**.

---

## Validation Performance Table (15% Unseen Validation Split)

| Algorithm | Val Accuracy | Val Precision | Val Recall | Val F1-Score | Val ROC-AUC | 5-Fold CV Mean Recall | 5-Fold CV Mean F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.7773 | 0.3833 | **0.6571** | **0.4842** | 0.8048 | 0.7649 | 0.5114 |
| **Decision Tree** | 0.6818 | 0.2840 | **0.6571** | **0.3966** | 0.6952 | 0.5476 | 0.3807 |
| **Random Forest** | 0.8318 | 0.4783 | **0.6286** | **0.5432** | 0.7243 | 0.5417 | 0.5477 |

---

## Model Selection Decision
- **Champion Selected**: **`Logistic Regression`**
- **Rationale**: Demonstrates the optimal balance of sensitivity to departure risks (capturing the majority of employees planning to leave) while maintaining high discriminant capability (ROC-AUC) and stability across 5-fold cross-validation.
- The fitted model pipeline has been saved to: `models/trained_models/best_model_pipeline.joblib`.
