# Final Model Evaluation Report (Unseen Test Partition)

## Executive Performance Summary
The champion machine learning model (**`Logistic Regression`**) was evaluated on the **15% unseen test partition (221 employee records)**. This partition was held out throughout the entire preprocessing, exploratory analysis, and training cycles.

| Evaluation Metric | Test Score | Strategic HR Meaning |
| :--- | :---: | :--- |
| **Accuracy** | **74.66%** | Overall fraction of correct predictions across stayers and leavers. |
| **Recall (Sensitivity)** | **72.22%** | Ability to detect employees who actually depart the organization. |
| **Precision** | **36.11%** | Proportion of predicted departures that were genuine leavers. |
| **F1-Score** | **0.4815** | Harmonic mean of precision and recall on the positive minority class. |
| **ROC-AUC** | **0.8365** | Area under the ROC Curve; measures ranking discrimination quality. |

---

## Test Confusion Matrix
```text
                  Predicted Stay (0)   Predicted Leave (1)
Actual Stay (0)          139                 46         (False Positives)
Actual Leave (1)         10                  26         (True Positives)
```

- **True Negatives (139)**: Retained employees correctly identified.
- **True Positives (26)**: Departing employees successfully flagged for retention interventions.
- **False Positives (46)**: Retained employees predicted as leaving (moderate cost: routine check-in).
- **False Negatives (10)**: Departing employees missed by the model (high cost: unexpected departure).

---

## Detailed Classification Report
```text
              precision    recall  f1-score   support

  Stayed (0)       0.93      0.75      0.83       185
    Left (1)       0.36      0.72      0.48        36

    accuracy                           0.75       221
   macro avg       0.65      0.74      0.66       221
weighted avg       0.84      0.75      0.78       221

```

---

## Saved Artifacts
- Confusion Matrix Plot: `outputs/visualizations/test_confusion_matrix.png`
- ROC Curve Plot: `outputs/visualizations/roc_curve_test.png`
- JSON Metrics: `outputs/metrics/test_metrics.json`
