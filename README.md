# HR Employee Attrition Analysis and Prediction

Trained and compared **Logistic Regression**, **SVM (Linear, Polynomial, RBF, Sigmoid)**, and **Decision Tree** models on the HR Employee Attrition dataset — achieving **88% accuracy** after **Decision Tree pruning (pre- and post-pruning)**.  
This was my **first Machine Learning project**,which I built in first year as part of an **IIT Kanpur ML course**, where I explored end-to-end data cleaning, feature analysis, model evaluation, and Flask deployment.  
Further experiments and model improvements were added beyond the course scope.

---

## Data Analysis and Preprocessing

- Dataset used: **HR Employee Attrition.csv**
- Loaded and analyzed dataset using **Pandas**, **NumPy**, and **Statsmodels**.
- Checked data types, null values, and feature distributions.
- Treated **outliers** using **IQR-based clipping** for numerical columns such as:
  - `MonthlyIncome`
  - `NumCompaniesWorked`
  - `TotalWorkingYears`
- Encoded categorical columns and normalized numerical features.
- Visualized feature relationships using **Seaborn** correlation heatmaps and boxplots.

---

## Statistical Analysis

- Performed **OLS regression** via **Statsmodels** to evaluate feature significance.
- Identified strong predictors of attrition such as:
  - `JobSatisfaction`
  - `MonthlyIncome`
  - `YearsSinceLastPromotion`
- Removed statistically weaker variables to stabilize the model and avoid redundancy.

---

## Model Development

Implemented and compared multiple models:

1. **Logistic Regression**
   - Baseline model for binary classification.
   - Simple, interpretable, and effective for benchmarking.

2. **Support Vector Machine (SVM)**
   - Tested with **Linear**, **Polynomial**, **RBF**, and **Sigmoid** kernels.
   - RBF kernel performed best among all, but still underperformed in recall.
   - Required normalization and parameter tuning.

3. **Decision Tree Classifier**
   - Final chosen model for deployment.
   - Applied both **Pre-Pruning** (max depth, min samples split) and **Post-Pruning** (cost-complexity pruning).
   - Pre-pruning reduced overfitting early; post-pruning optimized the final complexity.
   - Produced highly interpretable decision rules for HR teams.

---

## Model Evaluation

Evaluated models using:
- **Confusion Matrix**
- **Classification Report**
- **Accuracy, Precision, Recall, F1-score (for Attrition = 1)**

| Model | Accuracy | Precision (Attrition = 1) | Recall (Attrition = 1) | F1-score (Attrition = 1) |
|:------|:---------:|:--------------------------:|:----------------------:|:------------------------:|
| Logistic Regression | ~86.7% | ~75% | ~33% | ~45% |
| SVM (RBF Kernel) | ~84% | Low recall — not used |
| SVM (Poly Kernel) | ~84% | Low precision — not used |
| SVM (Sigmoid Kernel) | ~79% | Low accuracy — skipped |
| Decision Tree (Pre + Post-Pruned) | **~88%** | **~75%** | **~50%** | **~60%** |

- Final deployed model: **Decision Tree (with pre- and post-pruning)**  
- Saved as `model.pkl` and integrated into Flask backend.

---

## Testing and Validation

- Verified predictions on unseen HR data.
- Cross-checked consistency across multiple runs and pruning thresholds.
- Compared learning curves before and after pruning — confirmed reduced variance and improved generalization.

---

## Insights

- Employees with **low satisfaction**, **low income**, and **fewer promotions** showed the highest attrition probability.
- **Pre-pruning** prevented overfitting early, while **post-pruning** fine-tuned the tree depth and complexity.
- Despite SVM’s stronger boundary separation, Decision Tree was chosen for **explainability** and better real-world interpretability.

---

## ⚙️ Deployment

- Integrated model with a **Flask web app (`app.py`)**.
- User inputs HR attributes via frontend (IIT Kanpur-provided UI).
- The app loads `model.pkl` dynamically to give instant “Stay / Leave” predictions.

---

## Final Outcome

- Completed an **end-to-end ML pipeline**:
  1. EDA and outlier removal  
  2. Statistical feature analysis  
  3. Model comparison (Logistic, SVM, Decision Tree)  
  4. Pre- and Post-pruned Decision Tree deployment  
- Achieved **~88% accuracy** with strong interpretability.
- Fully functional web app integrated with Flask for live prediction.
- This project helped me build my foundation in Machine Learning concepts, model optimization, and practical deployment

---

**Author:** AJ  
**Final Model:** Decision Tree (Pre + Post-Pruned)  
**Peak Accuracy:** ~88%  
**Libraries Used:** Pandas, NumPy, Seaborn, Statsmodels, Scikit-learn, Flask  
**Dataset Source:** IIT Kanpur ML Course
