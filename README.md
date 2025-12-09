# Heart Disease Prediction using Machine Learning

## 🎯 Project Goal
To develop a highly accurate and reliable machine learning classification model to predict the presence of heart disease (Target variable) based on a comprehensive set of clinical and demographic features.

## ✨ Key Results and Performance
The project successfully identified the **Random Forest Classifier** as the optimal model after extensive evaluation and hyperparameter tuning.

| Metric | Random Forest (Tuned) |
| :--- | :--- |
| **ROC-AUC** (Mean CV Score) | **1.000 (±0.000)** |
| **Recall** (Mean CV Score) | **0.992 (±0.015)** |
| **Test Set Accuracy** | 1.000 |

*The high Recall score (Sensitivity) is critical for minimizing false negatives (missed cases) in a medical screening context.*

## 🧠 Model and Methodology
### Methodology
1.  **Exploratory Data Analysis (EDA):** Performed data inspection, descriptive statistics, and visualization (e.g., correlation heatmap, age distribution).
2.  **Preprocessing Pipeline:** Applied **`StandardScaler`** for numeric features and **`OneHotEncoder`** for categorical features using an `sklearn.compose.ColumnTransformer`.
3.  **Model Comparison:** Evaluated four baseline models: Logistic Regression, Decision Tree, Random Forest, and SVM.
4.  **Optimization:** Used **`GridSearchCV`** with 5-fold cross-validation, optimizing for **ROC-AUC** to select the final model.

### Key Predictors
Permutation and native feature importance analysis consistently highlighted the most influential factors:
* `Max_heart_rate`
* `oldpeak` (ST depression induced by exercise)
* `age`
* Features related to `chest_pain_type` and `vessels_colored_by_flourosopy`.

## ⚙️ Repository Structure
| File | Description |
| :--- | :--- |
| `heart_disease.ipynb` | The complete Jupyter Notebook containing all EDA, preprocessing steps, model training, evaluation, and feature importance analysis. |
| `heart_disease_model.pkl` | The final, tuned Random Forest model saved using `joblib`. |
| `heart.csv` | The raw dataset used for this analysis. |

## 🚀 How to Run the Project
### Prerequisites
* Python 3.8+
* The following libraries (install via `pip`):
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn joblib
    ```

### Execution
1.  Clone the repository:
    ```bash
    git clone [https://github.com/Ruhil1807/Heart-Disease-Prediction]
    cd heart-Disease-prediction
    ```
2.  Open and run the `heart_disease.ipynb` notebook in a Jupyter environment to step through the analysis, or load the pre-trained model directly using `joblib`.

## 🧑‍💻 Author 
* **Author:** Ruhil Patel