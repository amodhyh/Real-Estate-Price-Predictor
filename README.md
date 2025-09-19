# House Price Prediction using XGBoost

## Amodh Herath


---

## Project Overview
This project aims to predict house prices in South Carolina using machine learning techniques, specifically XGBoost. The workflow includes data preprocessing, exploratory data analysis, feature engineering, and model training/evaluation.

## Dataset Description
- **Source:** `raw/realestate_data_southcarolina_2025.csv`
- **Features:**
  - `listPrice`: Target variable (house price)
  - `type`: Property type (single_family, condos, townhomes, etc.)
  - `beds`, `baths`, `stories`, `sqft`, `year_built`, etc.
- **Note:** Land properties are excluded from analysis as they lack relevant features.

## Installation & Requirements
1. Clone the repository or download the code files.
2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage Instructions
1. Place the raw dataset in the `raw/` directory.
2. Run the Jupyter notebook `house_price_predict_21E045.ipynb` step by step to:
   - Preprocess data
   - Visualize features
   - Train and evaluate models
3. Processed data and results are saved in the `Processed/` directory.

## Project Structure
```
├── house_price_predict_21E045.ipynb   # Main notebook
├── Feature_Selection.ipynb            # Feature selection notebook
├── requirements.txt                   # Python dependencies
├── raw/
│   └── realestate_data_southcarolina_2025.csv
├── Processed/
│   └── realestate_data_southcarolina_2025_processed.csv
```

## Project Architecture
The system follows a modular pipeline for house price prediction:

1. **Data Ingestion**: Raw data is loaded from `raw/realestate_data_southcarolina_2025.csv`.
2. **Preprocessing & Feature Engineering**:
   - Remove irrelevant property types (e.g., land).
   - Encode categorical features (e.g., property type, location).
   - Impute missing values (median/mean for numerical, mode for categorical).
   - Predict missing `year_built` using a RandomForestRegressor.
   - Remove duplicates and outliers.
   - Feature selection using correlation analysis and domain knowledge.
3. **Exploratory Data Analysis (EDA)**:
   - Visualize distributions, correlations, and outliers using Seaborn and Matplotlib.
   - Generate heatmaps and boxplots for feature understanding.
4. **Model Training**:
   - Split data into training and test sets.
   - Train an XGBoost regressor with optimized hyperparameters.
   - Evaluate using RMSE, MAE, and R² metrics.
   - Analyze feature importance from the trained model.
5. **Prediction & Results**:
   - Predict house prices on test data.
   - Visualize predictions vs. actual values.
   - Save processed data and results in the `Processed/` directory.

**Workflow Diagram (Textual):**
```
Raw CSV → Preprocessing → Feature Engineering → EDA → Model Training (XGBoost) → Evaluation → Results
```

## Expanded Data Preprocessing Steps
- Remove land properties and irrelevant records.
- Encode categorical variables using one-hot or label encoding.
- Impute missing values:
  - Median/mean for numerical features
  - Mode for categorical features
- Predict missing `year_built` using RandomForestRegressor trained on available features.
- Remove duplicate and anomalous records.
- Feature selection based on correlation, variance, and domain knowledge.

## Expanded Model Training & Evaluation
- Data split: 80% training, 20% testing (stratified if needed).
- XGBoost regressor:
  - Hyperparameter tuning via grid search or cross-validation
  - Early stopping to prevent overfitting
- Evaluation metrics:
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - R² (Coefficient of Determination)
- Feature importance visualization to interpret model decisions.
- Save model and predictions for reproducibility.

## Notebooks & File Roles
- `house_price_predict_21E045.ipynb`: Main workflow, including preprocessing, EDA, model training, and evaluation.
- `Feature_Selection.ipynb`: In-depth feature selection and analysis.
- `requirements.txt`: Lists all required Python packages.
- `raw/`: Contains original dataset.
- `Processed/`: Stores processed datasets and results.

## Data Preprocessing Steps
- Remove land properties
- Encode categorical variables
- Handle missing values (median/mean imputation)
- Predict missing `year_built` using RandomForestRegressor
- Remove duplicates

## Model Training & Evaluation
- Train XGBoost regressor on processed data
- Evaluate using metrics such as RMSE, MAE, and R²
- Visualize feature importance and prediction results

## Results & Visualizations
- Distribution plots for features
- Correlation and covariance heatmaps
- Outlier detection and boxplots
- Model performance metrics

## References
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Seaborn Documentation](https://seaborn.pydata.org/)

---
For any questions or suggestions, please contact the project author.
