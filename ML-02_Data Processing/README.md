# 🍔 Food Vitamins, Minerals, Macronutrient
# Food Dataset Exploration – Data Exploration, Visualization, Cleaning & Feature Engineering
 
This project performs a full exploratory data analysis (EDA) and preprocessing pipeline on a **Food** dataset. It covers dataset exploration, visualization, data cleaning, and feature engineering, structured into four labs/parts as required by the assignment.
 
## Dataset
 
- **File:** `food.csv`
- **Target variable:** Dependent on the specific machine learning task (e.g., Categorical classification or Nutrient value prediction)
 
## Project Structure
 
CH02/├── lab.py
     ├── food.csv
     ├── food-nutrients-analysis.ipynb
     ├── which-foods-should-you-choose.ipynb
     └── README.md 
## Requirements
 
Install the required Python packages before running:
 
```bash
py -3.11 -m pip install pandas numpy matplotlib seaborn scikit-learn
How to RunOpen lab.py in VS Code with the Python/Jupyter extensions installed. The file is divided into cells using # %% markers. Run each cell sequentially (top to bottom) using the Run Cell button or Interactive Window, since later cells (Part 3 and Part 4) depend on transformations made in earlier cells.Bashpy -3.11 lab.py
Lab 1: Dataset ExplorationLoad the dataset with pandas.read_csvDisplay dataset shape (rows, columns)Display data types of each columnDisplay summary statistics (describe)Display count of missing values per columnDisplay count of duplicate recordsDisplay class distribution of the main categorical variablesLab 2: Data VisualizationHistogram of all numeric features to inspect distributions (e.g., calories, nutrients, or pricing)Correlation heatmap of numeric features to inspect relationships between variablesPart 3: Data CleaningStepDescriptionIncorrect Data CorrectionIdentified columns containing blank strings (" ") or invalid characters instead of proper missing values, replacing them with NaNData Type ConversionCategorical columns or corrupted numeric columns are converted to their correct types (float, int, or category)Mean vs Median ComparisonMean and median of highly skewed features are computed and visualized together on a histogram to decide the best imputation strategyMissing Value HandlingMissing values in critical numeric columns are filled using the median, since it is more robust to outliers/skewed distributions than the meanDuplicate RemovalDuplicate rows are identified and removed using drop_duplicates() to ensure data integrityPart 4: Feature EngineeringLabel Encoding — applied to binary categorical columns, converting two-category values (such as Yes/No or High/Low classifications) into 0/1One-Hot Encoding — applied to multi-category columns (such as food groups, categories, or types) using pd.get_dummies() with drop_first=True to avoid the dummy variable trapNotesLabel Encoding is used only for binary features to avoid introducing artificial ordinal relationships.One-Hot Encoding is used for features with more than two categories so the machine learning model does not assume a false order.Median imputation was chosen over mean imputation because the target numeric columns showed skewed distributions, making the median a more robust central estimate against outliers.AuthorPrepared as part of a Machine Learning / Data Preprocessing lab assignment (Lab 1–2, Part 3–4).
