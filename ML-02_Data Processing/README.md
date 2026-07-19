# 🍔 Food Vitamins, Minerals, Macronutrient

This project performs a full exploratory data analysis (EDA) and preprocessing pipeline on the **🍔 Food Vitamins, Minerals, Macronutrient** dataset. It covers dataset exploration, visualization, data cleaning, and feature engineering, structured into four labs/parts as required by the assignment.

## Dataset

- **Source:** Food Dataset (Local File)
- **File:** `food.csv`
- **Target variable:** `Category` / `Rating` (depending on specific model objectives)

## Project Structure

```
CH02/
├── lab.py
├── food.csv
├── food-nutrients-analysis.ipynb
├── which-foods-should-you-choose.ipynb
└── README.md
```

## Requirements

Install the required Python packages before running:

```bash
py -3.11 -m pip install pandas numpy matplotlib seaborn scikit-learn
```

## How to Run

Open `lab.py` in VS Code with the Python/Jupyter extensions installed. The file is divided into cells using `# %%` markers. Run each cell sequentially (top to bottom) using the **Run Cell** button or Interactive Window, since later cells (Part 3 and Part 4) depend on transformations made in earlier cells.

```bash
py -3.11 lab.py
```

## Lab 1: Dataset Exploration

- Load the dataset with `pandas.read_csv`
- Display dataset shape (rows, columns)
- Display data types of each column
- Display summary statistics (`describe`)
- Display count of missing values per column
- Display count of duplicate records
- Display class distribution of key categorical features

## Lab 2: Data Visualization

- **Histogram** of numeric features (e.g., Calories, Price, Nutrition values) to inspect distributions
- **Correlation heatmap** of numeric variables to inspect relationships between nutritional values and price

## Part 3: Data Cleaning

| Step | Description |
|---|---|
| Incorrect Data Correction | Identified blank strings (`" "`) or invalid characters in numeric columns (e.g., Price, Calories) and replaced them with `NaN` |
| Data Type Conversion | Converted relevant columns from `object` (string) to `float` or `int` for mathematical analysis |
| Mean vs Median Comparison | Mean and median of skewed columns are computed and visualized together on a histogram to decide the best imputation strategy |
| Missing Value Handling | Missing values are filled using the **median** for highly skewed features, or **mean** for normally distributed ones, ensuring robust data quality |
| Duplicate Removal | Duplicate rows are identified and removed using `drop_duplicates()` |

## Part 4: Feature Engineering

- **Label Encoding** — applied to binary categorical columns (e.g., IsVegetarian, Availability), converting two-category values into 0/1
- **One-Hot Encoding** — applied to multi-category columns (e.g., Category, CuisineType) using `pd.get_dummies()` with `drop_first=True` to avoid the dummy variable trap

## Notes

- Label Encoding is used only for binary features to avoid introducing artificial ordinal relationships.
- One-Hot Encoding is used for features with more than two categories so the model does not assume a false order.
- Median imputation is preferred over mean imputation for heavily skewed numeric attributes (such as Calories or Price) to provide a more robust central estimate.

## Author

Prepared as part of a Machine Learning / Data Preprocessing lab assignment (Lab 1–2, Part 3–4).
