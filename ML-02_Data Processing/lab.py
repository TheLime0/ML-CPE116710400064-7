import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(ML-02_Data Processing/food.csv)

# ==========================================
# LAB1: Dataset Exploration
# ==========================================
print("--- LAB1: Dataset Exploration ---")
# Load Dataset (ถ้าใช้ไฟล์จริงให้เปลี่ยนเป็น pd.read_csv('ชื่อไฟล์.csv'))
print("1. Dataset Shape:", df.shape)
print("\n2. Data Types:\n", df.dtypes)
print("\n3. Summary Statistics:\n", df.describe())
print("\n4. Missing Values:\n", df.isnull().sum())
print("\n5. Duplicate Records Count:", df.duplicated().sum())
print("\n6. Class Distribution:\n", df['Purchased'].value_counts())
print("-" * 40)

# ==========================================
# LAB2: Data Visualization
# ==========================================
# 1. Histogram
df.hist(bins=10, figsize=(10, 4))
plt.suptitle("Histogram of Numeric Features")
plt.show()

# 2. Correlation Heatmap (คำนวณเฉพาะคอลัมน์ที่เป็นตัวเลข)
plt.figure(figsize=(6, 4))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# ==========================================
# Part 3: Data Cleaning
# ==========================================
print("\n--- Part 3: Data Cleaning ---")

# เก็บค่า Mean/Median ก่อนจัดการ Missing Value เพื่อเปรียบเทียบ
age_mean_before = df['Age'].mean()
age_median_before = df['Age'].median()

# 1. Missing Value Handling (แทนที่ด้วย Median สำหรับ Age และ Mean สำหรับ Salary)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Salary'] = df['Salary'].fillna(df['Salary'].mean())

# 2. Duplicate Removal (ลบแถวที่ซ้ำซ้อน)
df = df.drop_duplicates().reset_index(drop=True)

# 3. Incorrect Data Correction & Data Type Conversion
# สมมติว่าเจอ Outlier คืออายุ 200 ซึ่งผิดปกติ เลยแก้เป็นค่ามัธยฐาน แล้วแปลงเป็น int
df.loc[df['Age'] > 100, 'Age'] = df['Age'].median()
df['Age'] = df['Age'].astype(int) 

# 4. Compare Mean & Median (เปรียบเทียบค่าก่อน-หลังล้างข้อมูลของคอลัมน์ Age)
print(f"Age Mean   -> Before: {age_mean_before:.2f} | After: {df['Age'].mean():.2f}")
print(f"Age Median -> Before: {age_median_before:.2f} | After: {df['Age'].median():.2f}")
print("-" * 40)

# ==========================================
# Part 4: Feature Engineering
# ==========================================
print("\n--- Part 4: Feature Engineering ---")

# 1. Label Encoding (เหมาะกับข้อมูล 2 กลุ่ม หรือข้อมูลที่มีลำดับขั้น เช่น Gender, Purchased)
le = LabelEncoder()
df['Gender_Encoded'] = le.fit_transform(df['Gender'])
df['Purchased_Encoded'] = le.fit_transform(df['Purchased'])

# 2. One-Hot Encoding (เหมาะกับข้อมูลกลุ่มที่ไม่มีลำดับขั้น เช่น Education)
df = pd.get_dummies(df, columns=['Education'], dtype=int)

print("\nFinal Preprocessed Dataset:")
print(df.to_string())
