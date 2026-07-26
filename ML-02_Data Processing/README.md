# 🍔 Food Vitamins, Minerals, Macronutrient

โปรเจกต์นี้ทำการวิเคราะห์ข้อมูลเชิงสำรวจ (EDA) และกระบวนการเตรียมข้อมูล (Preprocessing) แบบครบวงจรบนชุดข้อมูล **🍔 Food Vitamins, Minerals, Macronutrient** โดยครอบคลุมการสำรวจข้อมูล การแสดงผลด้วยกราฟ การทำความสะอาดข้อมูล และการสร้างคุณลักษณะ (Feature Engineering) แบ่งออกเป็น 4 แล็บ/ส่วนตามที่โจทย์กำหนด

## Dataset

- **แหล่งที่มา:** [🍔 Food Vitamins, Minerals, Macronutrient](https://www.kaggle.com/datasets/mexwell/food-vitamins-minerals-macronutrient/code)
- **ไฟล์:** `food.csv`
- **ตัวแปรเป้าหมาย (Target variable):** `Category` (ตัวแปรเชิงหมวดหมู่ — เช่น Milk, Infant formula, Vegetables) ขึ้นอยู่กับวัตถุประสงค์ของการวิเคราะห์

## โครงสร้างไฟล์

```
CH02/
├── lab.py
├── food.csv
├── food-nutrients-analysis.ipynb
├── which-foods-should-you-choose.ipynb
└── README.md
```

## สิ่งที่ต้องติดตั้ง

ติดตั้งไลบรารี Python ที่จำเป็นก่อนรัน:

```bash
py -3.11 -m pip install pandas numpy matplotlib seaborn scikit-learn
```

## วิธีรัน

เปิดไฟล์ `lab.py` ด้วย VS Code ที่ติดตั้ง Python/Jupyter extension แล้ว ไฟล์นี้แบ่งเป็นเซลล์ด้วยเครื่องหมาย `# %%` ให้รันทีละเซลล์ตามลำดับ (บนลงล่าง) โดยใช้ปุ่ม **Run Cell** หรือ Interactive Window เนื่องจากเซลล์ท้าย ๆ (Part 3 และ Part 4) ขึ้นอยู่กับการแปลงข้อมูลที่ทำไว้ในเซลล์ก่อนหน้า

```bash
py -3.11 lab.py
```

## Lab 1: การสำรวจชุดข้อมูล (Dataset Exploration)

- โหลดชุดข้อมูลด้วย `pandas.read_csv`
- แสดงขนาดของชุดข้อมูล (จำนวนแถว, คอลัมน์)
- แสดงชนิดข้อมูล (data type) ของแต่ละคอลัมน์
- แสดงสถิติเชิงสรุป (`describe`)
- แสดงจำนวนค่าที่ขาดหาย (missing values) ในแต่ละคอลัมน์
- แสดงจำนวนแถวที่ซ้ำกัน (duplicate records)
- แสดงการกระจายตัวของคลาส (class distribution) ของตัวแปรเชิงหมวดหมู่หลัก

## Lab 2: การแสดงผลข้อมูลด้วยกราฟ (Data Visualization)

- **Histogram** ของตัวแปรเชิงตัวเลข (เช่น Calories, Price, ค่าทางโภชนาการต่าง ๆ) เพื่อดูการกระจายตัวของข้อมูล
- **Correlation heatmap** ของตัวแปรเชิงตัวเลข เพื่อดูความสัมพันธ์ระหว่างค่าทางโภชนาการกับราคา

## Part 3: การทำความสะอาดข้อมูล (Data Cleaning)

| ขั้นตอน | รายละเอียด |
|---|---|
| แก้ไขข้อมูลที่ไม่ถูกต้อง | ตรวจพบค่าว่าง (`" "`) หรืออักขระที่ไม่ถูกต้องในคอลัมน์ตัวเลข (เช่น Price, Calories) และแทนที่ด้วย `NaN` |
| การแปลงชนิดข้อมูล | แปลงคอลัมน์ที่เกี่ยวข้องจากชนิด `object` (string) เป็น `float` หรือ `int` เพื่อให้สามารถคำนวณทางคณิตศาสตร์ได้ |
| เปรียบเทียบ Mean กับ Median | คำนวณค่าเฉลี่ย (mean) และค่ามัธยฐาน (median) ของคอลัมน์ที่มีการกระจายตัวเบ้ (skewed) แล้วแสดงร่วมกันบน histogram เพื่อตัดสินใจเลือกวิธี imputation ที่เหมาะสม |
| การจัดการค่าที่ขาดหาย | เติมค่าที่ขาดหายด้วย **median** สำหรับตัวแปรที่มีการกระจายตัวเบ้มาก หรือ **mean** สำหรับตัวแปรที่มีการกระจายตัวใกล้เคียงปกติ เพื่อให้ได้ข้อมูลที่มีคุณภาพและน่าเชื่อถือ |
| การลบข้อมูลซ้ำ | ตรวจสอบและลบแถวที่ซ้ำกันด้วย `drop_duplicates()` |

## Part 4: การสร้างคุณลักษณะ (Feature Engineering)

- **Label Encoding** — ใช้กับคอลัมน์เชิงหมวดหมู่แบบสองค่า (binary) เช่น IsVegetarian, Availability โดยแปลงค่าสองหมวดหมู่ให้เป็น 0/1
- **One-Hot Encoding** — ใช้กับคอลัมน์ที่มีหลายหมวดหมู่ (multi-category) เช่น Category, CuisineType โดยใช้ `pd.get_dummies()` พร้อม `drop_first=True` เพื่อหลีกเลี่ยงปัญหา dummy variable trap

## หมายเหตุ

- ใช้ Label Encoding เฉพาะกับตัวแปรแบบ binary เท่านั้น เพื่อหลีกเลี่ยงการสร้างความสัมพันธ์เชิงลำดับ (ordinal) ที่ไม่มีอยู่จริงในข้อมูล
- ใช้ One-Hot Encoding กับตัวแปรที่มีมากกว่าสองหมวดหมู่ เพื่อไม่ให้โมเดลตีความว่าหมวดหมู่เหล่านั้นมีลำดับ
- เลือกใช้ Median imputation แทน Mean imputation สำหรับตัวแปรตัวเลขที่มีการกระจายตัวเบ้มาก (เช่น Calories หรือ Price) เนื่องจากให้ค่ากลางที่แม่นยำและทนทานต่อค่าผิดปกติ (robust) มากกว่า

## ผู้จัดทำ

บุณยวีร์ บุญวงศ์ / 116710400064-7 

จัดทำขึ้นเป็นส่วนหนึ่งของแบบฝึกหัดวิชา Machine Learning / Data Preprocessing (Lab 1–2, Part 3–4)
