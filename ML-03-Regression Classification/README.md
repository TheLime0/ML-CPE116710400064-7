# ใบงานที่ 3: Regression & Classification
วิชา Machine Learning (04-624-201) — ภาควิชาวิศวกรรมคอมพิวเตอร์

## ภาพรวม

โปรเจกต์นี้ทำตามใบงานที่ 3 ครบทั้ง 3 แล็บ:

- **LAB 1: Regression** — Simple & Multiple Linear Regression ทำนายอายุสุนัข (Age Prediction)
- **LAB 2: Classification** — Logistic Regression จำแนกเพศสุนัข (Gender Prediction) พร้อม Decision Boundary และ Confusion Matrix
- **LAB 3: Model Comparison** — เปรียบเทียบ Simple vs Multiple Regression, Training vs Testing, Regression vs Classification

นอกจากนี้ยังมีการสาธิต **PCA (Principal Component Analysis)** สำหรับลดจำนวนคุณลักษณะ (Feature Reduction) จากภาพถ่ายสุนัขโดยตรง

## Dataset

โปรเจกต์นี้ใช้ข้อมูล 2 ชุดที่ผู้ใช้ให้มา:

- **แหล่งที่มา:** [🐶 Dogs Dataset - 3000 Records 🐾](https://www.kaggle.com/datasets/waqi786/dogs-dataset-3000-records)

1. **`dogs_dataset.csv`** (ข้อมูลตาราง, 3,000 แถว) — คอลัมน์ `Breed, Age (Years), Weight (kg), Color, Gender`
   ใช้เป็นข้อมูลหลักสำหรับ Regression (Age) และ Classification (Gender) เพราะมี label จริงติดมากับข้อมูล

2. **ชุดภาพถ่ายสุนัข** (18,247 ภาพ ใน 1,174 โฟลเดอร์ ขนาด 160x160) — **ไม่มี label อายุ/เพศ** กำกับมาด้วย
   จึงใช้สาธิตเทคนิค PCA ลดมิติภาพเท่านั้น (ไม่ได้นำไปทำนายอายุ/เพศ เพราะไม่มี ground-truth จริง หากปั้น label เองจะทำให้ผลลัพธ์ไม่น่าเชื่อถือ)

> **หมายเหตุ:** ชุดภาพต้นฉบับที่อัปโหลดมาก่อนหน้านี้ (archive__2_.zip) เป็น dataset สำหรับงาน Face Recognition/Verification (จัดกลุ่มภาพตามรหัสบุคคล/ชื่อ) ไม่มี label อายุ-เพศ จึงใช้ประกอบเฉพาะส่วน PCA feature-reduction เท่านั้น

## โครงสร้างไฟล์

```
lab_project/
├── README.md                     # ไฟล์นี้
├── build_report.js               # สคริปต์สร้างรายงาน Word (Lab3_Report.docx)
├── data/
│   └── dogs_dataset.csv          # ข้อมูลตารางหลัก
├── src/
│   ├── image_pca.py              # LAB: PCA feature reduction บนภาพสุนัข ("Eigen-dogs")
│   ├── lab1_regression.py        # LAB 1: Simple & Multiple Linear Regression (Age)
│   ├── lab2_classification.py    # LAB 2: Logistic Regression (Gender) + Decision Boundary + Confusion Matrix
│   └── lab3_comparison.py        # LAB 3: สรุปเปรียบเทียบผลลัพธ์ทั้งหมด
└── outputs/                      # ผลลัพธ์ที่สร้างจากการรันโค้ด (กราฟ .png, ผลลัพธ์ .json, รายงาน .docx)
```

## วิธีรัน

รันตามลำดับ (แต่ละสคริปต์ต้องรันก่อนตัวถัดไป เพราะ `lab3_comparison.py` และ `build_report.js` อ่านผลลัพธ์ JSON ที่สคริปต์ก่อนหน้าสร้างไว้):

```bash
# ติดตั้ง dependency ที่จำเป็น (ถ้ายังไม่มี)
pip install pandas numpy scikit-learn matplotlib pillow --break-system-packages

cd src
python3 image_pca.py            # สร้าง PCA feature-reduction demo บนภาพสุนัข
python3 lab1_regression.py      # LAB 1: Regression
python3 lab2_classification.py  # LAB 2: Classification
python3 lab3_comparison.py      # LAB 3: Model Comparison

# สร้างรายงาน Word (ต้องมี Node.js + npm package "docx")
cd ..
node build_report.js
```

ผลลัพธ์ทั้งหมด (กราฟ `.png`, สรุปผล `.json`, รายงาน `Lab3_Report.docx`) จะถูกเขียนไปที่โฟลเดอร์ `outputs/`

## สรุปผลลัพธ์สำคัญ

| งาน | โมเดล | ตัวชี้วัดหลัก | ผลลัพธ์ (Test set) |
|---|---|---|---|
| PCA (ภาพ) | — | จำนวน components ที่อธิบาย 95% variance | ~167 จาก 4,096 features (อัตราบีบอัด ~24.5 เท่า) |
| Regression (Age) | Simple Linear Regression | R² | ≈ 0.00 |
| Regression (Age) | Multiple Linear Regression | R² | ≈ 0.00 |
| Classification (Gender) | Logistic Regression | Accuracy / ROC AUC | ≈ 0.53 / ≈ 0.51 |

**ข้อสังเกต:** ค่า R² และ Accuracy ใกล้เคียงศูนย์/การเดาสุ่ม เนื่องจาก `Breed, Weight, Color` ใน `dogs_dataset.csv` ไม่มีความสัมพันธ์เชิงสถิติที่แท้จริงกับ `Age` หรือ `Gender` — เป็นผลลัพธ์ที่ซื่อตรงต่อข้อมูลจริง ไม่ได้ปรับแต่งให้ดูดีเกินจริง รายละเอียดการวิเคราะห์และข้อเสนอแนะอยู่ในรายงาน `Lab3_Report.docx`

## ผู้จัดทำ

บุณยวีร์ บุญวงศ์ / 1167104000064-7 
