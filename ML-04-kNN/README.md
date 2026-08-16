# LEB 1: KNN on a Dataset of Your Choice

**Dataset:** `dogs_dataset.csv` — 3000 dogs, columns: `Breed`, `Age (Years)`, `Weight (kg)`, `Color`, `Gender`.

**Task:** Predict `Gender` (Female/Male) from `Breed`, `Age`, `Weight`, `Color` using K-Nearest Neighbors.

## Pipeline
1. **Load & explore** — 3000 rows, no missing values, 53 breeds, 16 colors, target is balanced (1520 Female / 1480 Male).
2. **Preprocess** — categorical features (`Breed`, `Color`) one-hot encoded; numeric features (`Age`, `Weight`) standardized with `StandardScaler`. All fitted only on the training split to avoid leakage.
3. **Split** — 80% train / 20% test, stratified on `Gender`, `random_state=42`.
4. **Train** — `KNeighborsClassifier` at k = 3, 5, 7.
5. **Evaluate** — accuracy on the held-out test set.

Run it:
```bash
python lab1_knn.py
```

## Results

| k | Accuracy |
|---|----------|
| 3 | 0.5100 |
| 5 | **0.5150** |
| 7 | 0.5017 |

**Best k = 5**, accuracy = 0.515

![Accuracy vs k](outputs/accuracy_vs_k.png)
![Confusion Matrix](outputs/confusion_matrix_best_k.png)

## Discussion

All three k values land close to 51%, barely above the 50% baseline of random guessing on a balanced two-class target. Accuracy is not monotonic in k here — it rises slightly from k=3 to k=5, then falls at k=7 — but the differences (≤1.3 points) are within noise for a 600-row test set, so no k value meaningfully outperforms another.

The underlying reason is the dataset itself: a dog's breed, age, weight, and coat color have no real biological or statistical relationship to its sex. Breed and color are attributes of the dog population, not sex-linked traits, and age/weight vary by breed rather than by gender. KNN can only exploit structure that exists in the feature space, and here that structure doesn't correlate with the target — so results near chance-level are the expected, correct outcome rather than a modeling failure. This is a useful illustration that KNN's performance is bounded by how informative the input features are, not just by tuning k.

## Files
- `dogs_dataset.csv` — dataset
- `lab1_knn.py` — full pipeline (load → preprocess → train k=3,5,7 → evaluate → plots)
- `outputs/accuracy_vs_k.png` — accuracy across k values
- `outputs/confusion_matrix_best_k.png` — confusion matrix for the best k
- `outputs/lab1_knn_results.json` — numeric results
