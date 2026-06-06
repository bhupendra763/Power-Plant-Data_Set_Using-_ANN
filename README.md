# ⚡ Power Plant Energy Output Prediction — ANN Regression

A deep learning project using an **Artificial Neural Network (ANN)** built with **PyTorch** to predict the net hourly electrical energy output (PE) of a Combined Cycle Power Plant (CCPP) based on ambient environmental conditions.

---

## 📌 Problem Statement

Predicting the power output of a power plant is critical for efficient energy management. This project frames it as a **regression task**: given four environmental sensor readings, predict the plant's electrical energy output in megawatts (MW).

---

## 📂 Dataset

**File:** `powerPlant.csv`  
**Source:** [UCI Machine Learning Repository — Combined Cycle Power Plant Dataset](https://archive.ics.uci.edu/ml/datasets/combined+cycle+power+plant)

| Feature | Description | Unit |
|--------|-------------|------|
| `AT` | Ambient Temperature | °C |
| `V` | Exhaust Vacuum | cm Hg |
| `AP` | Ambient Pressure | millibar |
| `RH` | Relative Humidity | % |
| `PE` | Net Hourly Electrical Energy Output *(Target)* | MW |

**Dataset Stats:**

| Property | Value |
|----------|-------|
| Rows | 9,568 |
| Columns | 5 |
| Missing Values | None |
| Duplicates | Checked |

---

## 🧰 Tech Stack

| Tool | Purpose |
|------|---------|
| `Python` | Core language |
| `Pandas` | Data loading & EDA |
| `NumPy` | Numerical operations |
| `Matplotlib` | Loss curve visualization |
| `scikit-learn` | Train/test split, StandardScaler, R² metric |
| `PyTorch` | ANN model, training loop, DataLoader |

---

## 🔬 Project Workflow

```
Data Loading → EDA → Preprocessing → Tensor Conversion → 
Model Building → Training → Evaluation → Prediction
```

### 1. 📊 Exploratory Data Analysis (EDA)
- Checked dataset shape, dtypes, null values, and duplicates
- Reviewed descriptive statistics across all features

### 2. 🎯 Feature & Target Selection
- **Features (X):** `AT`, `V`, `AP`, `RH`
- **Target (y):** `PE`

### 3. ✂️ Train-Test Split
- 80% training / 20% testing
- `random_state=42` for reproducibility

### 4. ⚖️ Feature Scaling
- Applied `StandardScaler` (fit on train, transform on both sets)

### 5. 🔄 Tensor Conversion & DataLoader
- Converted NumPy arrays to `torch.float32` tensors
- Used `TensorDataset` + `DataLoader` with `batch_size=32`

### 6. 🧠 ANN Architecture

```
Input (4 features)
    ↓
Linear(4 → 6) + ReLU
    ↓
Linear(6 → 6) + ReLU
    ↓
Linear(6 → 1)   ← Output: predicted PE
```

### 7. 🏋️ Training
- **Loss Function:** MSELoss
- **Optimizer:** Adam (default lr)
- **Epochs:** 100
- Best model saved via `torch.save()` based on lowest validation loss

### 8. 📈 Evaluation Metrics
- Training MSE Loss
- Testing MSE Loss
- **R² Score** on test set

---

## 📉 Loss Curve

Training and validation losses are plotted across 100 epochs to visualize convergence and check for overfitting.

---

## 📁 Project Structure

```
power-plant-ann/
│
├── powerPlant.csv          # Dataset
├── PowerPlant.ipynb        # Main Jupyter Notebook
├── best_model.pt           # Saved best model weights
└── README.md               # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib scikit-learn torch
```

### Run the Notebook

```bash
jupyter notebook PowerPlant.ipynb
```

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Training MSE loss | 20.504594802856445 |
| Testing MSE loss| 18.82399559020996|
| R² Score | 93.42149937507415|

> Results may vary slightly due to random weight initialization.

---

## 🙌 Acknowledgements
- Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/combined+cycle+power+plant)
- Built as a practice project for ANN regression with PyTorch
