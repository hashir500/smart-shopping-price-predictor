# Smart Shopping Price Predictor

A machine learning project that predicts discount percentages for Amazon India products — helping shoppers identify which product categories and price tiers offer the best deals.

---

## Problem Statement

Every online shopper has wondered — *"Should I buy this now, or wait for a sale?"*

Most people guess. This project uses machine learning to answer that question by analyzing patterns in real Amazon product data — finding which categories discount the most, how ratings relate to pricing, and predicting the discount a product is likely to receive.

---

## Dataset

- **Source:** Amazon India Sales Dataset (Kaggle)
- **Size:** 1,465 products across 9 categories
- **Key columns used:** `actual_price`, `discounted_price`, `discount_percentage`, `rating`, `rating_count`, `category`

---

## Project Structure

```
Smart-Shopping-Price-Predictor/
│
├── Smart_Shopping_Price_Predictor.ipynb   # Main notebook
├── amazon.csv                             # Dataset
└── README.md
```

---

## Workflow

### 1. Data Cleaning
- Removed `₹` symbols and commas from price columns
- Stripped `%` from discount percentages
- Handled missing and malformed values in `rating_count`
- Converted all relevant columns from object to numeric types

### 2. Feature Engineering
- `discount_amount` — difference between actual and discounted price
- `main_category` — extracted top-level category from the nested category string
- `category_encoded` — label-encoded category for use in Random Forest
- `product_name_len` — length of product name as a numeric signal

### 3. Exploratory Data Analysis
- **HomeImprovement** offers the highest average discounts (~60%)
- **Toys & Games** offers the least (~0%)
- Lower-rated products tend to receive higher discounts — likely clearance pricing
- Premium-priced products are discounted less consistently than budget ones

### 4. Model Training

| Model | MAE | Notes |
|---|---|---|
| Linear Regression | 15.6% | Baseline — no categorical features |
| Random Forest (unregularized) | Train: 4.5% / Test: 10.7% | Overfitting detected |
| Random Forest (regularized) ✓ | Train: 10.8% / Test: 11.8% | Overfitting resolved |

### 5. Regularization
The initial Random Forest overfit the training data (MAE gap of ~6%). Fixed by tuning:
- `max_depth=10` — limits how specific each tree's rules can be
- `min_samples_leaf=5` — each rule must apply to at least 5 products
- `max_features='sqrt'` — forces trees to be diverse

### 6. Prediction
The final model predicts discount percentages across four product price tiers (Budget → Luxury), showing how actual price influences the expected discount level.

---

## Key Findings

- Product category is the strongest predictor of discount percentage
- The gap between train and test MAE closed from ~6% to ~1% after regularization
- Budget-tier products receive deeper and more consistent discounts than premium ones
- Rating alone is a weak predictor — high-rated products are not necessarily discounted less

---

## Tech Stack

| Tool | Usage |
|---|---|
| Python | Core language |
| Pandas | Data loading and cleaning |
| Seaborn / Matplotlib | Visualizations |
| Scikit-learn | Model training and evaluation |

---

## How to Run

1. Clone the repository
```bash
git clone https://github.com/yourusername/smart-shopping-price-predictor.git
```

2. Install dependencies
```bash
pip install pandas scikit-learn seaborn matplotlib
```

3. Open the notebook
```bash
jupyter notebook Smart_Shopping_Price_Predictor.ipynb
```

4. Update the dataset path in cell 3 to point to your local `amazon.csv`

---

## CV Listing

**Consumer Behavior Analysis & Price Optimization**
- Built a regularized Random Forest regression model to predict discount percentages across 1,465 Amazon India products, achieving a test MAE of ~11.8%
- Engineered features from product category, rating, and pricing data to identify which product tiers and categories offer the deepest discounts — enabling smarter purchase timing decisions
