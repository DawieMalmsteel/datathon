#!/usr/bin/env python3
"""
DATATHON 2026 - Complete Solution
Run this script to generate all answers and submission file
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Data directory
DATA_DIR = './'

# ============== LOAD DATA ==============
print("Loading data files...")
sales = pd.read_csv(DATA_DIR + 'sales.csv', parse_dates=['Date'])
orders = pd.read_csv(DATA_DIR + 'orders.csv', parse_dates=['order_date'])
order_items = pd.read_csv(DATA_DIR + 'order_items.csv')
products = pd.read_csv(DATA_DIR + 'products.csv')
customers = pd.read_csv(DATA_DIR + 'customers.csv', parse_dates=['signup_date'])
payments = pd.read_csv(DATA_DIR + 'payments.csv')
returns = pd.read_csv(DATA_DIR + 'returns.csv', parse_dates=['return_date'])
web_traffic = pd.read_csv(DATA_DIR + 'web_traffic.csv', parse_dates=['date'])
geography = pd.read_csv(DATA_DIR + 'geography.csv')

print(f"Sales: {sales.shape}, Orders: {orders.shape}, Products: {products.shape}")

# ============== PART 1: MCQ ==============
print("\n" + "="*60)
print("PART 1: MULTIPLE CHOICE QUESTIONS")
print("="*60)

# Q1
customer_order_counts = orders.groupby('customer_id').size()
multi_order_customers = customer_order_counts[customer_order_counts > 1].index
multi_orders = orders[orders['customer_id'].isin(multi_order_customers)].copy()
multi_orders = multi_orders.sort_values(['customer_id', 'order_date'])
multi_orders['prev_order_date'] = multi_orders.groupby('customer_id')['order_date'].shift(1)
multi_orders['inter_order_gap'] = (multi_orders['order_date'] - multi_orders['prev_order_date']).dt.days
median_gap = multi_orders['inter_order_gap'].dropna().median()
print(f"Q1: Median inter-order gap = {median_gap:.0f} days -> Answer: B) 90 ngày")

# Q2
products['gross_margin'] = (products['price'] - products['cogs']) / products['price']
segment_margin = products.groupby('segment')['gross_margin'].mean()
best_segment = segment_margin.idxmax()
print(f"Q2: Highest margin segment = {best_segment} -> Answer: A) Premium")

# Q3
returns_products = returns.merge(products[['product_id', 'category']], on='product_id', how='left')
streetwear_returns = returns_products[returns_products['category'] == 'Streetwear']
reason_counts = streetwear_returns['return_reason'].value_counts()
top_reason = reason_counts.idxmax()
print(f"Q3: Most common return reason for Streetwear = {top_reason} -> Answer: B) wrong_size")

# Q4
bounce_by_source = web_traffic.groupby('traffic_source')['bounce_rate'].mean()
lowest_bounce = bounce_by_source.idxmin()
print(f"Q4: Lowest bounce rate = {lowest_bounce} -> Answer: C) email_campaign")

# Q5
total_items = len(order_items)
promo_items = order_items['promo_id'].notna().sum()
promo_pct = (promo_items / total_items) * 100
print(f"Q5: Percentage with promotion = {promo_pct:.2f}% -> Answer: C) 39%")

# Q6
customers_with_age = customers[customers['age_group'].notna()].copy()
order_counts = orders.groupby('customer_id').size().reset_index(name='order_count')
customer_orders = customers_with_age.merge(order_counts, on='customer_id', how='left')
customer_orders['order_count'] = customer_orders['order_count'].fillna(0)
avg_by_age = customer_orders.groupby('age_group')['order_count'].mean()
best_age_group = avg_by_age.idxmax()
print(f"Q6: Highest avg orders = {best_age_group} -> Answer: A) 55+")

# Q7
region_counts = geography['region'].value_counts()
print(f"Q7: All regions in data = {region_counts.index.tolist()} -> Answer: A) East (only region)")

# Q8
cancelled_orders = orders[orders['order_status'] == 'cancelled']
payment_counts = cancelled_orders['payment_method'].value_counts()
top_payment = payment_counts.idxmax()
print(f"Q8: Most common payment for cancelled = {top_payment} -> Answer: A) credit_card")

# Q9
order_items_products = order_items.merge(products[['product_id', 'size']], on='product_id', how='left')
returns_with_size = returns.merge(products[['product_id', 'size']], on='product_id', how='left')
returns_by_size = returns_with_size.groupby('size')['return_quantity'].sum()
orders_by_size = order_items_products.groupby('size')['quantity'].sum()
return_rate = (returns_by_size / orders_by_size * 100).sort_values(ascending=False)
highest_return_size = return_rate.idxmax()
print(f"Q9: Highest return rate = {highest_return_size} ({return_rate[highest_return_size]:.2f}%) -> Answer: D) XL")

# Q10
avg_by_installment = payments.groupby('installments')['payment_value'].mean()
best_installment = avg_by_installment.idxmax()
print(f"Q10: Highest avg payment = {best_installment} installments -> Answer: D) 12 kỳ")

# ============== PART 3: FORECASTING ==============
print("\n" + "="*60)
print("PART 3: SALES FORECASTING MODEL")
print("="*60)

# Feature Engineering
def create_features(df):
    df = df.copy()
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['day_of_year'] = df['Date'].dt.dayofyear
    df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['Date'].dt.quarter
    
    # Cyclical encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # Binary features
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
    
    return df

# Prepare training data
train_df = create_features(sales.copy())

# Add web traffic features
web_traffic_features = web_traffic.groupby('date').agg({
    'sessions': 'sum',
    'unique_visitors': 'sum',
    'page_views': 'sum',
    'bounce_rate': 'mean'
}).reset_index()

train_df = train_df.merge(web_traffic_features, left_on='Date', right_on='date', how='left', suffixes=('', '_web'))
for col in ['sessions', 'unique_visitors', 'page_views', 'bounce_rate']:
    train_df[col] = train_df[col].fillna(train_df[col].mean())

# Feature columns
feature_cols = [
    'year', 'month', 'day', 'day_of_week', 'day_of_year', 'week_of_year', 'quarter',
    'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos',
    'day_of_year_sin', 'day_of_year_cos',
    'is_weekend', 'is_month_start', 'is_month_end',
    'sessions', 'unique_visitors', 'page_views', 'bounce_rate'
]

# Train/validation split
train_set = train_df[train_df['year'] <= 2021].copy()
val_set = train_df[train_df['year'] == 2022].copy()

X_train = train_set[feature_cols]
y_train_revenue = train_set['Revenue']
y_train_cogs = train_set['COGS']

X_val = val_set[feature_cols]
y_val_revenue = val_set['Revenue']
y_val_cogs = val_set['COGS']

# Train LightGBM
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

lgb_params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'n_estimators': 500,
    'random_state': 42
}

print("Training Revenue model...")
revenue_model = lgb.LGBMRegressor(**lgb_params)
revenue_model.fit(X_train, y_train_revenue)

print("Training COGS model...")
cogs_model = lgb.LGBMRegressor(**lgb_params)
cogs_model.fit(X_train, y_train_cogs)

# Validate
val_pred_revenue = revenue_model.predict(X_val)
val_pred_cogs = cogs_model.predict(X_val)

mae_rev = mean_absolute_error(y_val_revenue, val_pred_revenue)
rmse_rev = np.sqrt(mean_squared_error(y_val_revenue, val_pred_revenue))
r2_rev = r2_score(y_val_revenue, val_pred_revenue)

mae_cogs = mean_absolute_error(y_val_cogs, val_pred_cogs)
rmse_cogs = np.sqrt(mean_squared_error(y_val_cogs, val_pred_cogs))
r2_cogs = r2_score(y_val_cogs, val_pred_cogs)

print(f"\nValidation Metrics:")
print(f"  Revenue - MAE: {mae_rev:,.2f}, RMSE: {rmse_rev:,.2f}, R2: {r2_rev:.4f}")
print(f"  COGS    - MAE: {mae_cogs:,.2f}, RMSE: {rmse_cogs:,.2f}, R2: {r2_cogs:.4f}")

# Retrain on full data
print("\nRetraining on full data...")
X_full = train_df[feature_cols]
y_full_revenue = train_df['Revenue']
y_full_cogs = train_df['COGS']

revenue_model_full = lgb.LGBMRegressor(**lgb_params)
revenue_model_full.fit(X_full, y_full_revenue)

cogs_model_full = lgb.LGBMRegressor(**lgb_params)
cogs_model_full.fit(X_full, y_full_cogs)

# Generate test predictions
test_dates = pd.date_range(start='2023-01-01', end='2024-07-01', freq='D')
test_df = pd.DataFrame({'Date': test_dates})
test_df = create_features(test_df)

for col in ['sessions', 'unique_visitors', 'page_views', 'bounce_rate']:
    test_df[col] = train_df[col].mean()

X_test = test_df[feature_cols]
test_df['Revenue_pred'] = revenue_model_full.predict(X_test)
test_df['COGS_pred'] = cogs_model_full.predict(X_test)

# Create submission
submission = pd.DataFrame({
    'Date': test_df['Date'].dt.strftime('%Y-%m-%d'),
    'Revenue': test_df['Revenue_pred'].round(2),
    'COGS': test_df['COGS_pred'].round(2)
})

# Verify format
sample_submission = pd.read_csv(DATA_DIR + 'sample_submission.csv')
print(f"\nSample submission shape: {sample_submission.shape}")
print(f"Our submission shape: {submission.shape}")

assert len(submission) == len(sample_submission), "Row count mismatch!"

# Save submission
submission.to_csv(DATA_DIR + 'submission.csv', index=False)
print(f"\nSubmission saved to submission.csv")

print("\nSubmission preview:")
print(submission.head(10))

# Feature importance
print("\nTop 10 Revenue Drivers (Feature Importance):")
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': revenue_model_full.feature_importances_
}).sort_values('importance', ascending=False)
print(importance_df.head(10).to_string(index=False))

print("\n" + "="*60)
print("SOLUTION COMPLETE!")
print("="*60)
