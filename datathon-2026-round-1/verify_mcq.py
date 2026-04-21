#!/usr/bin/env python3
"""
Verify and fix MCQ answers
"""
import pandas as pd
import numpy as np

DATA_DIR = './'

# Load data
sales = pd.read_csv(DATA_DIR + 'sales.csv', parse_dates=['Date'])
orders = pd.read_csv(DATA_DIR + 'orders.csv', parse_dates=['order_date'])
order_items = pd.read_csv(DATA_DIR + 'order_items.csv')
products = pd.read_csv(DATA_DIR + 'products.csv')
customers = pd.read_csv(DATA_DIR + 'customers.csv', parse_dates=['signup_date'])
payments = pd.read_csv(DATA_DIR + 'payments.csv')
returns = pd.read_csv(DATA_DIR + 'returns.csv', parse_dates=['return_date'])
web_traffic = pd.read_csv(DATA_DIR + 'web_traffic.csv', parse_dates=['date'])
geography = pd.read_csv(DATA_DIR + 'geography.csv')
orders_items = order_items.merge(orders[['order_id', 'order_date', 'zip']], on='order_id')

print("="*60)
print("VERIFYING MCQ ANSWERS")
print("="*60)

# Q1: Check again - median inter-order gap
customer_order_counts = orders.groupby('customer_id').size()
multi_order_customers = customer_order_counts[customer_order_counts > 1].index
multi_orders = orders[orders['customer_id'].isin(multi_order_customers)].copy()
multi_orders = multi_orders.sort_values(['customer_id', 'order_date'])
multi_orders['prev_order_date'] = multi_orders.groupby('customer_id')['order_date'].shift(1)
multi_orders['inter_order_gap'] = (multi_orders['order_date'] - multi_orders['prev_order_date']).dt.days
median_gap = multi_orders['inter_order_gap'].dropna().median()
print(f"\nQ1: Median inter-order gap = {median_gap:.0f} days")
print(f"Options: A) 30, B) 90, C) 144, D) 365")
print(f"Answer: C) 144 ngày")

# Q2: Recalculate gross margin
products['gross_margin'] = (products['price'] - products['cogs']) / products['price']
segment_margin = products.groupby('segment')['gross_margin'].mean().sort_values(ascending=False)
print(f"\nQ2: Gross margin by segment:")
print(segment_margin)
print(f"Options: A) Premium, B) Performance, C) Activewear, D) Standard")
print(f"Answer: A) Premium (if Premium has highest margin)")

# Check if it's Premium
print(f"\nChecking segment 'Premium': {segment_margin.get('Premium', 'N/A')}")

# Q7: Revenue by region - need to join through orders
orders_with_zip = orders.merge(geography, on='zip', how='left')
orders_with_items = order_items.merge(orders_with_zip[['order_id', 'region', 'order_date']], on='order_id')
orders_with_items['order_value'] = orders_with_items['quantity'] * orders_with_items['unit_price']

# Join with sales to get dates
orders_date = orders[['order_id', 'order_date']].copy()
orders_date['year'] = orders_date['order_date'].dt.year
orders_with_items = orders_with_items.merge(orders_date, on='order_id')

# Filter for 2012-2022
orders_with_items = orders_with_items[orders_with_items['year'] <= 2022]
revenue_by_region = orders_with_items.groupby('region')['order_value'].sum()
print(f"\nQ7: Revenue by region:")
print(revenue_by_region.sort_values(ascending=False))
print(f"Options: A) West, B) Central, C) East, D) All equal")
print(f"Answer: A) West")

# Q9: Return rate by size - recalculate more carefully
order_items_products = order_items.merge(products[['product_id', 'size']], on='product_id', how='left')
total_by_size = order_items_products.groupby('size')['quantity'].sum()
print(f"\nQ9: Total ordered quantity by size:")
print(total_by_size)

returns_with_size = returns.merge(products[['product_id', 'size']], on='product_id', how='left')
returned_by_size = returns_with_size.groupby('size')['return_quantity'].sum()
print(f"\nQ9: Total returned quantity by size:")
print(returned_by_size)

return_rate = (returned_by_size / total_by_size * 100).sort_values(ascending=False)
print(f"\nQ9: Return rate by size (%):")
print(return_rate)
print(f"Options: A) S, B) M, C) L, D) XL")
print(f"Answer: A) S")

# Q10: Verify installment again
avg_by_installment = payments.groupby('installments')['payment_value'].mean().sort_values(ascending=False)
print(f"\nQ10: Average payment by installments:")
print(avg_by_installment)
print(f"Options: A) 1, B) 3, C) 6, D) 12")
print(f"Answer: C) 6 kỳ")

print("\n" + "="*60)
print("FINAL ANSWERS")
print("="*60)
print("""
Q1: C) 144 ngày
Q2: Need to verify - Standard vs Premium
Q3: B) wrong_size
Q4: C) email_campaign  
Q5: C) 39%
Q6: A) 55+
Q7: A) West
Q8: A) credit_card
Q9: A) S
Q10: C) 6 kỳ
""")
