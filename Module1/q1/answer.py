import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------
# Load Dataset
# ---------------------------------------
df = pd.read_csv("sales.csv")

print("=" * 50)
print("First 5 Records")
print("=" * 50)
print(df.head())

print("\nDataset Information")
print(df.info())

# ---------------------------------------
# Check Missing Values
# ---------------------------------------
print("\nMissing Values")
print(df.isnull().sum())

# ---------------------------------------
# Clean Missing Values
# ---------------------------------------

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Fill missing values
numeric_columns = ["Units Sold", "Unit Price", "Unit Cost",
                   "Total Revenue", "Total Cost", "Total Profit"]

for col in numeric_columns:
    df[col] = df[col].fillna(df[col].median())

categorical_columns = ["Region", "Country", "Item Type",
                       "Sales Channel", "Order Priority"]

for col in categorical_columns:
    df[col] = df[col].fillna("Unknown")

# Convert dates
df["Order Date"] = pd.to_datetime(df["Order Date"])

# Save cleaned dataset
df.to_csv("cleaned_sales.csv", index=False)

print("\nCleaned dataset saved successfully.")

# ---------------------------------------
# Summary Statistics
# ---------------------------------------

summary = df.describe(include="all")

print("\nSummary Statistics")
print(summary)

summary.to_csv("summary_statistics.csv")

# ---------------------------------------
# Total Revenue by Product Category
# (Item Type acts as Product Category)
# ---------------------------------------

revenue_by_category = (
    df.groupby("Item Type")["Total Revenue"]
      .sum()
      .sort_values(ascending=False)
)

print("\nRevenue by Product Category")
print(revenue_by_category)

# ---------------------------------------
# Top 10 Customers
# ---------------------------------------
# NOTE:
# This dataset does not contain customer information.
# Hence Top 10 Countries by Revenue is shown instead.

top10 = (
    df.groupby("Country")["Total Revenue"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
)

print("\nTop 10 Countries by Revenue")
print(top10)

# ---------------------------------------
# Monthly Sales
# ---------------------------------------

df["Month"] = df["Order Date"].dt.to_period("M").astype(str)

monthly_sales = (
    df.groupby("Month")["Total Revenue"]
      .sum()
)

# ---------------------------------------
# Plot 1 : Monthly Sales
# ---------------------------------------

plt.figure(figsize=(10,5))

monthly_sales.plot(marker="o")

plt.title("Monthly Sales")

plt.xlabel("Month")

plt.ylabel("Revenue")

plt.grid(True)

plt.tight_layout()

plt.savefig("monthly_sales.png")

plt.show()

# ---------------------------------------
# Plot 2 : Revenue by Product Category
# ---------------------------------------

plt.figure(figsize=(10,5))

revenue_by_category.plot(kind="bar", color="skyblue")

plt.title("Revenue by Product Category")

plt.xlabel("Item Type")

plt.ylabel("Revenue")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("revenue_by_category.png")

plt.show()

