import pandas as pd

# 读取数据
file_path = "customer_churn.csv"
df = pd.read_csv(file_path)

# 查看数据基本情况
print("\n==== 数据预览 ====\n")
print(df.head())

print("\n==== 数据类型 ====\n")
print(df.dtypes)

print("\n==== 是否有缺失值 ====\n")
print(df.isnull().sum())

print("\n==== CHURN_BINARY 分布 ====\n")
print(df["CHURN_BINARY"].value_counts(normalize=True))
