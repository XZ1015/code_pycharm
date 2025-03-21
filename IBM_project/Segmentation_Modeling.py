import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 1️⃣ 读取数据
file_path = "car_sales.csv"
df = pd.read_csv(file_path)

# 2️⃣ 预处理数据
df.columns = df.columns.str.strip()  # 清理列名
df.dropna(inplace=True)  # 删除缺失值
df = pd.get_dummies(df, drop_first=True)  # 处理类别变量（如车型、地区等）

# 确保所有列都是数值型
df = df.apply(pd.to_numeric, errors='coerce')

# 3️⃣ 标准化数据
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# 4️⃣ 确定最佳 K 值（聚类数）
inertia = []
silhouette_scores = []
K_range = range(2, 10)  # 选择 2~10 个聚类

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(df_scaled)
    inertia.append(kmeans.inertia_)  # 计算误差平方和（SSE）
    silhouette_scores.append(silhouette_score(df_scaled, clusters))

# Elbow Method
plt.figure(figsize=(6, 4))
plt.plot(K_range, inertia, marker='o', linestyle='-')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia (SSE)")
plt.title("Elbow Method for Optimal K")
plt.show()

# Silhouette Score
plt.figure(figsize=(6, 4))
plt.plot(K_range, silhouette_scores, marker='s', linestyle='--', color="red")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Optimal K")
plt.show()

# choose the best K and train the model
best_k = K_range[np.argmax(silhouette_scores)]
kmeans = KMeans(n_clusters=best_k, random_state=42)
df["Cluster"] = kmeans.fit_predict(df_scaled)

cluster_sizes = df["Cluster"].value_counts().sort_index()

plt.figure(figsize=(6, 4))
sns.barplot(x=cluster_sizes.index, y=cluster_sizes.values, hue=cluster_sizes.index, palette="Blues_r", legend=False)
plt.xlabel("Cluster ID")
plt.ylabel("Number of Customers")
plt.title("Number of Customers in Each Cluster")
plt.show()


feature_importance = np.abs(kmeans.cluster_centers_).mean(axis=0)
feature_importance_df = pd.DataFrame({
    "Feature": df.drop(columns=["Cluster"]).columns,
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)

# curve the most important 10
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", hue="Feature", data=feature_importance_df.head(10), palette="Reds_r", legend=False)
plt.title("Top 10 Feature Importances in Clustering")
plt.show()

df.to_csv("car_sales_clustered.csv", index=False)

print(f"\nBest K: {best_k}")
print("Clustered data saved to car_sales_clustered.csv")
