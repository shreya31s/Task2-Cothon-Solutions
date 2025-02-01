import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

# Load the dataset
df = pd.read_csv("customer_data.csv")

# Display the first few rows and check column names
print(df.head())
print(df.columns)  # Check the actual column names

# Check for missing values
print("Missing values in each column:")
print(df.isnull().sum())

# Check data types
print("\nData types:")
print(df.dtypes)

# Fill missing values for numeric columns only
numeric_cols = df.select_dtypes(include=['number']).columns  # Select numeric columns

# Fill missing values in numeric columns with their mean
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# For categorical columns, fill with mode or drop missing values
# For the 'Satisfaction Level' column, fill with mode
if 'Satisfaction Level' in df.columns:
    df['Satisfaction Level'].fillna(df['Satisfaction Level'].mode()[0], inplace=True)


# Verify if there are any missing values left
print("\nMissing values after filling:")
print(df.isnull().sum())

# Save the cleaned data
df.to_csv("cleaned_customer_data.csv", index=False)
print("Cleaned data saved successfully!")

# Selecting Relevant Features
selected_features = ['Total Spend', 'Items Purchased', 'Days Since Last Purchase']
X = df[selected_features]

# Normalize the Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dimensionality Reduction (Optional)
pca = PCA(n_components=2)  # Reduce to 2D for visualization
X_pca = pca.fit_transform(X_scaled)

# Finding Optimal Clusters Using Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow Method Graph
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Assuming the optimal K is 4 based on the Elbow method
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Apply DBSCAN (alternative clustering method)
dbscan = DBSCAN(eps=0.5, min_samples=5)
df['Cluster_DBSCAN'] = dbscan.fit_predict(X_scaled)

# Visualization of Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Cluster'], palette='viridis')
plt.title('Customer Segments')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.show()

# Analyze Cluster Characteristics
# Calculate mean only for numeric columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns  # Select only numeric columns
cluster_analysis = df.groupby('Cluster')[numeric_cols].mean()  # Group by 'Cluster' and calculate the mean
print(cluster_analysis)

# Save the Results
df.to_csv("segmented_customers.csv", index=False)