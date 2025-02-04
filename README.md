# Customer Segmentation using Clustering Algorithms

## 📌 Project Overview
This project focuses on **Customer Segmentation** using clustering algorithms such as **K-Means** and **DBSCAN**. The goal is to analyze customer data, identify patterns, and segment customers into different groups based on their purchasing behavior. 

## 📂 Dataset
- **Source**: Kaggle / Custom Dataset
- **Columns**:
  - `Customer ID`: Unique identifier for each customer
  - `Gender`: Male/Female
  - `Age`: Customer age
  - `City`: Location of customer
  - `Membership Type`: Type of subscription
  - `Total Spend`: Amount spent by the customer
  - `Items Purchased`: Number of items bought
  - `Average Rating`: Average rating given by the customer
  - `Discount Applied`: Boolean (True/False) indicating discount usage
  - `Days Since Last Purchase`: Recency of customer activity
  - `Satisfaction Level`: Customer feedback

## 🔧 Project Requirements
To run this project, install the required Python packages:
```bash
pip install -r requirements.txt
```

### Required Libraries
- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`
- `sklearn`

## 🚀 Steps to Run the Project
### 1️⃣ Load & Preprocess Data
- Load dataset using Pandas
- Handle missing values (fill with mode for categorical, mean for numerical)
- Convert categorical features into numerical using Label Encoding

### 2️⃣ Exploratory Data Analysis (EDA)
- Perform data visualization using **Seaborn** & **Matplotlib**
- Generate scatter plots, histograms, and correlation heatmaps

### 3️⃣ Feature Selection
- Select relevant features: `Total Spend`, `Items Purchased`, `Days Since Last Purchase`
- Apply **StandardScaler** for normalization

### 4️⃣ Apply Clustering Algorithms
#### K-Means Clustering
- Determine optimal `K` using **Elbow Method**
- Fit model and predict clusters
#### DBSCAN Clustering
- Apply DBSCAN for density-based clustering

### 5️⃣ Visualization of Clusters
- Scatter plots for visualizing segmented groups
- Heatmaps for analyzing cluster characteristics

## 🛠️ How to Run the Code
Run the main Python script:
```bash
python customer_segmentation.py
```

## 📊 Data Visualization
To visualize the clustering results in **VS Code**:
1. Run the script in **Jupyter Notebook** or enable inline plotting in VS Code using:
   ```python
   %matplotlib inline
   ```
2. Generate plots using Seaborn:
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   sns.scatterplot(x=df['Total Spend'], y=df['Items Purchased'], hue=clusters)
   plt.show()
   ```

## 📁 Project Structure
```
📂 customer_segmentation
│── customer_segmentation.py  # Main script
│── data.csv                  # Customer data file
│── requirements.txt           # Dependencies
│── README.md                  # Project documentation
└── 📂 visualizations         # Folder containing generated plots
```

## 🏆 Project Deliverables
- ✅ **GitHub Repository Submission**
- ✅ **LinkedIn Post with Project Video**
- ✅ **Email Submission (PDF + Dataset + GitHub Link)**

## 👤 Author
**Shreya S.**

🔗 Connect with me on LinkedIn: [@shreya31s](https://www.linkedin.com/in/shreya31s/)
