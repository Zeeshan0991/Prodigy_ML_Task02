
# Customer Segmentation Using K-Means Clustering

## Overview

This project demonstrates customer segmentation using the K-Means Clustering algorithm. The goal is to group customers based on their annual income and spending scores, providing insights into distinct customer segments.

## Project Structure

- **1. Importing Necessary Libraries**: Importing essential libraries such as numpy, pandas, matplotlib, seaborn, and sklearn's KMeans.
- **2. Data Collection and Analysis**: Loading and examining the dataset.
- **3. Data Preprocessing**: Selecting relevant columns for clustering.
- **4. Choosing the Number of Clusters**: Using the Within-Cluster Sum of Squares (WCSS) method to determine the optimal number of clusters.
- **5. Training the K-Means Clustering Model**: Applying the K-Means algorithm to the selected data.
- **6. Visualization**: Plotting the clustered data points and centroids for visualization.

## Key Steps

1. **Import Libraries**:
    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.cluster import KMeans
    ```

2. **Load Data**:
    ```python
    customer_data = pd.read_csv('Mall_Customers.csv')
    ```

3. **Data Exploration**:
    ```python
    print(customer_data.head())
    print(customer_data.shape)
    print(customer_data.info())
    print(customer_data.isnull().sum())
    ```

4. **Select Features for Clustering**:
    ```python
    X = customer_data.iloc[:, [3, 4]].values
    ```

5. **Determine Optimal Number of Clusters**:
    ```python
    wss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X)
        wss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wss)
    plt.title('The Elbow Point Graph')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.show()
    ```

6. **Train the K-Means Model**:
    ```python
    kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
    Y = kmeans.fit_predict(X)
    print(Y)
    ```

7. **Visualize Clusters**:
    ```python
    plt.figure(figsize=(8, 8))
    plt.scatter(X[Y==0, 0], X[Y==0, 1], s=50, c='green', label='Cluster 1')
    plt.scatter(X[Y==1, 0], X[Y==1, 1], s=50, c='red', label='Cluster 2')
    plt.scatter(X[Y==2, 0], X[Y==2, 1], s=50, c='yellow', label='Cluster 3')
    plt.scatter(X[Y==3, 0], X[Y==3, 1], s=50, c='violet', label='Cluster 4')
    plt.scatter(X[Y==4, 0], X[Y==4, 1], s=50, c='blue', label='Cluster 5')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='cyan', label='Centroids')
    plt.title('Customer Groups')
    plt.xlabel('Annual Income')
    plt.ylabel('Spending Score')
    plt.legend()
    plt.show()
    ```

## Dataset

The dataset used in this project is `Mall_Customers.csv`, which contains the following columns:
- CustomerID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1-100)

## Results

The project successfully identified 5 distinct customer segments based on annual income and spending score, which can be visualized using scatter plots with the cluster centroids marked.

## Conclusion

This project demonstrates the application of K-Means Clustering for customer segmentation. The insights gained from clustering can help businesses understand their customer base better and tailor marketing strategies accordingly.

## Repository Link

You can find the complete code and analysis in the GitHub repository [https://github.com/Zeeshan0991/Prodigy_ML_Task02]

## Acknowledgements

Thanks to ProdigyInfoTech for the opportunity to work on this interesting project.

#ProdigyInfoTech #MachineLearning #Task02 #Customer_Segmentation #KMeansClustering
