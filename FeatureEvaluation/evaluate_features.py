import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE, Isomap
from sklearn.metrics import silhouette_score

# 设置文件路径
feature_folders = {
    "channel": "feature/channel/0618_channel_features.csv",
    "hog": "feature/hog/0618_hog_features.csv",
    "cnn": "feature/cnn/0618_cnn_features.csv"
}

# feature_folders = {
#     "channel": "feature/channel/0854_channel_features.csv",
#     "hog": "feature/hog/0854_hog_features.csv",
#     "cnn": "feature/cnn/0854_cnn_features.csv"
# }

# feature_folders = {
#     "channel": "feature/channel/1066_channel_features.csv",
#     "hog": "feature/hog/1066_hog_features.csv",
#     "cnn": "feature/cnn/1066_cnn_features.csv"
# }

# 读取CSV文件并加载数据
def load_features(folder_path):
    df = pd.read_csv(folder_path)
    data = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values

    return data, labels

# 聚类算法评估
def cluster_and_evaluate(data, labels, n_clusters_range):
    best_score = -1
    best_k = 2
    kmeans_results = []
    for k in n_clusters_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        silhouette_avg = silhouette_score(data, kmeans.labels_)
        kmeans_results.append((k, silhouette_avg))
        print(f"The number of clusters: {k} with Silhouette Score: {silhouette_avg}")
        if silhouette_avg > best_score:
            best_score = silhouette_avg
            best_k = k
    print(f"Best number of clusters: {best_k} with Silhouette Score: {best_score}")
    # 绘制聚类效果
    best_kmeans = KMeans(n_clusters = best_k, random_state=42)
    best_kmeans.fit(data)
    return best_kmeans.labels_


# 流形学习降维
def manifold_learning(data, method = 'tsne', n_components = 2):
    if method == 'tsne':
        model = TSNE(n_components = n_components, random_state=42)
    elif method == 'isomap':
        model = Isomap(n_components = n_components)
    else:
        raise ValueError("Unsupported method. Choose 'tsne' or 'isomap'.")
    return model.fit_transform(data)


# 可视化聚类结果
def plot_clustered_data(data, labels, title = "Clustered Data"):
    plt.figure(figsize = (10, 8))
    sns.scatterplot(x = data[:, 0], y = data[:, 1], hue = labels, palette = "Set1", s = 80, edgecolor = 'k')
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend()
    plt.show()


# 主程序
def main():
    # 加载所有特征
    channel_data, channel_labels = load_features(feature_folders['channel'])
    hog_data, hog_labels = load_features(feature_folders['hog'])
    cnn_data, cnn_labels = load_features(feature_folders['cnn'])


    # 聚类算法评估
    # 选择簇的数量范围
    n_clusters_range = range(2, 6)
    print("Evaluating Channel Features...")
    channel_labels_pred = cluster_and_evaluate(channel_data, channel_labels, n_clusters_range)
    print("Evaluating HOG Features...")
    hog_labels_pred = cluster_and_evaluate(hog_data, hog_labels, n_clusters_range)
    print("Evaluating CNN Features...")
    cnn_labels_pred = cluster_and_evaluate(cnn_data, cnn_labels, n_clusters_range)

    # 流形学习降维并可视化
    print("Dimensionality Reduction for Channel Features (t-SNE)...")
    channel_tsne = manifold_learning(channel_data, method = 'tsne')
    plot_clustered_data(channel_tsne, channel_labels, "Channel Features - t-SNE")

    print("Dimensionality Reduction for HOG Features (t-SNE)...")
    hog_tsne = manifold_learning(hog_data, method = 'tsne')
    plot_clustered_data(hog_tsne, hog_labels, "HOG Features - t-SNE")

    print("Dimensionality Reduction for CNN Features (t-SNE)...")
    cnn_tsne = manifold_learning(cnn_data, method='tsne')
    plot_clustered_data(cnn_tsne, cnn_labels, "CNN Features - t-SNE")

if __name__ == "__main__":
    main()
