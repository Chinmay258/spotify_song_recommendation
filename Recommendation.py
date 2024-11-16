import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# Step 1: Load the Dataset
df = pd.read_csv('spotify_dataset.csv')

# Step 2: Preprocessing
df = df.dropna(subset=['track_name', 'track_artist'])

features = ['danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)

# Step 3: Exploratory Data Analysis
# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df[features].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.savefig("correlation_matrix.png")
plt.close()

# Feature distributions
for feature in features:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[feature], kde=True, color='blue')
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.savefig(f"distribution_{feature}.png")
    plt.close()

# Step 4: Clustering
inertia = []
range_clusters = range(1, 11)
for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow curve
plt.figure(figsize=(8, 4))
plt.plot(range_clusters, inertia, marker='o', linestyle='-', color='blue')
plt.title("Elbow Method For Optimal Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.savefig("elbow_plot.png")
plt.close()

optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['cluster'] = kmeans.fit_predict(df_scaled)

# Step 5: Visualizing Clusters using t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_features = tsne.fit_transform(df_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=tsne_features[:, 0], y=tsne_features[:, 1], hue=df['cluster'], palette="viridis")
plt.title("Clusters of Songs (t-SNE Visualization)")
plt.xlabel("t-SNE Feature 1")
plt.ylabel("t-SNE Feature 2")
plt.legend(title="Cluster")
plt.savefig("tsne_clusters.png")
plt.close()

# Step 6: Recommendation System
def recommend_songs(song_name, df, n_recommendations=5):

    # Find the cluster of the input song
    matching_songs = df[df['track_name'].str.contains(song_name, case=False, na=False)]
    if matching_songs.empty:
        return f"No song found with name: {song_name}"

    song_cluster = matching_songs.iloc[0]['cluster']

    # Filter songs in the same cluster
    similar_songs = df[df['cluster'] == song_cluster]

    # Exclude the input song from recommendations
    similar_songs = similar_songs[~similar_songs['track_name'].str.contains(song_name, case=False, na=False)]

    # Randomly select n recommendations
    recommendations = similar_songs.sample(n=min(n_recommendations, len(similar_songs)))
    return recommendations[['track_name', 'track_artist', 'playlist_genre']]


song_name = "Love Me Away" # Select songs from dataset
recommendations = recommend_songs(song_name, df)
print(recommendations)


