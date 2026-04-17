from sklearn.cluster import KMeans
from src.pipeline import run

css_df = run()
X = css_df[["success_rate_scaled", "explosiveness_scaled", "reliability_scaled"]]
kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
css_df["cluster"] = kmeans.labels_

css_df.to_csv('outputs/css_df_clustered.csv')

clustered_df = css_df.groupby(css_df['cluster']).count()
print(clustered_df.head(20))