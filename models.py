import pandas as pd
from sklearn.neighbors import NearestNeighbors

def load_data():
    try:
        movies = pd.read_csv("movies.csv")
        ratings = pd.read_csv("ratings.csv")
        print("Movies and ratings data loaded successfully.")
        return movies, ratings
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise e

def get_movie_list(limit=None):
    movies, _ = load_data()
    if limit:
        movies = movies.head(limit)
    print("Movies for dropdown:", movies[['movieId', 'title']].to_dict('records'))
    return movies[['movieId', 'title']].to_dict('records')

def recommend_movies(movie_id, n_recommendations=5):
    movies, _ = load_data()

    if movie_id not in movies['movieId'].values:
        raise ValueError("Movie ID not found in the dataset")

    movie_features = movies['genres'].str.get_dummies(sep='|')

    model = NearestNeighbors(n_neighbors=n_recommendations+1, algorithm='auto')
    model.fit(movie_features)

    movie_idx = movies.index[movies['movieId'] == movie_id].tolist()[0]
    distances, indices = model.kneighbors([movie_features.iloc[movie_idx]])

    recommendations = []
    for i in range(1, n_recommendations+1):
        recommended_movie_id = movies.iloc[indices.flatten()[i]]['movieId']
        recommendations.append(movies[movies['movieId'] == recommended_movie_id].title.values[0])

    return recommendations
