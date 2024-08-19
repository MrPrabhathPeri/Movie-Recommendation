from flask import Flask, request, render_template
import pandas as pd
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Load movies data
def load_movies():
    try:
        movies = pd.read_csv("movies.csv")
        movies = movies[['movieId', 'title']].head(30)
        return movies.to_dict('records')
    except Exception as e:
        print(f"Error loading movies: {e}")
        return []

def recommend_movies(movie_id, n_recommendations=5):
    try:
        movies = pd.read_csv("movies.csv")
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
    except Exception as e:
        print(f"Error in recommendation system: {e}")
        return []

@app.route("/", methods=["GET", "POST"])
def index():
    movies = load_movies()
    recommendations = None

    if request.method == "POST":
        try:
            movie_id = int(request.form["movie_id"])
            recommendations = recommend_movies(movie_id)
        except Exception as e:
            print(f"Error during recommendation: {e}")
            return f"An error occurred: {e}"

    return render_template("index.html", movies=movies, recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
