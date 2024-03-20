from flask import Flask, render_template, request
import random
import requests
import pandas as pd
import numpy as np
from ast import literal_eval

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from surprise import SVD, Reader, Dataset
from surprise.model_selection import train_test_split, cross_validate
import warnings
from pandas.errors import SettingWithCopyWarning
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

app = Flask(__name__)


# Read CSV files
def read_csv_chunks(filename, chunk_size=50):
    chunks = pd.read_csv(filename, low_memory=True, chunksize=chunk_size)
    return pd.concat(chunks)


credits = read_csv_chunks("archive/credits.csv")
keywords = read_csv_chunks("archive/keywords.csv")
links_small = read_csv_chunks("archive/links_small.csv")
md = read_csv_chunks("archive/movies_metadata.csv")
ratings = read_csv_chunks("archive/ratings_small.csv")

if "genres" in md.columns:
    md["genres"] = (
        md["genres"]
        .fillna("[]")
        .apply(literal_eval)
        .apply(lambda x: [i["name"] for i in x] if isinstance(x, list) else [])
    )
else:
    md["genres"] = [[]] * len(md)

# Try-except block to handle missing columns gracefully
try:
    vote_counts = md[md["vote_count"].notnull()]["vote_count"].astype("int")
    vote_averages = md[md["vote_average"].notnull()]["vote_average"].astype("int")
except KeyError:
    print("'vote_count' and/or 'vote_average' columns not found in DataFrame")
    vote_counts = pd.Series([], dtype=int)
    vote_averages = pd.Series([], dtype=int)

# Calculate parameters for weighted rating
C = vote_averages.mean()
m = vote_counts.quantile(0.95)

# Filter qualified movies based on vote count and average
try:
    qualified = md[
        (md["vote_count"] >= m)
        & (md["vote_count"].notnull())
        & (md["vote_average"].notnull())
    ]
    qualified = qualified[
        ["title", "vote_count", "vote_average", "popularity", "genres"]
    ]
    qualified["vote_count"] = qualified["vote_count"].astype("int")
    qualified["vote_average"] = qualified["vote_average"].astype("int")
except KeyError:
    print("'vote_count' and/or 'vote_average' columns not found in DataFrame")
    qualified = pd.DataFrame(
        [], columns=["title", "vote_count", "vote_average", "popularity", "genres"]
    )


# Calculate weighted rating for qualified movies
def weighted_rating(x):
    v = x["vote_count"]
    R = x["vote_average"]
    return (v / (v + m) * R) + (m / (m + v) * C)


qualified["wr"] = qualified.apply(weighted_rating, axis=1)
qualified = qualified.sort_values("wr", ascending=False).head(250)

# Expand 'genres' into multiple columns
s = (
    md.apply(lambda x: pd.Series(x["genres"]), axis=1)
    .stack()
    .reset_index(level=1, drop=True)
)
s.name = "genre"
gen_md = md.drop("genres", axis=1).join(s)
gen_md.head(3).transpose()


# percentile -> compares individual data points to the overall distribution.
def build_chart(genre, percentile=0.85):
    try:
        # Filter movies based on genre
        df = gen_md[gen_md["genre"] == genre]

        # Extract vote count and average
        vote_counts = df[df["vote_count"].notnull()]["vote_count"].astype("int")
        vote_averages = df[df["vote_average"].notnull()]["vote_average"].astype("int")
        C = vote_averages.mean()
        m = vote_counts.quantile(percentile)

        # Select movies with vote counts greater than or equal to m
        qualified = df[
            (df["vote_count"] >= m)
            & (df["vote_count"].notnull())
            & (df["vote_average"].notnull())
        ][["title", "year", "vote_count", "vote_average", "popularity"]]
        qualified["vote_count"] = qualified["vote_count"].astype("int")
        qualified["vote_average"] = qualified["vote_average"].astype("int")

        # Calculate weighted rating for qualified movies
        qualified["wr"] = qualified.apply(
            lambda x: (x["vote_count"] / (x["vote_count"] + m) * x["vote_average"])
            + (m / (m + x["vote_count"]) * C),
            axis=1,
        )

        # Sort and select top movies
        qualified = qualified.sort_values("wr", ascending=False).head(250)

        return qualified

    except KeyError:
        print(f"'{genre}' genre not found or missing columns in DataFrame")
        return pd.DataFrame(
            [],
            columns=["title", "year", "vote_count", "vote_average", "popularity", "wr"],
        )


# Example usage:
# top_action_movies = build_chart("Action")
# print(top_action_movies)

try:
    links_small = links_small[links_small["tmdbId"].notnull()]["tmdbId"].astype("int")
except KeyError:
    print("'tmdbId' column not found in 'links_small' DataFrame")
    links_small = pd.Series([], dtype=int)


## Pre-processing step


def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan


try:
    # Convert 'id' column to integers using the convert_int function
    md["id"] = md["id"].apply(convert_int)

    # Drop rows with null IDs
    md = md.dropna(subset=["id"])

except Exception as e:
    print(f"Error: {e}")
    md = pd.DataFrame([], columns=md.columns)

try:
    # Convert 'id' columns to integers
    keywords["id"] = keywords["id"].astype("int")
    credits["id"] = credits["id"].astype("int")
    md["id"] = md["id"].astype("int")

    # Merge dataframes to create smd for content-based recommendation
    md = md.merge(credits, on="id").merge(keywords, on="id")
    smd = md[md["id"].isin(links_small)]

    # Preprocess data: convert strings to lists, calculate cast/crew size
    smd["cast"] = smd["cast"].apply(literal_eval)
    smd["crew"] = smd["crew"].apply(literal_eval)
    smd["keywords"] = smd["keywords"].apply(literal_eval)
    smd["cast_size"] = smd["cast"].apply(len)
    smd["crew_size"] = smd["crew"].apply(len)

    # Create description by combining overview and tagline
    smd["tagline"] = smd["tagline"].fillna("")
    smd["description"] = smd["overview"] + smd["tagline"]
    smd["description"] = smd["description"].fillna("")

    # Use TF-IDF vectorization for movie descriptions
    tfidf_vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
    tfidf_matrix = tfidf_vectorizer.fit_transform(smd["description"])

    # Compute cosine similarity for recommendation
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Reset index and create indices Series for quick lookup
    smd = smd.reset_index()
    titles = smd["title"]
    indices = pd.Series(smd.index, index=smd["title"])

except Exception as e:
    print(f"Error: {e}")


# Function to get recommendations based on title
def get_recommendations(title):
    try:
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:31]  # Get top 30 similar movies
        movie_indices = [i[0] for i in sim_scores]
        return titles.iloc[
            movie_indices
        ].tolist()  # Return list of recommended movie titles
    except KeyError:
        print(f"'{title}' not found in movie titles")
        return []


def get_director(x):
    for i in x:
        if i["job"] == "Director":
            return i["name"]
    return np.nan


try:
    # Apply get_director function to 'crew' column to extract directors
    smd.loc[:, "director"] = smd["crew"].apply(get_director)

    # Process 'cast' and 'keywords' columns
    smd.loc[:, "cast"] = smd["cast"].apply(
        lambda x: [i["name"] for i in x] if isinstance(x, list) else []
    )
    smd.loc[:, "cast"] = smd["cast"].apply(lambda x: x[:3] if len(x) >= 3 else x)
    smd.loc[:, "keywords"] = smd["keywords"].apply(
        lambda x: [i["name"] for i in x] if isinstance(x, list) else []
    )

    # Lowercase and remove spaces from 'cast' and 'director' columns
    smd.loc[:, "cast"] = smd["cast"].apply(
        lambda x: [str.lower(i.replace(" ", "")) for i in x]
    )
    smd.loc[:, "director"] = (
        smd["director"].astype("str").apply(lambda x: str.lower(x.replace(" ", "")))
    )

    # Duplicate 'director' column to ensure it contributes more to the similarity score
    smd.loc[:, "director"] = smd["director"].apply(lambda x: [x, x, x])

    # Count occurrences of keywords to filter out rare keywords
    s = smd.apply(lambda x: pd.Series(x["keywords"]), axis=1).stack().value_counts()
    s = s[s > 1]

    # Initialize stemmer for keyword stemming
    stemmer = SnowballStemmer("english")

    # Function to filter keywords and perform stemming
    def filter_keywords(x):
        words = []
        for i in x:
            if i in s:
                words.append(stemmer.stem(i))
        return words

    # Apply keyword filtering and stemming
    smd.loc[:, "keywords"] = smd["keywords"].apply(filter_keywords)
    smd.loc[:, "keywords"] = smd["keywords"].apply(
        lambda x: [str.lower(i.replace(" ", "")) for i in x]
    )

    # Combine 'keywords', 'cast', 'director', and 'genres' into 'soup'
    smd.loc[:, "soup"] = smd["keywords"] + smd["cast"] + smd["director"] + smd["genres"]
    smd.loc[:, "soup"] = smd["soup"].apply(lambda x: " ".join(x))

    # Vectorize 'soup' using CountVectorizer
    count_vectorizer = CountVectorizer(stop_words="english", min_df=1)
    count_matrix = count_vectorizer.fit_transform(smd["soup"])

    # Compute cosine similarity based on count_matrix
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    # Reset index and create indices Series for quick lookup
    smd = smd.reset_index()
    titles = smd["title"]
    indices = pd.Series(smd.index, index=smd["title"])

except Exception as e:
    print(f"Error: {e}")
    # Handle error or set smd to empty DataFrame if necessary
    smd = pd.DataFrame([], columns=smd.columns)
    indices = pd.Series([], dtype=int)
    cosine_sim = None
    titles = pd.Series([], dtype=str)


def improved_recommendations(title):
    try:
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:26]  # Exclude the most similar movie (itself)
        movie_indices = [i[0] for i in sim_scores]

        # Select relevant columns for recommended movies
        movies = smd.iloc[movie_indices][
            ["title", "vote_count", "vote_average", "year"]
        ]

        # Filter movies with valid vote count and average
        vote_counts = movies[movies["vote_count"].notnull()]["vote_count"].astype("int")
        vote_averages = movies[movies["vote_average"].notnull()]["vote_average"].astype(
            "int"
        )

        # Calculate C (mean of vote averages) and m (quantile of vote counts)
        C = vote_averages.mean()
        m = vote_counts.quantile(0.60)

        # Filter qualified movies based on vote count and average
        qualified = movies[
            (movies["vote_count"] >= m)
            & (movies["vote_count"].notnull())
            & (movies["vote_average"].notnull())
        ]
        qualified["vote_count"] = qualified["vote_count"].astype("int")
        qualified["vote_average"] = qualified["vote_average"].astype("int")

        # Calculate weighted rating for qualified movies using weighted_rating function
        qualified["wr"] = qualified.apply(weighted_rating, axis=1)

        # Sort by weighted rating and select top 10 movies
        qualified = qualified.sort_values("wr", ascending=False).head(10)

        return qualified

    except KeyError:
        print(f"'{title}' not found in movie titles")
        return pd.DataFrame([], columns=["title", "vote_count", "vote_average", "year"])

    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame([], columns=["title", "vote_count", "vote_average", "year"])


# Example usage:
# recommended_movies = improved_recommendations("The Dark Knight")
# print(recommended_movies)


# CF based recommendation system
from surprise import SVD
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split, cross_validate
from surprise.accuracy import rmse, mae
import requests

# Assuming you have already defined 'reader' and 'ratings'
reader = Reader()
data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Create an instance of the SVD model with desired hyperparameters
svd = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)

# Use cross_validate for cross-validation with desired metrics
cv_results = cross_validate(svd, data, measures=["RMSE", "MAE"], cv=5, verbose=True)

# Extract RMSE and MAE from cross-validation results
avg_rmse = sum(cv_results["test_rmse"]) / len(cv_results["test_rmse"])
avg_mae = sum(cv_results["test_mae"]) / len(cv_results["test_mae"])

print(f"Average RMSE: {avg_rmse}")
print(f"Average MAE: {avg_mae}")

# Fit the model to the entire training set
trainset = data.build_full_trainset()
svd.fit(trainset)

# Example: Predict rating for user 1 and item 302
predicted_rating = svd.predict(1, 302).est
print(f"Predicted rating for user 1 and item 302: {predicted_rating}")

# Evaluate model on test set for final performance metrics
test_predictions = svd.test(testset)
test_rmse = rmse(test_predictions)
test_mae = mae(test_predictions)

print(f"Test RMSE: {test_rmse}")
print(f"Test MAE: {test_mae}")


# Hybrid recommendation system
def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan


try:
    id_map = pd.read_csv("./archive/links_small.csv")[["movieId", "tmdbId"]]
    id_map["tmdbId"] = id_map["tmdbId"].apply(convert_int)
    id_map.columns = ["movieId", "id"]

    # Merge id_map with smd to get the corresponding titles
    id_map = id_map.merge(smd[["title", "id"]], on="id").set_index("title")

    # Create indices_map for quick lookup of movie IDs
    indices_map = id_map.set_index("id")

except FileNotFoundError:
    print("Error: File not found.")
    id_map = pd.DataFrame([], columns=["movieId", "id"])
    indices_map = pd.DataFrame([], columns=["movieId", "id"])

except Exception as e:
    print(f"Error: {e}")
    id_map = pd.DataFrame([], columns=["movieId", "id"])
    indices_map = pd.DataFrame([], columns=["movieId", "id"])


def hybrid(title, min_userId=1, max_userId=600):
    try:
        userId = random.randint(min_userId, max_userId)  # Generate random userId
        idx = indices[title]
        tmdbId = id_map.loc[title]["id"]
        movie_id = id_map.loc[title]["movieId"]
        sim_scores = list(enumerate(cosine_sim[int(idx)]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:26]
        movie_indices = [i[0] for i in sim_scores]
        movies = smd.iloc[movie_indices][
            ["title", "vote_count", "vote_average", "release_date", "id"]
        ]
        movies["est"] = movies["id"].apply(
            lambda x: svd.predict(userId, indices_map.loc[x]["movieId"]).est
        )
        movies = movies.sort_values("est", ascending=False)

        # Return movie details along with names
        return movies.head(12)
    except KeyError:
        raise IndexError("Movie not found. Please enter a valid movie title.")


# Replace with your TMDB API key
TMDB_API_KEY = "e65f96397db5471ad7bab643b6f327ca"


def get_movie_details(movie_title):
    base_url = "https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "query": movie_title,
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    if "results" in data and data["results"]:
        movie_details = data["results"][0]
        return {
            "title": movie_details.get("title", ""),
            "overview": movie_details.get("overview", ""),
            "poster_path": "https://image.tmdb.org/t/p/w500/"
            + movie_details.get("poster_path", ""),
        }

    raise IndexError(f"Movie not found: {movie_title}")


def get_recommendations(movie_title):
    search_url = "https://api.themoviedb.org/3/search/movie"
    recommendations_url = (
        "https://api.themoviedb.org/3/movie/{movie_id}/recommendations"
    )

    # Step 1: Search for the given movie title
    search_params = {
        "api_key": TMDB_API_KEY,
        "query": movie_title,
    }

    search_response = requests.get(search_url, params=search_params)
    search_data = search_response.json()

    if "results" in search_data and search_data["results"]:
        movie_id = search_data["results"][0].get("id")

        # Step 2: Get recommendations for the found movie ID
        recommendations_params = {
            "api_key": TMDB_API_KEY,
        }

        recommendations_response = requests.get(
            recommendations_url.format(movie_id=movie_id), params=recommendations_params
        )
        recommendations_data = recommendations_response.json()

        recommendations = []
        for result in recommendations_data.get("results", [])[:13]:
            recommendations.append(
                {
                    "title": result.get("title", ""),
                    "overview": result.get("overview", ""),
                    "poster_path": "https://image.tmdb.org/t/p/w500/"
                    + result.get("poster_path", ""),
                }
            )

        return recommendations

    else:
        raise IndexError(f"Movie not found: {movie_title}")


@app.route("/")
def home():
    return render_template("index2.html")


@app.route("/recommend", methods=["POST"])
def recommend():
    if request.method == "POST":
        movie_title = request.form["movie_title"]
        try:
            # Use hybrid function to get recommendations
            recommendations_data = hybrid(movie_title)
            recommendations = []
            for index, row in recommendations_data.iterrows():
                recommendations.append(
                    {
                        "title": row["title"],
                        "overview": get_movie_details(row["title"])["overview"],
                        "poster_path": get_movie_details(row["title"])["poster_path"],
                    }
                )

            return render_template(
                "index2.html",
                movie_details=get_movie_details(movie_title),
                recommendations=recommendations,
            )

        except IndexError as e:
            print(f"Error in getting movie details: {e}")
            return render_template("index2.html", error_message=str(e))


if __name__ == "__main__":
    app.run(debug=True)
