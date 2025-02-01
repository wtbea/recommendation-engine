from sklearn import neighbors
from sklearn.metrics.pairwise import cosine_similarity
import polars as pl
from collections import Counter
from scipy.sparse import csr_matrix
import numpy as np
ratings = pl.read_csv('./data/ratings.csv')
movies = pl.read_csv('./data/movies.csv')

n_ratings = len(ratings) 
n_movies = ratings['movieId'].n_unique()
n_users = ratings['userId'].n_unique()

mean_ratings = ratings.group_by('userId').agg(pl.col('rating').mean())

mean_rating_value = mean_ratings['rating'].mean()
rounded_mean_rating = round(mean_rating_value, 2)

movie_ratings = ratings.join(movies, on='movieId')
five_star_ratings = movie_ratings.filter(pl.col('rating') == 5.0)

five_star_counts = five_star_ratings.group_by('title').agg(pl.count('rating').alias('five_star_count'))

top_movies = five_star_counts.sort('five_star_count', descending=True).head(10)

genre_frequency = Counter(g for genres in movies['genres'] for g in genres.split('|'))

#print(f"There are {len(genre_frequency)} genres.")
#print("The 5 most common genres: \n", genre_frequency.most_common(5))
#print(f"Number of ratings: {n_ratings}")
#print(f"Number of unique movieId's: {n_movies}")
#print(f"Number of unique users: {n_users}")
#print(f"Average number of ratings per user: {round(n_ratings/n_users, 2)}")
#print(f"Average number of ratings per movie: {round(n_ratings/n_movies, 2)}")
#print(f"Mean rating per user: {rounded_mean_rating}.")

def create_X(df):

    M = df['userId'].n_unique()
    N = df['movieId'].n_unique()

    user_mapper = dict(zip(np.unique(df["userId"]), list(range(M))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(N))))

    user_inv_mapper = dict(zip(list(range(M)), np.unique(df["userId"])))
    movie_inv_mapper = dict(zip(list(range(N)), np.unique(df["movieId"])))

    user_index = [user_mapper[i] for i in df['userId']]
    item_index = [movie_mapper[i] for i in df['movieId']]

    X = csr_matrix((df["rating"], (user_index,item_index)), shape=(M,N))
    
    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_X(ratings)

n_total = X.shape[0]*X.shape[1]
n_ratings = X.nnz
sparsity = n_ratings/n_total
#print(f"Matrix sparsity: {round(sparsity*100,2)}%")

n_ratings_per_user = X.getnnz(axis=1)

#print(f"Most active user rated {n_ratings_per_user.max()} movies.")
#print(f"Least active user rated {n_ratings_per_user.min()} movies.")
n_ratings_per_movie = X.getnnz(axis=0)

#print(f"Most rated movie has {n_ratings_per_movie.max()} ratings.")
#print(f"Least rated movie has {n_ratings_per_movie.min()} ratings.")

def find_similar_movies(movie_id, X, movie_mapper, movie_inv_mapper, k, metric='cosine'):
    X = X.T
    neighbour_ids = []
    
    movie_ind = movie_mapper[movie_id]
    movie_vec = X[movie_ind]
    if isinstance(movie_vec, (np.ndarray)):
        movie_vec = movie_vec.reshape(1,-1)
    kNN = neighbors.NearestNeighbors(n_neighbors=k+1, algorithm="brute", metric=metric)
    kNN.fit(X)
    neighbour = kNN.kneighbors(movie_vec, return_distance=False)
    for i in range(0,k):
        n = neighbour.item(i)
        neighbour_ids.append(movie_inv_mapper[n])
    neighbour_ids.pop(0)
    return neighbour_ids

similar_movies = find_similar_movies(1, X, movie_mapper, movie_inv_mapper, k=10)
movie_titles = dict(zip(movies['movieId'], movies['title']))

movie_id = 159858

similar_movies = find_similar_movies(movie_id, X, movie_mapper, movie_inv_mapper, metric='cosine', k=10)
movie_title = movie_titles[movie_id]

print(f"Because you watched {movie_title}:")
for i in similar_movies:
    print(movie_titles[i])

print(f"There are {n_movies} unique movies in our movies dataset.")

genres = set(g for G in movies['genres'] for g in G.split('|'))  # Split genres into individual ones

for g in genres:
    movies = movies.with_columns(
        (movies['genres'].apply(lambda x: int(g in x))).alias(g)  # Create binary columns for each genre
    )

# Drop unnecessary columns
movie_genres = movies.drop(['movieId', 'title', 'genres'])

cosine_sim = cosine_similarity(movie_genres, movie_genres)
print(f"Dimensions of our genres cosine similarity matrix: {cosine_sim.shape}")