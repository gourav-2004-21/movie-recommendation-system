import pandas as pd
import ast
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =============================
# Load datasets
# =============================
movies = pd.read_csv("tmdb_5000_movies.csv", encoding="utf-8", engine="python")


credits = pd.read_csv(
    "tmdb_5000_credits.csv",
    encoding="utf-8",
    engine="python"
)
# =============================
# Merge datasets
# =============================
movies = movies.merge(credits, on='title')

# =============================
# Select required columns
# =============================
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
movies.dropna(inplace=True)

# =============================
# Helper functions
# =============================
def convert(text):
    return [i['name'] for i in ast.literal_eval(text)]

def convert_cast(text):
    return [i['name'] for i in ast.literal_eval(text)[:3]]

def fetch_director(text):
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            return [i['name']]
    return []

# =============================
# Apply transformations
# =============================
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert_cast)
movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# =============================
# Remove spaces
# =============================
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

# =============================
# Create tags
# =============================
movies['tags'] = (
    movies['overview']
    + movies['genres']
    + movies['keywords']
    + movies['cast']
    + movies['crew']
)

# =============================
# Final dataframe
# =============================
new_df = movies[['movie_id','title','tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

# =============================
# Vectorization
# =============================
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# =============================
# Cosine similarity
# =============================
similarity = cosine_similarity(vectors)

# =============================
# Save pickle files
# =============================
pickle.dump(new_df, open('movies.pkl','wb'))
pickle.dump(similarity, open('similarity.pkl','wb'))

print("✅ Model trained and pickle files created successfully!")
