import streamlit as st
import pickle

# =============================
# Load data
# =============================
movies = pickle.load(open('movies.pkl','rb'))
similarity = pickle.load(open('similarity.pkl','rb'))

# =============================
# Recommendation function
# =============================
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []
    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title)

    return recommended_movies

# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox(
    "Select a movie you like:",
    movies['title'].values
)

if st.button("Recommend"):
    st.subheader("Recommended Movies:")
    for movie in recommend(selected_movie):
        st.write("👉", movie)
