from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 📝 Sample Documents (like resumes, web pages, etc.)
documents = [
    "Python developer with knowledge in machine learning and Flask.",
    "Experienced Java developer who builds Android applications.",
    "Skilled in data analysis, NumPy, Pandas, and visualization.",
    "Cybersecurity analyst with network monitoring skills.",
]

# 🔍 User Query (like a job description or search phrase)
query = "looking for a python flask machine learning engineer"

# 🧠 TF-IDF Vectorization
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents + [query])

# 📐 Cosine Similarity
cos_sim = cosine_similarity(doc_vectors[-1], doc_vectors[:-1])

# 📊 Display Results
print("🔍 IR - Search Results:\n")
for i, score in enumerate(cos_sim[0]):
    print(f"Doc {i+1} Score: {score:.4f} - {documents[i]}")
