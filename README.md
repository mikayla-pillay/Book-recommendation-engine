# Book-recommendation-engine
A machine learning recommendation system built as a university group project for COMP316. Recommends books based on content similarity, comparing two NLP approaches: TF-IDF vectorisation and SBERT (sentence transformers).

What it does:

Preprocesses a Goodreads dataset through a full NLP pipeline: lowercasing, stopword removal, lemmatization (NLTK WordNetLemmatizer), and bigram extraction
Builds a TF-IDF model and an SBERT model, both using cosine similarity to find the most similar books to a query
Evaluates both models using Precision@5, Precision@10, and NDCG@10
Visualises results with matplotlib bar charts comparing model performance

Technologies: Python 3, scikit-learn, sentence-transformers, NLTK, pandas, matplotlib, Google Colab
How to run:
Open COMP316_GroupProject.ipynb in Google Colab or Jupyter Notebook and run all cells.
What I learned: NLP preprocessing pipelines, TF-IDF vectorisation, cosine similarity, sentence embeddings (SBERT), and evaluation metrics for recommendation systems.
