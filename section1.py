original=r'Natural Language Processing (NLP) enables computers to understand and interpret human language. While computers excel at processing structured data, such as spreadsheets or databases, natural language in its unstructured form (text, speech, etc.) presents a unique challenge. NLP bridges this gap by allowing machines to process and understand human languages, making it an essential tool in modern AI systems.'
#Source for corpus: https://www.geeksforgeeks.org/nlp/introduction-to-natural-language-processing/
print('Original Corpus:\n',original)
import nltk
corpus=nltk.sent_tokenize(original)
print('Sentence-tokenized:\n',corpus)
#Lowercasing
lowercased_corpus=[i.lower() for i in corpus]
print('Lowercased Corpus: ',)
print(lowercased_corpus)
#Removing Punctuation & Special Characters
import re
corpus_cleaned = [re.sub(r'[^\w\s]', '', i) for i in lowercased_corpus]
print('Removed Punctuation & Special Characters from all lines in the corpus:')
print(corpus_cleaned)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# Remove stopwords function for any language
def remove_stopwords(text, language):
    stop_words = set(stopwords.words(language))
    word_tokens = text.split()
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_text)
stopremov=[remove_stopwords(i, "english") for i in corpus_cleaned]
print('Removed stopwords:')
print(stopremov)
#Remove URLs
def remove_urls(text):
    return re.compile(r'https?://\S+|www\.\S+').sub(r'', text)
print('Removed URLs')
removed_url=[remove_urls(i) for i in stopremov]
print(removed_url)
#Remove HTML Tags
html_tags_pattern = r'<.*?>'
removed_html_tags=[re.sub(html_tags_pattern, '', i) for i in removed_url]
print('Removed HTML Tags')
print(removed_html_tags)
# from nltk.stem.porter import PorterStemmer
# stemmer = PorterStemmer()
# def stem_words(text):
#     word_tokens = text.split()
#     stems = [stemmer.stem(word) for word in word_tokens]
#     return ' '.join(stems)
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
def lemmatize_word(text):
    word_tokens = text.split()
    lemmas = [lemmatizer.lemmatize(word, pos ='v') for word in word_tokens]
    return ' '.join(lemmas)
sl=[lemmatize_word(i) for i in removed_html_tags]
print('Lemmatized corpus:')
print(sl)
w_tokenized=[word_tokenize(i) for i in sl]
print("Tokenized Corpus:")
print(w_tokenized)
# Creating a Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names_out()
X_array = X.toarray()

print("Unique Word List: \n", feature_names)
print("Bag of Words Matrix: \n", X_array)
# Visualizing results in tabular form
import pandas as pd
print(pd.DataFrame(data=X_array, columns=feature_names, index=corpus))
# Calculating Product of Term Frequency & Inverse Document Frequency
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
terms = tfidf_vectorizer.get_feature_names_out()
print(pd.DataFrame(tfidf_matrix.toarray(), columns=terms))
from wordcloud import WordCloud
import matplotlib.pyplot as plt
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(vectorizer.vocabulary_)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')

print('tst')
import nltk
from sklearn.preprocessing import OneHotEncoder
import numpy as np
# Download NLTK data
#nltk.download('punkt')
all_words = [word for sentence in w_tokenized for word in sentence]
# Reshape the list of words into a 2D array for OneHotEncoder
word_array = np.array(all_words).reshape(-1, 1)
# Apply OneHotEncoder
one_hot_encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = one_hot_encoder.fit_transform(word_array)
# Print the one-hot encoded data
print("One-hot encoded matrix:\n", one_hot_encoded)
from gensim.models import Word2Vec
cbow_model = Word2Vec(w_tokenized, vector_size=100, window=5, min_count=1, sg=0, alpha=0.03, min_alpha=0.0007, epochs=100)
skipgram_model = Word2Vec(w_tokenized, vector_size=100, window=5, min_count=1, sg=1, alpha=0.03, min_alpha=0.0007, epochs=100)

cbow_model.train(w_tokenized, total_examples=len(w_tokenized), epochs=100)
skipgram_model.train(w_tokenized, total_examples=len(w_tokenized), epochs=100)

word_vectors_cbow = cbow_model.wv
similarity_cbow = word_vectors_cbow.similarity('nlp', 'language')
print(f"Similarity between 'nlp' and 'language': {similarity_cbow} with CBOW")
word_vectors_skipgram= skipgram_model.wv
similarity_skip = word_vectors_skipgram.similarity('nlp', 'language')
print(f"Similarity between 'nlp' and 'language': {similarity_skip} with Skip-Gram")
plt.show()