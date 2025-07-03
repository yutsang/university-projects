"""
topic_modeling.py
Topic modeling and word cloud utilities for tweet sentiment analysis pipeline.
Requires pandas, gensim, wordcloud, and matplotlib to be installed in your environment.
"""

import pandas as pd
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from typing import List

def preprocess_text(text: str) -> List[str]:
    """Tokenize and remove stopwords for topic modeling."""
    return [token for token in simple_preprocess(text) if token not in STOPWORDS]

def lda_topic_modeling(df: pd.DataFrame, text_col: str = "text", num_topics: int = 10):
    """
    Perform LDA topic modeling and print topics.
    Args:
        df: DataFrame with tweet text.
        text_col: Name of the column containing tweet text.
        num_topics: Number of topics for LDA.
    Returns:
        lda_model, dictionary, corpus
    """
    processed = df[text_col].apply(preprocess_text)
    dictionary = Dictionary(processed)
    corpus = [dictionary.doc2bow(doc) for doc in processed]
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
    for topic in lda_model.show_topics(num_topics=num_topics):
        print(topic)
    coherence_model_lda = CoherenceModel(model=lda_model, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    print(f"Coherence score: {coherence_model_lda.get_coherence():.4f}")
    print(f"Perplexity score: {lda_model.log_perplexity(corpus):.4f}")
    return lda_model, dictionary, corpus

def generate_wordcloud(df: pd.DataFrame, text_col: str = "text", output_path: str = None):
    """
    Generate and display a word cloud from tweet text.
    Args:
        df: DataFrame with tweet text.
        text_col: Name of the column containing tweet text.
        output_path: If provided, save the word cloud image to this path.
    """
    text = " ".join(df[text_col].tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='magma', max_words=100).generate(text)
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    if output_path:
        plt.savefig(output_path)
    plt.show() 