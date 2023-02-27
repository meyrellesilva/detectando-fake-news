import nltk
import pandas as pd
import numpy as np
import re

# Carrega o conjunto de dados de notícias
df = pd.read_csv('noticias.csv')

# Pré-processamento de texto
def preprocess_text(text):
    # Remove pontuações e caracteres especiais
    text = re.sub(r'[^\w\s]', '', text)
    # Converte para letras minúsculas
    text = text.lower()
    # Remove stopwords
    stop_words = set(nltk.corpus.stopwords.words('portuguese'))
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    # Realiza a lematização
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

# Cria um conjunto de dados de treinamento
df['processed_text'] = df['text'].apply(preprocess_text)
train_data = list(zip(df['processed_text'], df['label']))

# Treina o modelo de classificação de notícias
classifier = nltk.NaiveBayesClassifier.train(train_data)

# Função que analisa a veracidade de uma notícia
def check_news(news):
    # Pré-processamento do texto da notícia
    processed_news = preprocess_text(news)
    # Usa o modelo treinado para determinar a classificação da notícia
    result = classifier.classify(processed_news)
    return result

# Exemplo de uso
news = "Cientistas descobrem cura para o câncer"
classification = check_news(news)
if classification == 'fake':
    print("Essa notícia é falsa!")
else:
    print("Essa notícia é verdadeira.")
