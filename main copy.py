import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import numpy as np
from fastapi import FastAPI

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")


# Preprocesamiento de texto
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word.isalnum() and word not in stopwords.words("spanish")
    ]
    return " ".join(words)


# Función para codificar texto
def encode_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy().squeeze()


# Conexión a la base de datos
client = MongoClient(
    "mongodb+srv://germys:LWBVI45dp8jAIywv@douvery.0oma0vw.mongodb.net/Production"
)
db = client["Production"]
products_collection = db["products"]

# Obtener los productos
products = products_collection.find({})
df = pd.DataFrame(list(products))

# Preprocesar el nombre y la descripción de los productos
df["name"] = df["name"].apply(preprocess_text)
df["description"] = df["description"].apply(preprocess_text)

# Combina el nombre y la descripción en una sola columna
df["text"] = df["name"] + " " + df["description"]

# Codificar los textos de los productos
df["text_encoded"] = df["text"].apply(encode_text)

# Crear una matriz de similitud del coseno
similarity_matrix = cosine_similarity(np.vstack(df["text_encoded"]))


# Función para obtener productos similares
def get_similar_products(query):
    query_encoded = encode_text(query)
    similarity_scores = cosine_similarity(
        [query_encoded], np.vstack(df["text_encoded"])
    )[0]
    similar_indices = similarity_scores.argsort()[:-5:-1]
    similar_items = [(similarity_scores[i], df["name"][i]) for i in similar_indices]
    return similar_items


# Buscar productos similares
query = "lapto de ap"
print(get_similar_products(query))

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello"}


@app.get("/get_similar_products/{query}")
async def get_similar(query: str):
    products = get_similar_products(query)
    return {"similar_products": products}
