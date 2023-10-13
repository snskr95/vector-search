from triples_data import *
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
from keytotext import pipeline

# model_name = "gpt2"  # You can choose other variants of GPT-2
# model = GPT2LMHeadModel.from_pretrained(model_name)
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)


text_embedding_list =[

]


def get_sentence(triple):
    subj, pred, obj = triple
    triples_text = f"{subj} {pred} {obj} ."
    # input_ids = tokenizer.encode(triples_text, return_tensors="pt", max_length=512, truncation=True)
    #
    # output_ids = model.generate(input_ids, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50,
    #                             top_p=0.95)
    # generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return triples_text



def get_embedding(text):
    model_2 = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Sentences we want to encode. Example:
    sentence = []
    sentence.append(text)

    # Sentences are encoded by calling model.encode()
    embedding = model_2.encode(sentence)
    return embedding[0]


def generate_text_embeddings(docs):

    for doc in docs:
        print(doc["id"])
        for triple in set(doc['triplePaths']):
            triple_array = [triple.split("|")[0], triple.split("|")[1], triple.split("|")[2]]
            gen_text = get_sentence((triple.split("|")[0], triple.split("|")[1], triple.split("|")[2]))
            text_embedding_list.append({
                "words":triple_array,
                "embedding": get_embedding(gen_text),
                "entity": doc
            })

    return text_embedding_list


def get_cosine_similarity(v1, v2):
    word1_embedding = np.array(v1)
    word2_embedding = np.array(v2)
    # Reshape the arrays to match the expected input shape of cosine_similarity
    word1_embedding = word1_embedding.reshape(1, -1)
    word2_embedding = word2_embedding.reshape(1, -1)
    # Calculate cosine similarity
    similarity = cosine_similarity(word1_embedding, word2_embedding)[0][0]
    return similarity

def get_most_relevant_doc(query):
    query_embedding = get_embedding(query)
    new_list = sorted(text_embedding_list, key=lambda x: get_cosine_similarity(x['embedding'], query_embedding), reverse=True)


    # Load the base pre-trained T5 model
    # It will download three files: 1. config.json, 2. tokenizer.json, 3. pytorch_model.bin (~850 MB)
    nlp = pipeline("k2t-base")

    # Configure the model parameters
    config = {"do_sample": True, "num_beams": 4, "no_repeat_ngram_size": 3, "early_stopping": True}

    # Provide list of keywords into the model as input
    recommendations = {}

    recommendations['most_relevant'] =  {"result": nlp(new_list[0]["words"], **config),
     "entity":new_list[0]["entity"]}


    recommendations['recommended'] =  list(map(lambda x: {"result": nlp(x['words'], **config), "entity": x["entity"] }, new_list[1:6]))
    return recommendations


    # return new_list[0]["words"]






