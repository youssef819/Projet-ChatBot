import pandas as pd
import numpy as np

from models.retrieve.similarities import cosine_sim, euclidian_sim
from models.retrieve.sent2vec import sent2vec
from src.preprocess import normalizeString


def answer_question(question, s2v_model, w2v_model, data, similarity, k=1):
    """
    Uses vector representation and similarity function 
    to provide K possible answers to the question
    """
    # Clean question
    question = normalizeString(question)
    embed_matrix = s2v_model.get_embedding_matrix()
    query_vec = s2v_model.seq_vec_sent(question)
    result = ''

    if query_vec.shape[0] == embed_matrix.shape[0]:
        
        # Compute similarity
        X = similarity(query_vec, embed_matrix)[0]

        # Get best matchs indexes'
        indexes = np.argsort(X)

        # Extract responses from data
        Y = data.iloc[indexes, 1].drop_duplicates()
        Y_indexes = list(Y.index)
        responses = Y.to_numpy()[-k:]

        for i, rep in enumerate(responses):
            result += "Similarity : {} - {}\n".format(float(X[Y_indexes[-k+i]]), rep)
        
    else:
        resilt += 'seqvec embedding dimensions {} \n'\
                  'model embedding dimensions {}'.format(query_vec.shape, embed_questions.shape)

    return result


if __name__ == '__main__':
    
    ### Read data
    path = "QA_data_deleted.csv"
    data = pd.read_csv(path, encoding="UTF-8", header=None, names=["Question","Answer"]) 
    data["Question"] = data["Question"].apply(lambda x: normalizeString(x) if isinstance(x, str) else x)
    data["Answer"] = data["Answer"].apply(lambda x: normalizeString(x) if isinstance(x, str) else x) 
    
    ### Question
    s2v = sent2vec(data, 'cooc_2')
    w2v = s2v.get_model()

    ### Answer question
    input_question = ""
    while input_question not in ['quit', 'q']: 
        input_question = input(">>>")
        answers = answer_question(input_question, s2v, w2v, data, cosine_sim, 10)
        print(answers)