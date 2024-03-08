import json
import pandas as pd
import re
import os




def clean_text(text):
    """
    Nettoie le texte des caractères spéciaux sauf les lettres avec accents et les chiffres
    """
    # Remplace les caractères spéciaux (sauf les lettres avec accents et les chiffres) par un espace
    cleaned_text = re.sub(r'[^\w\sàáâäèéêëìíîïòóôöùúûüçÈÉÊËÌÍÎÏÒÓÔÖÙÚÛÜÇ0-9]', '', text)
    return cleaned_text


def read_qa_data(json_file_path):
    """
    Read Q&A data from JSON file
    """
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    questions = 0
    answers = 0
    qa_list= []
    for article, qa_pairs in data.items():
        content = ""
        # print(qa_pairs)
        for j in range(len(qa_pairs)):
            # Join the partial content of the article
            qa_pair=qa_pairs[j]
            content += qa_pair[0] + "\n"
            
            # Extract the suggested question and answer
           
            for qa_item in qa_pair[1:]:
                print(qa_item)
                parts = re.split(r'question:|response:', qa_item, flags=re.IGNORECASE)
                if "question" in qa_item:
                #if len(parts) >= 3:
                    question = clean_text(parts[1].strip())
                else:
                    answer = clean_text(parts[1].strip())
            

            qa_list.append([question,answer,article])
                
                    
    
    df = pd.DataFrame(qa_list, columns=['question', 'answer', 'num_article'])
    df['question'] = df['question'].str.lower()
    df['answer'] = df['answer'].str.lower()

    
    return df

json_file_path="Dataset Q_R/articles.json"

df=read_qa_data(json_file_path)

output_dir = os.path.dirname(json_file_path)
output_csv_path = os.path.join(output_dir, 'qa_data.csv')
df.to_csv(output_csv_path, index=False)