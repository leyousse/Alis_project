from flask import Flask,request, jsonify
import json
from bdd import insert_phrase
#import sys
#sys.path.insert(1, '/ml')
from models import generate_sentences_english_gpt2,generate_sentences_french_gpt2,question_answer

app = Flask(__name__)
app.config["DEBUG"] = True

##liste = generate_sentences_english_gpt2('bonjour',num_return_sequences=2)
@app.route('/generate_sentences_english_gpt2',methods=['POST'])
def finish_sentences_english_gpt2():
    req_data = request.get_json()
    
    debut_phrase = req_data["phrase"]
    print("debut_phrase",debut_phrase,type(debut_phrase))
    
    liste_reponse = generate_sentences_english_gpt2(debut_phrase,num_return_sequences=1)
    json_string = json.dumps(liste_reponse)
    print(json_string)

    return json_string

@app.route('/generate_sentences_french_gpt2',methods=['POST'])
def finish_sentences_french_gpt2():
    req_data = request.get_json()
    
    debut_phrase = req_data["phrase"]
    print("debut_phrase",debut_phrase,type(debut_phrase))
    
    liste_reponse = generate_sentences_french_gpt2(debut_phrase,num_return_sequences=1)
    json_string = json.dumps(liste_reponse)
    print(json_string)

    return json_string

@app.route('/insert_to_dabatase',methods=['POST'])
def insert_to_dabatase():
    req_data = request.get_json()
    
    phrase = req_data["phrase"]
    print("QUESTIONN",phrase,type(phrase))

    reponse = req_data["reponse"]
    interlocuteur = req_data["interlocuteur"]
    
    print(interlocuteur,phrase,reponse)
    status = insert_phrase(phrase,reponse,interlocuteur)
    if status > 0:
        return "ça a marché"
    else:
        return "problème !!!"
        
@app.route('/questions_reponses',methods=['POST'])
def question_reponses():
    req_data = request.get_json()
    
    question = req_data["phrase"]
    print("QUESTIONN",question,type(question))
    liste_reponse = question_answer(question)
    print(liste_reponse)
    
    #liste_reponse = ["reponse1","reponse2"]
    json_string = json.dumps(liste_reponse)
    print(json_string)

    return json_string


app.run()