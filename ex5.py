import wikipedia, spacy
from typing import List, Set, Tuple, Dict
from collections import defaultdict
import numpy as np 
import google.generativeai as genai
import json 
import os


PROPN = "PROPN"
VERB = "VERB"
ADP = "ADP"

# Set your Palm API key
PALM_API_KEY = "AIzaSyAiFTOCes5aHyoBxuR6mNSpcuRw6oM2M0c"


# ----------- Section 1 UTILS ------------
def get_proper_noun_sequences(doc):
    """Find sequences of consecutive proper nouns in the document.
    Returns list of (start_idx, end_idx) tuples."""
    sequences = []
    current_sequence = []
    
    for i, token in enumerate(doc):
        if token.pos_ == PROPN:
            current_sequence.append(i)
        elif len(current_sequence) > 0: # if token.pos_ is not  "PROPN"
            sequences.append((current_sequence[0], current_sequence[-1] + 1))
            current_sequence = []
    if current_sequence:   #if the last word is a PropN
        sequences.append((current_sequence[0], current_sequence[-1] + 1))
    return sequences

def check_if_valid_pair(tokens_betweens):
    if len(tokens_betweens) == 0 :
        return False
    has_punct = any(token.pos_ == "PUNCT" for token in tokens_betweens)
    has_verb = any(token.pos_ == "VERB" for token in tokens_betweens)
    if (not has_punct) and has_verb:
        return True
    else:
        return False

def fetch_subject_relation_object(current_prop,next_prop,doc):
    subject = doc[current_prop[0]:current_prop[1]]
    object = doc[next_prop[0]:next_prop[1]]
    token_betweens = doc[current_prop[1]:next_prop[0]]
    relation = [token for token in token_betweens if token.pos_ in [VERB,ADP]]
    return subject, relation, object

def get_subject_relation_object_by_pos(doc):
    sequences_list = get_proper_noun_sequences(doc)
    relation = []
    for i in range(len(sequences_list)-1):
        current_prop, next_prop = sequences_list[i], sequences_list[i+1]
        token_betweens = doc[current_prop[1]:next_prop[0]]
        if check_if_valid_pair(token_betweens):
            relation.append(fetch_subject_relation_object(current_prop,next_prop,doc))
    return relation

# ----------- END of Section 1 UTILS ------------

def find_proper_noun_heads(doc):
    """
    Find all tokens that are proper nouns (PROPN) but don't have a compound dependency.
    These are the heads of proper noun phrases.
    """
    return [token for token in doc if token.pos_ == "PROPN" and token.dep_ != "compound"]

def get_proper_noun_set(head):
    """
    For a proper noun head, return the set containing it and all its compound children.
    Example: For "Smith" in "John Jerome Smith", returns {"John", "Jerome", "Smith"}
    """
    proper_noun_set = {head.text}
    for child in head.children:
        if child.dep_ == "compound" and child.pos_ == "PROPN":
            proper_noun_set.add(child.text)        
    return proper_noun_set

def meets_condition_1(head1, head2):
    """
    Check if two proper noun heads meet condition #1:
    - Same head token
    - First has nsubj dependency
    - Second has dobj dependency
    """
    if (head1.head == head2.head and
            head1.dep_ == "nsubj" and
            head2.dep_ == "dobj"):
        return head1.head
    return None

def meets_condition_2(head1: spacy.tokens.Token, head2: spacy.tokens.Token) -> bool:
    """
    Check if two proper noun heads meet condition #2:
    - h1's parent is h2's grandparent
    - h1 has nsubj dependency
    - h2's parent has prep dependency
    - h2 has pobj dependency
    """
    if head1.dep_ != "nsubj":
        return None

    h2_parent = head2.head   # Get h2's parent (h′)

    if head2.dep_ != "pobj": # Check if h2 is a prepositional object
        return None
        
    if h2_parent.dep_ != "prep": # Check if h2's parent is a preposition
        return None
    
    if head1.head == h2_parent.head:
        return {head1.head,head2.head}
    return None

def get_subject_relation_object_by_dep_tree(doc):
    relations = []
    prop_noun_head = find_proper_noun_heads(doc)
    for i in range(len(prop_noun_head)-1):
        head1 = prop_noun_head[i]
        head2 = prop_noun_head[i+1]
        head1_set, head2_set = get_proper_noun_set(head1),get_proper_noun_set(head2)
        relation_by_1 = meets_condition_1(head1, head2)
        relation_by_2 = meets_condition_2(head1, head2)
        if relation_by_1:
            relations.append((head1_set,relation_by_1,head2_set))
        elif relation_by_2:
            relations.append((head1_set,relation_by_2,head2_set))
    return relations

# ---------- Extractor using LLM -------------
def create_prompt(persona, text):
    prompt = f"""
    Given the following text about {persona}, extract subject-relation-object triplets.

    Text:
    {text}

    Output the triplets in JSON format, where each triplet is a dictionary with keys "subject", "relation", and "object". If no triplets are found, return an empty JSON array.

    Example:
    Text: Barack Obama was the 44th president of the United States. Michelle Obama is his wife.
    Output:
    [
        {{"subject": "Barack Obama", "relation": "was the 44th president of", "object": "United States"}},
        {{"subject": "Michelle Obama", "relation": "is wife of", "object": "Barack Obama"}}
    ]
    """
    return prompt
    
def extract_using_llm(page_dict):
    for persona in page_dict:
        print(F"-----{persona}------------")
        text = page_dict[persona]
        prompt = create_prompt(persona,text)

        genai.configure(api_key=PALM_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        json_data = response.text

        # Specify the path to the file where you want to save the JSON data
        file_path = f'{persona}.json'
        # Write the JSON string to the file
        with open(file_path, 'w') as file:
            file.write(json_data)
        



if __name__ == "__main__":

    page_dict = {}
    page_dict['Donald Trump']  = dict()
    page_dict['Ruth Bader Ginsburg'] = dict()
    page_dict['J. K. Rowling'] = dict()
    for persona in page_dict:
        print(f"----- {persona}-------")
        search_result = wikipedia.search(persona)
        page_dict[persona]["wikipedia_page"] = wikipedia.page(search_result[0],auto_suggest=False).content
        page_dict[persona]["nlp_doc"] = nlp(page_dict[persona]["wikipedia_page"])
        page_dict[persona]["relation_by_pos"] = get_subject_relation_object_by_pos(page_dict[persona]["nlp_doc"])
        print(f"Number of Triplets by Pos - {len(page_dict[persona]['relation_by_pos'])}")
        page_dict[persona]["relation_by_tree"] = get_subject_relation_object_by_dep_tree(page_dict[persona]["nlp_doc"])
        print(f"Number of Triplets by Tree - {len(page_dict[persona]['relation_by_tree'])}")
    
    samples_pos = []
    for persona in page_dict:
        random_idx = np.random.randint(1, len(page_dict[persona]["relation_by_pos"]),5)
        for i in random_idx:
            samples_pos.append(page_dict[persona]["relation_by_pos"][i])
    samples_tree = []
    for persona in page_dict:
        random_idx = np.random.randint(1, len(page_dict[persona]["relation_by_tree"]),5)
        for i in random_idx:
            samples_tree.append(page_dict[persona]["relation_by_tree"][i])
    
    extract_using_llm(page_dict)