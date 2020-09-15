#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 16:12:20 2020

@author: daniel
"""

import sys
sys.path.append("..")
from common.download_utils import download_week3_resources

#download_week3_resources()

#Library that allow to use embeddings
import gensim

#wv_embedding is the embedding loaded
wv_embeddings = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=500000) ######### YOUR CODE HERE #############
type(wv_embeddings )

#print("Vector for a common word that is in the vocabulary: ", wv_embeddings.get_vector("motorcycle"))
#print("Vector for a common word that is NOT in the vocabulary: ", wv_embeddings.get_vector("home34"))
#print("Vector of a word", wv_embeddings['king'])
#print("Validate if a word exists", 'breakfast' in wv_embeddings)

def check_embeddings(embeddings):
    error_text = "Something wrong with your embeddings ('%s test isn't correct)."
    
    #Example of use of the method most_similar of the model
    #most_similar = embeddings.most_similar(positive=['woman', 'king'], negative=['man'])
    #most_similar = embeddings.most_similar(positive=['car', 'truck', 'minivan'], negative=['motorcycle', 'bicycle'])
    #print(most_similar)
    
#    if len(most_similar) < 1 or most_similar[0][0] != 'queen':
#        return error_text % "Most similar"

    #Example of use of the method doesnt_match of the model
#    doesnt_match = embeddings.doesnt_match(['stone', 'cereal', 'dinner', 'lunch'])
#    print(doesnt_match)
##    if doesnt_match != 'cereal':
##        return error_text % "Doesn't match"
#    
#    #Example of use of the method most_similar_to_given of the model
    most_similar_to_given = embeddings.most_similar_to_given('music', ['water', 'sound', 'backpack', 'mouse'])
    print(most_similar_to_given)
#    if most_similar_to_given != 'sound':
#        return error_text % "Most similar to given"
#    
#    return "These embeddings look good."

print(check_embeddings(wv_embeddings))

#numpy is a library that allow to use arrays and matrix
#This function converts a question in a vector
import numpy as np
def question_to_vec(question, embeddings, dim=300):
    """
        question: a string
        embeddings: dict where the key is a word and a value is its' embedding
        dim: size of the representation
        result: vector representation for the question
    """
    ######################################
    ######### YOUR CODE HERE #############
    ######################################
    
    result = np.zeros(dim) #Vector de 300 dimensiones para el vector de la frase
    cnt = 0
    words = question.split()
    for word in words: #All word vectors composing the phrase are summed
        if word in embeddings:
            result += np.array(embeddings[word])
            cnt += 1
    if cnt != 0: #This would happen if no word was found in the embedding
        result /= cnt
    return result

def question_to_vec_tests():
    if (np.zeros(300) != question_to_vec('', wv_embeddings)).any():
        return "You need to return zero vector for empty question."
    if (np.zeros(300) != question_to_vec('thereisnosuchword', wv_embeddings)).any():
        return "You need to return zero vector for the question, which consists only unknown words."    
    if (wv_embeddings['word'] != question_to_vec('word', wv_embeddings)).any():
        print(question_to_vec('word', wv_embeddings))        
        return "You need to check the corectness of your function."
    if ((wv_embeddings['I'] + wv_embeddings['am']) / 2 != question_to_vec('I am', wv_embeddings)).any():
        return "Your function should calculate a mean of word vectors."
    if (wv_embeddings['word'] != question_to_vec('thereisnosuchword word', wv_embeddings)).any():
        return "You should not consider words which embeddings are unknown."
    return "Basic tests are passed."

print(question_to_vec_tests())

import nltk
nltk.download('stopwords')  #el, la, los, un, unos, sobre, para
from util import array_to_string

#question2vec_result = []
#for question in open('data/test_embeddings.tsv'):
#    question = question.strip()
#    answer = question_to_vec(question, wv_embeddings)
#    question2vec_result = np.append(question2vec_result, answer)
#
#grader.submit_tag('Question2Vec', array_to_string(question2vec_result))

'''
Metric 1 to validate the precision of a model
'''
def hits_count(dup_ranks, k):
    """
        dup_ranks: list of duplicates' ranks; one rank per question;
                   length is a number of questions which we are looking for duplicates;
                   rank is a number from 1 to len(candidates of the question); 
                   e.g. [2, 3] means that the first duplicate has the rank 2, the second one — 3.
        k: number of top-ranked elements (k in Hits@k metric)

        result: return Hits@k value for current ranking
    """
    ######################################
    ######### YOUR CODE HERE #############
    ######################################
    
    return np.average(np.array(dup_ranks) <= np.array([k]))

def test_hits():
    # *Evaluation example*
    # answers — dup_i
    answers = ["How does the catch keyword determine the type of exception that was thrown"]
    
    # candidates_ranking — the ranked sentences provided by our model
    candidates_ranking = [["How Can I Make These Links Rotate in PHP", 
                           "How does the catch keyword determine the type of exception that was thrown",
                           "NSLog array description not memory address",
                           "PECL_HTTP not recognised php ubuntu"]]
    # dup_ranks — position of the dup_i in the list of ranks +1
    dup_ranks = [candidates_ranking[i].index(answers[i]) + 1 for i in range(len(answers))]
    
    # correct_answers — the expected values of the result for each k from 1 to 4
    correct_answers = [0, 1, 1, 1]
    for k, correct in enumerate(correct_answers, 1):
        if not np.isclose(hits_count(dup_ranks, k), correct):
            return "Check the function."
    
    # Other tests
    answers = ["How does the catch keyword determine the type of exception that was thrown", 
               "Convert Google results object (pure js) to Python object"]
    
    # The first test: both duplicates on the first position in ranked list
    candidates_ranking = [["How does the catch keyword determine the type of exception that was thrown",
                           "How Can I Make These Links Rotate in PHP"], 
                          ["Convert Google results object (pure js) to Python object",
                           "WPF- How to update the changes in list item of a list"]]
    dup_ranks = [candidates_ranking[i].index(answers[i]) + 1 for i in range(len(answers))]
    correct_answers = [1, 1]
    for k, correct in enumerate(correct_answers, 1):
        if not np.isclose(hits_count(dup_ranks, k), correct):
            return "Check the function (test: both duplicates on the first position in ranked list)."
        
    # The second test: one candidate on the first position, another — on the second
    candidates_ranking = [["How Can I Make These Links Rotate in PHP", 
                           "How does the catch keyword determine the type of exception that was thrown"], 
                          ["Convert Google results object (pure js) to Python object",
                           "WPF- How to update the changes in list item of a list"]]
    dup_ranks = [candidates_ranking[i].index(answers[i]) + 1 for i in range(len(answers))]
    correct_answers = [0.5, 1]
    for k, correct in enumerate(correct_answers, 1):
        if not np.isclose(hits_count(dup_ranks, k), correct):
            return "Check the function (test: one candidate on the first position, another — on the second)."

    # The third test: both candidates on the second position
    candidates_ranking = [["How Can I Make These Links Rotate in PHP", 
                           "How does the catch keyword determine the type of exception that was thrown"], 
                          ["WPF- How to update the changes in list item of a list",
                           "Convert Google results object (pure js) to Python object"]]
    dup_ranks = [candidates_ranking[i].index(answers[i]) + 1 for i in range(len(answers))]
    correct_answers = [0, 1]
    for k, correct in enumerate(correct_answers, 1):
        if not np.isclose(hits_count(dup_ranks, k), correct):
            return "Check the function (test: both candidates on the second position)."

    return "Basic test are passed."

print(test_hits())

'''
Metric 2 to validate the precision of a mode
'''
def dcg_score(dup_ranks, k):
    """
        dup_ranks: list of duplicates' ranks; one rank per question; 
                   length is a number of questions which we are looking for duplicates; 
                   rank is a number from 1 to len(candidates of the question); 
                   e.g. [2, 3] means that the first duplicate has the rank 2, the second one — 3.
        k: number of top-ranked elements (k in DCG@k metric)

        result: return DCG@k value for current ranking
    """
    ######################################
    ######### YOUR CODE HERE #############
    ######################################
    return np.average((np.array(dup_ranks) <= np.array([k]))*1./(np.log2(1. + np.array(dup_ranks))))

def test_dcg():
    # *Evaluation example*
    # answers — dup_i
    answers = ["How does the catch keyword determine the type of exception that was thrown"]
    
    # candidates_ranking — the ranked sentences provided by our model
    candidates_ranking = [["How Can I Make These Links Rotate in PHP", 
                           "How does the catch keyword determine the type of exception that was thrown",
                           "NSLog array description not memory address",
                           "PECL_HTTP not recognised php ubuntu"]]
    # dup_ranks — position of the dup_i in the list of ranks +1
    dup_ranks = [candidates_ranking[i].index(answers[i]) + 1 for i in range(len(answers))]
    
    # correct_answers — the expected values of the result for each k from 1 to 4
    correct_answers = [0, 1 / (np.log2(3)), 1 / (np.log2(3)), 1 / (np.log2(3))]
    for k, correct in enumerate(correct_answers, 1):
        if not np.isclose(dcg_score(dup_ranks, k), correct):
            return "Check the function."
    
    # Other tests
    answers = ["How does the catch keyword determine the type of exception that was thrown", 
               "Convert Google results object (pure js) to Python object"]

    # The first test: both duplicates on the first position in ranked list
    candidates_ranking = [["How does the catch keyword determine the type of exception that was thrown",
                           "How Can I Make These Links Rotate in PHP"], 
                          ["Convert Google results object (pure js) to Python object",
                           "WPF- How to update the changes in list item of a list"]]
    dup_ranks = [candidates_ranking[i].index(answers[i]) + 1 for i in range(len(answers))]
    correct_answers = [1, 1]
    for k, correct in enumerate(correct_answers, 1):
        if not np.isclose(dcg_score(dup_ranks, k), correct):
            return "Check the function (test: both duplicates on the first position in ranked list)."
        
    # The second test: one candidate on the first position, another — on the second
    candidates_ranking = [["How Can I Make These Links Rotate in PHP", 
                           "How does the catch keyword determine the type of exception that was thrown"], 
                          ["Convert Google results object (pure js) to Python object",
                           "WPF- How to update the changes in list item of a list"]]
    dup_ranks = [candidates_ranking[i].index(answers[i]) + 1 for i in range(len(answers))]
    correct_answers = [0.5, (1 + (1 / (np.log2(3)))) / 2]
    for k, correct in enumerate(correct_answers, 1):
        if not np.isclose(dcg_score(dup_ranks, k), correct):
            return "Check the function (test: one candidate on the first position, another — on the second)."
        
    # The third test: both candidates on the second position
    candidates_ranking = [["How Can I Make These Links Rotate in PHP",
                           "How does the catch keyword determine the type of exception that was thrown"], 
                          ["WPF- How to update the changes in list item of a list",
                           "Convert Google results object (pure js) to Python object"]]
    dup_ranks = [candidates_ranking[i].index(answers[i]) + 1 for i in range(len(answers))]
    correct_answers = [0, 1 / (np.log2(3))]
    for k, correct in enumerate(correct_answers, 1):
        if not np.isclose(dcg_score(dup_ranks, k), correct):
            return "Check the function (test: both candidates on the second position)."

    return "Basic test are passed."

print(test_dcg())

'''
Files:
-train corpus contains similar sentences at the same row.
-validation corpus contains the following columns: question, similar question, negative example 1, negative example 2, ...
-test corpus contains the following columns: question, example 1, example 2, 
Upload of the Validation corpus to evaluate the current solution
'''


def read_corpus(filename):
    data = []
    for line in open(filename, encoding='utf-8'):
        data.append(line.strip().split('\t'))
    return data

validation = read_corpus('data/validation.tsv') ######### YOUR CODE HERE #############
#print(validation)


#Use of cosine_similarity to calculte the proximity between a question and a list of candidates
from sklearn.metrics.pairwise import cosine_similarity

'''
Explanation of composition of the response
sim emulates a cosine similariry list
Qs emulates a candidates list
'''
sim = [0.2, .99, -.3]
Qs = ['Q1', 'Q2', 'Q3']
merged_list=list(zip(sim, range(3), Qs))
print(merged_list)
sorted_list  = sorted(merged_list, key=lambda x: x[0], reverse=True) #Descendend order
print(sorted_list)
result = [(b,c) for a,b,c in sorted_list]
print(result)


'''
This function receives a question, a list of candidates, an embedding and a dimension.
Return a list of pairs with the calculated cosine distances for each candidate in desdencent order.
Most similar candidates are shown first.
Results are represented as following:
(initial position in candidates list, candidate)
'''
def rank_candidates(question, candidates, embeddings, dim=300):
    """
        question: a string
        candidates: a list of strings (candidates) which we want to rank
        embeddings: some embeddings
        dim: dimension of the current embeddings
        
        result: a list of pairs (initial position in the list, question)
    """
    ######################################
    ######### YOUR CODE HERE #############
    ######################################
    
    #Vectorize the question
    q_vecs = np.array([question_to_vec(question, embeddings, dim) for i in range(len(candidates))])
    
    #Vectorize each of the candidates
    cand_vecs = np.array([question_to_vec(candidate, embeddings, dim) for candidate in candidates])
    
    #Here is where the magic happens, the cosine similarity between a question and all candidates is calculated
    cosines = np.array(cosine_similarity(q_vecs, cand_vecs)[0])
    
    merged_list = list(zip(cosines, range(len(candidates)), candidates))
    #print(merged_list)
    sorted_list  = sorted(merged_list, key=lambda x: x[0], reverse=True)
    result = [(b,c) for a,b,c in sorted_list]
    
    return result

#def test_rank_candidates():
#    questionTweets = []
#    candidatesTweets = []
#    f = open("pruebaTweetsSimilares.tsv","+w",encoding="utf-8")
#    cases = open("week3\\data\\testTweets.tsv", encoding='utf-8').readlines()
#    fileSize = len(cases)
#    i=0
#    while i < fileSize:
#        try:
#            questionTweets.append(cases[i])
#            temp = []
#            for j in range(1,4):
#                i += j
#                temp.append(cases[i]) 
#            candidatesTweets.append(temp)
#        except:
#            print("hola")
#
#    questions = ['converting string to list', 'Sending array via Ajax fails']
#    candidates = [['Convert Google results object (pure js) to Python object', 
#                   'C# create cookie from string and send it',
#                   'How to use jQuery AJAX for an outside domain?'], 
#                  ['Getting all list items of an unordered list in PHP', 
#                   'WPF- How to update the changes in list item of a list', 
#                   'select2 not displaying search results']]
#    results = [[(1, 'C# create cookie from string and send it'), 
#                (0, 'Convert Google results object (pure js) to Python object'), 
#                (2, 'How to use jQuery AJAX for an outside domain?')],
#               [(0, 'Getting all list items of an unordered list in PHP'), 
#                (2, 'select2 not displaying search results'), 
#                (1, 'WPF- How to update the changes in list item of a list')]]
#    ranks = []
#    for i in range(len(questionTweets)):
#        try:
#            f.write('tweet:'+'\n')
#            f.write(questionTweets[i])
#            f.write('tweets similares:'+'\n')
#            rank = rank_candidates(questionTweets[i],candidatesTweets[i],results)
#            for r in rank:
#                f.write(r[1])
#                f.write('\n')
#        except:
#            print("hola2")
#        
#    #for question, q_candidates, result in zip(questionTweets, candidatesTweets, results):
#        #ranks = rank_candidates(question, q_candidates, wv_embeddings, 300)
#        #print(ranks)
#    
#    #for i in range(len(questionTweets)):
#     #   f.write("question:")
#      #  f.write(questionTweets[i])
#       # for c in ranks[i]:
#        #    for e in c:
#         #       f.write("-------------------------------------------------------")
#          #      f.write(e)
#            
#        #if not np.all(ranks == result):
#            
#    return "Basic tests are passed."

'''
This code invokes the function rank_candidates, sending 1 question, 1 list of candidates 
and 1 list with the right results.
In all cases, 300 is the dimension of the embedding. 
The function validates if the result is right and return a message.
'''    
    
def test_rank_candidates():
    questions = ['converting string to list', 'Sending array via Ajax fails']
    candidates = [['Convert Google results object (pure js) to Python object', 
                   'C# create cookie from string and send it',
                   'How to use jQuery AJAX for an outside domain?'], 
                  ['Getting all list items of an unordered list in PHP', 
                   'WPF- How to update the changes in list item of a list', 
                   'select2 not displaying search results']]
    results = [[(1, 'C# create cookie from string and send it'), 
                (0, 'Convert Google results object (pure js) to Python object'), 
                (2, 'How to use jQuery AJAX for an outside domain?')],
               [(0, 'Getting all list items of an unordered list in PHP'), 
                (2, 'select2 not displaying search results'), 
                (1, 'WPF- How to update the changes in list item of a list')]]
    for question, q_candidates, result in zip(questions, candidates, results):        
        ranks = rank_candidates(question, q_candidates, wv_embeddings, 300)
        print(ranks)
        if not np.all(ranks == result):
            return "Check the function."
    return "Basic tests are passed."

print(test_rank_candidates())

'''
Each line of validation contains the following columns: question, similar question, negative example 1, negative example 2, ...
'''

wv_ranking = []
for line in validation:
    q, *ex = line
    ranks = rank_candidates(q, ex, wv_embeddings)
    wv_ranking.append([r[0] for r in ranks].index(0) + 1)
    
for k in [1, 5, 10, 100, 500, 1000]:
    print("DCG@%4d: %.3f | Hits@%4d: %.3f" % (k, dcg_score(wv_ranking, k), k, hits_count(wv_ranking, k)))

#Here    
for line in validation[:3]:
    q, *examples = line
    print(q, *examples[:3])
    
'''
text_prepare is the  function that clean up the text before the model processes it
'''
from util import text_prepare
prepared_validation = []
for line in validation:
    prepared_validation.append([text_prepare(sentence) for sentence in line])
    ######### YOUR CODE HERE #############

wv_prepared_ranking = []
for line in prepared_validation:
    q, *ex = line
    ranks = rank_candidates(q, ex, wv_embeddings)
    wv_prepared_ranking.append([r[0] for r in ranks].index(0) + 1)
    
for k in [1, 5, 10, 100, 500, 1000]:
    print("DCG@%4d: %.3f | Hits@%4d: %.3f" % (k, dcg_score(wv_prepared_ranking, k), 
                                              k, hits_count(wv_prepared_ranking, k)))
#here


def prepare_file(in_, out_):
    out = open(out_, 'w')
    for line in open(in_, encoding='utf8'):
        line = line.strip().split('\t')
        new_line = [text_prepare(q) for q in line]
        print(*new_line, sep='\t', file=out)
    out.close()

######################################
######### YOUR CODE HERE #############
######################################
    
'''
Here the Train and Test files are preprocessed
'''    
prepare_file('data/train.tsv', 'data/prepared_train.tsv')
prepare_file('data/test.tsv', 'data/prepared_test.tsv')


###### STARSPACE SOLUTION #######
'''
The main difference between word2vec and StarSpace is that word2vec is a model
with a word-per-word training. Instead, StarSpace is a model which consider 
the full sentence for trainning.
-word2vec: Embedding for each word
-starspace: Embedding for the full sentence. Use similar sentence pairs.

Starspace uses:
    Training dataset: Contains positive samples.
    Negative samples are generated randomly.
'''


######### TRAINING HAPPENING HERE #############
#Here the tunning may occur!
#!starspace train -trainFile "data/prepared_train.tsv" -model starspace_embedding \
#-trainMode 3 -adagrad true -ngrams 1 -epoch 5 -dim 100 -similarity cosine -minCount 2 \
#-verbose true -fileFormat labelDoc -negSearchLimit 10 -lr 0.05

#Here I am reading the starspace file and I am building the dictionary starspace_embeddings 
#The dictionary contains the embedding of each word. It has 95058 words
starspace_embeddings = dict()
for line in open('starspace_embedding.tsv', encoding='utf-8'):
    row = line.strip().split('\t')
    starspace_embeddings[row[0]] = np.array(row[1:], dtype=np.float32)

#Here I am using my new embedding (starspace_embeddings) to identify similarities
#between stackoverflow questions
ss_prepared_ranking = []
for line in prepared_validation:
    q, *ex = line
    ranks = rank_candidates(q, ex, starspace_embeddings, 100)
    ss_prepared_ranking.append([r[0] for r in ranks].index(0) + 1)

for k in [1, 5, 10, 100, 500, 1000]:
    print("DCG@%4d: %.3f | Hits@%4d: %.3f" % (k, dcg_score(ss_prepared_ranking, k), 
                                               k, hits_count(ss_prepared_ranking, k)))

