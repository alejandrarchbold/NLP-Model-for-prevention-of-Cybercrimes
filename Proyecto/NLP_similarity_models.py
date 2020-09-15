#!/usr/bin/env python
# coding: utf-8

# # Detecting cybercrimes using similarity models
# 

# ## OS

# In[1]:


import os # The OS module in Python provides a way of using operating system dependent functionality
import pandas as pd # Read data from file


# In[2]:


os.listdir("C:/Users/Alejandra/Desktop/NLP-Model-for-prevention-of-Cybercrimes/Proyecto") # Reads and shows files in entered location


# ## Data

# In[3]:


data = pd.read_excel("blm_.xlsx",sheet_name="Archive") # Importing the database (tweets)


# In[4]:


data.head() # Returns the first rows of the database


# In[5]:


# converting to list the column "text" where it contains the tweets
cols = ['from_user', 'text']
text = data[cols].values.tolist()


# In[6]:


print(text) # printing the tweets


# In[7]:


text[0][0] # accessing to the first tweet from the list 'text'


# In[8]:


len(text) # number of tweets


# In[9]:


tweets = [text[i][1] for i in range(len(text))]
tweets
len(tweets)


# In[10]:


users = [text[i][0] for i in range(len(text))]
users
len(users)


# ## Data Preparation
# 
# - ## Functions to remove URL links, @mention, #hashtags

# In[11]:


import string # String contains methods that allow the use of characters which are considered punctuation characters, 
                # digits, uppercase, lowercase, etc.
import re # Regular expression operations


# In[12]:


def strip_links(text): # function to remove/strip URL links
    link_regular_expression = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links = re.findall(link_regular_expression, text)
    for link in links:
        text = text.replace(link[0], ', ')   
    return text


# In[13]:


def strip_all_entities(text): # function to remove/strip mentions, hashtags, characters from some users
    entity_prefixes = ['@','#','\\','_']
    for separator in  string.punctuation:
        if separator not in entity_prefixes :
            text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)


# source: https://stackoverflow.com/questions/8376691/how-to-remove-hashtag-user-link-of-a-tweet-using-regular-expression

# In[14]:


def strip_all(list_text): # function that removes all URL links, @mention, #hashtags
    list_stripped = []
    for t in list_text:
        word = strip_all_entities(strip_links(t))
        list_stripped.append(word)
    return(list_stripped)


# In[15]:


tweets = strip_all([text[i][1] for i in range(len(text))])
text = [[users[i],tweets[i]] for i in range(len(users))]
text


# - ## Function to convert tweets in lowercase

# In[16]:


def lower_case(list_text): # Convert lowercase tweets for language processing
    list_lower_case = []
    for i in list_text:
        word = i.strip()
        new_word = word.lower()
        list_lower_case.append(new_word)
    return(list_lower_case)


# In[17]:


tweets = lower_case([text[i][1] for i in range(len(text))])
text = [[users[i],tweets[i]] for i in range(len(users))]
text


# - ## Replace the emojis for their names

# In[18]:


import emoji # Emoji codes


# In[19]:


def replace_name_emoji(list_text): # function that replaces emoticons by their names
    list_name_emoji = []
    for l in list_text:
        name_emoji = emoji.demojize(l, delimiters=(' ', ' '))
        list_name_emoji.append(name_emoji) 
    return(list_name_emoji)


# In[20]:


tweets = replace_name_emoji([text[i][1] for i in range(len(text))])
tweets = [tweets[i].replace("_"," ") for i in range(len(tweets))]
text = [[users[i],tweets[i]] for i in range(len(users))]
text


# ## Translation

# In[21]:


# provides authentication credentials to application code:

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:/Users/Alejandra/Desktop/Proyecto/My First Project-689f409050bd.json"


# In[22]:


from google.cloud import translate_v2 as translate

def translate_text(text,target='en'):
    translate_client = translate.Client()
    result = translate_client.translate(text, target_language=target, format_= "text")
    
    return(result['translatedText'])


# In[23]:


tweets = [text[i][1] for i in range(len(text))]
tweets


# In[24]:


test_tweets = []

character_count = 0
translated_tweets = 0

for f in tweets:
    tweets_translated = translate_text(f, target = "en")
    test_tweets.append(tweets_translated)
    
    translated_tweets += 1
    character_count += len(f)


# In[25]:


text = [[users[i],test_tweets[i]] for i in range(len(users))]
text


# In[26]:


len(text)


# In[27]:


print("Total characters translated: {0}".format(character_count))
print("Translated Tweets: {0}".format(translated_tweets))


# In[28]:


text[0][1] # first translated tweet


# In[29]:


# exporting translated tweets


# In[30]:


base = pd.DataFrame(text, columns=["users","tweets"])


# In[31]:


base.to_excel('test.xlsx')


# 
# ## Additional Text Cleaning

# - ## Delete blank rows

# In[32]:


df = pd.read_excel("test.xlsx")

df.to_string(index=False)
nan_value = float("nan")

df.replace("", nan_value, inplace=True)

df.dropna(subset = ["tweets"], inplace=True)
df.reset_index(drop=True, inplace=True) 

df.drop("Unnamed: 0", axis=1, inplace=True)

df.head()


# In[33]:


df['tweets']


# In[34]:


len(df)


# In[35]:


df.isna().sum()


# In[36]:


len(df)


# In[37]:


def write_excel(tab,name):
    writer = pd.ExcelWriter(name+'.xlsx', engine='xlsxwriter')
    # Convert the dataframe to an XlsxWriter Excel object.
    tab.to_excel(writer, sheet_name='Sheet1',index=False)
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()


write_excel(df,"test2")


# In[38]:


cols = ['users', 'tweets']
text = df[cols].values.tolist()
text
len(text)


# In[39]:


tweets = [text[i][1] for i in range(len(text))]
len(tweets)


# In[40]:


users = [text[i][0] for i in range(len(text))]
len(users)


# In[41]:


test_tweets = tweets_temp['tweets'].tolist()


# In[51]:


#test_tweets


# In[42]:


write_excel(tweets,"test3")


# ### Grading

# In[41]:


from grader import Grader


# In[42]:


grader = Grader()


# ## Word Embedding

# In[43]:


import gensim

#wv_embedding is the embedding loaded
wv_embeddings = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=500000) ######### YOUR CODE HERE #############
type(wv_embeddings )


# ## From word to text embedding

# In[44]:


#This function converts a question in a vector
import numpy as np
def tweets_to_vec(tweet, embeddings, dim=300):

    result = np.zeros(dim) #300 dimensional vector for phrase vector
    cnt = 0
    words = tweet.split()
    for word in words: #All word vectors composing the phrase are summed
        if word in embeddings:
            result += np.array(embeddings[word])
            cnt += 1
    if cnt != 0: #This would happen if no word was found in the embedding
        result /= cnt
    return result


# In[45]:


import nltk
nltk.download('stopwords')


# In[46]:


import util


# In[47]:


tweet2vec_result = []
for tweet in test_tweets:
    tweet = tweet.strip()
    answer = tweets_to_vec(tweet, wv_embeddings)
    tweet2vec_result = np.append(tweet2vec_result, answer)
    print("hola", tweet2vec_result)
    print(tweet2vec_result)


# ## Evaluation of text similarity

# ### HitsCount and DCGScore

# In[48]:


'''
Metric 1 to validate the precision of a model
'''
def hits_count(dup_ranks, k):
    
    return np.average(np.array(dup_ranks) <= np.array([k]))


# In[49]:


'''
Metric 2 to validate the precision of a model
'''

def dcg_score(dup_ranks, k):

    return np.average((np.array(dup_ranks) <= np.array([k]))*1./(np.log2(1. + np.array(dup_ranks))))


# ## Cosine similarity 

# In[50]:


from sklearn.metrics.pairwise import cosine_similarity


# In[51]:


tweet_vec= []
for tweet in test_tweets:
    # Strip removes the first space in the string
    tweet = tweet.strip()
    
    answer = tweets_to_vec(tweet, wv_embeddings)
    # Print each question with its vectorization
    tweet_vec.append(answer)


# In[52]:


len(tweet_vec)


# In[53]:


tweet_vec


# In[54]:


a=cosine_similarity([tweet_vec[0]],[tweet_vec[1]])


# In[56]:


def cosine_measure(list_text):
    list_total =[]
    sum = 1
    for i in list_text:
        sum = sum+1
        list_cosine =[]
        for j in list_text:
            if (i-j).all():
                a=cosine_similarity([i],[j])
                list_cosine.append(a)
            else:
                list_cosine.append(1)
        list_total.append(list_cosine)
        print(sum)
    return(list_total)


# In[57]:


cosines = cosine_measure(tweet_vec)


# In[58]:


cosines


# In[59]:


len(cosines[0])


# In[60]:


np.argmin(cosines[0])
np.argmax(cosines[0])

cosines[0][1][0][0]


# In[61]:


def column_organization_to_validation(list_text):
    final_list=[]
    for i in range(0,len(list_text)):
        temp=[]
        for j in range(0,len(list_text[i])):
            try:
                temp.append(list_text[i][j][0][0])
            except:
                temp.append(1)
        temp = np.array(temp)
        final_list.append(temp)
    return(final_list)

#def column_organization_to_validation(list_text):
 #   final_list=[]
  #  for i in range(0,len(list_text)):
   #     temp=[]
    #    for j in range(0,len(list_text[i])):
     #       temp.append(list_text[i][j][0][0])
      #  final_list.append(temp)
   # return(final_list)


# In[62]:


final_cosines = column_organization_to_validation(cosines)


# In[63]:


sorted(final_cosines[2])


# In[71]:


a=sorted(final_cosines[0])
#a[0]
#a.index(a[0])
a.index(0.0)


# In[72]:


import random
def data_validation_classification(cosines,tweets):
    total=[]
    for i in range(0,len(cosines)):
        #print(i)
        temp=[]
        organized=sorted(cosines[i])
        min1=random.choice(organized[0:30])
        min2=random.choice(organized[0:30])
        max1=organized[len(organized)-1]
        print(min1)
        pos_min_1=organized.index(min1)
        pos_min_2=organized.index(min2)
        pos_max_1=organized.index(max1)
        temp.append(tweets[i])
        if pos_max_1>=i:
            temp.append(tweets[(pos_max_1+1)])
        else:
            temp.append(tweets[pos_max_1])
        if pos_min_1>=i:
            temp.append(tweets[(pos_min_1+1)])
        else:
            temp.append(tweets[pos_min_1])
        if pos_min_2>=i:
            temp.append(tweets[(pos_min_2+1)])
        else:
            temp.append(tweets[pos_min_2])
        total.append(temp)
    return(total)


# In[73]:


u=data_validation_classification(final_cosines, test_tweets)


# In[ ]:


import xlsxwriter as ws


# In[ ]:


workbook = ws.Workbook('validation.csv') 
worksheet = workbook.add_worksheet()


# In[ ]:


row=0
for i in range(0,len(u)):
    worksheet.write(row, 0, u[i][0])
    worksheet.write(row, 1, u[i][1]) 
    worksheet.write(row, 2, u[i][2]) 
    worksheet.write(row, 3, u[i][3])
    row+=1
workbook.close()


# ### First solution: pre-trained embeddings
# We will work with predefined train and validation:
# 
# - train corpus contains similar sentences at the same row.
# - validation corpus contains the following columns: tweet, similar tweet, negative example 1, negative example 2.
# 

# In[ ]:


def read_corpus(filename):
    data = []
    for line in open(filename, encoding='utf-8'):
        data.append(line.strip().split(','))
    return data


# In[85]:


validation = read_corpus('validation.csv')


# In[86]:


len(validation)


# In[87]:


print(validation)


# In[88]:


def rank_candidates(tweet, candidates, embeddings, dim=300):


    t_vecs = np.array([tweets_to_vec(tweet, embeddings, dim) for i in range(len(candidates))])
    cand_vecs = np.array([tweets_to_vec(candidate, embeddings, dim) for candidate in candidates])
    cosines = np.array(cosine_similarity(t_vecs, cand_vecs)[0])
    merged_list = list(zip(cosines, range(len(candidates)), candidates))
    #print(merged_list)
    sorted_list  = sorted(merged_list, key=lambda x: x[0], reverse=True)
    result = [(b,c) for a,b,c in sorted_list]
    
    return result


# In[89]:


wv_ranking = []
for line in validation:
    for l in line:
        t, *ex = l
        ranks = rank_candidates(t, ex, wv_embeddings)
        wv_ranking.append([r[0] for r in ranks].index(0) + 1)


# In[90]:


for k in [1, 5, 10, 100, 500, 1000]:
    print("DCG@%4d: %.3f | Hits@%4d: %.3f" % (k, dcg_score(wv_ranking, k), k, hits_count(wv_ranking, k)))


# In[91]:


from util import text_prepare


# In[92]:


prepared_validation = []
for line in validation:
    for l in line:
        prepared_validation.append([text_prepare(sentence) for sentence in l])


# In[93]:


wv_prepared_ranking = []
for line in prepared_validation:
    q, *ex = line
    ranks = rank_candidates(q, ex, wv_embeddings)
    wv_prepared_ranking.append([r[0] for r in ranks].index(0) + 1)


# In[94]:


for k in [1, 5, 10, 100, 500, 1000]:
    print("DCG@%4d: %.3f | Hits@%4d: %.3f" % (k, dcg_score(wv_prepared_ranking, k), 
                                              k, hits_count(wv_prepared_ranking, k)))


# ## Graphical representation of tweets

# - ## Basic sentiment analysis

# In[95]:


import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# In[97]:


def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
    return TextBlob(text).sentiment.polarity

base["Subjectivity"] = base["tweets"].apply(getSubjectivity)
base["Polarity"] = base["tweets"].apply(getPolarity)


# In[98]:


base


# In[99]:


allWords = ' '.join([twts for twts in base['tweets']])
wordCloud = WordCloud(width = 500, height = 300, random_state = 21, max_font_size = 119).generate(allWords)

plt.imshow(wordCloud, interpolation = "bilinear")
plt.axis('off')
plt.show()


# In[100]:


def getAnalysis(score):
    if score < 0:
        return "Negative"
    elif score == 0:
        return "Neutral"
    else:
        return "Positive"


# In[104]:


base['Analysis'] = base['Polarity'].apply(getAnalysis)
# base


# In[106]:


j=1
sortedDF = base.sort_values(by=['Polarity'], ascending = "False")
for i in range(0, sortedDF.shape[0]):
    if (sortedDF["Analysis"][i] == "Negative"):
        print(str(j) + ') '+sortedDF['tweets'][i])
        print()
        j = j+1


# In[108]:


plt.figure(figsize =(8,6))
for i in range(0, base.shape[0]):
    plt.scatter(base['Polarity'][i], base['Subjectivity'][i], color = 'Blue')
    
plt.title('Sentiment Analysis')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')


# In[120]:


base['Analysis'].value_counts().plot(kind = 'pie')
plt.show()


# - ## Clustering

# In[74]:


matrix_cosines = np.stack(final_cosines,axis=0)


# In[75]:


from sklearn.cluster import SpectralClustering


a=SpectralClustering(8).fit_predict(matrix_cosines)

import matplotlib.pyplot as plt  

plt.imshow(matrix_cosines)
plt.colorbar()
plt.show()


# In[76]:


names=[]
for i in range(0,len(matrix_cosines)):
    names.append(str(i))
vis=pd.DataFrame(matrix_cosines,columns=names)


# In[77]:


# Graficos de kmeans

from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns


# In[78]:


clustering_kmeans = KMeans(n_clusters=8, precompute_distances="auto", n_jobs=-1)
vis['clusters'] = clustering_kmeans.fit_predict(vis)


# In[79]:


vis = vis.iloc[:, :-1]
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(vis)
principalDf = pd.DataFrame(data = principalComponents, columns = ['pca1', 'pca2'])
vis['clusters'] = clustering_kmeans.fit_predict(vis)


# In[80]:


#plot clusters
sns.scatterplot(x="pca1", y="pca2", hue=vis['clusters'], data=principalDf)
plt.title('K-means Clustering with 2 dimensions')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




