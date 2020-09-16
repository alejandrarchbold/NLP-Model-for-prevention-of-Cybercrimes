# -*- coding: utf-8 -*-
"""
Created on Wed May 20 13:14:18 2020

@author: Alejandra y Julian
"""


# libraries

import pandas as pd

import numpy as np

import gensim

from sklearn.metrics.pairwise import cosine_similarity

import os

from textblob import TextBlob

import seaborn as sns

from wordcloud import WordCloud

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



# Define data directory

os.chdir("C:/Users/Julian/Desktop/Universidad/ciberseguridad/ciberseguridad/codigos")


"""
Functions:
    
    escribir_excel: Write a dataframe into a xlsx file, receives 2 arguments
                    1. tab= pandas data frame
                    2. nombres= xlsx file name
    
    eliminar_tweets_repetidos: Delete repeat tweets, arguments:
                    1. tabla= data frame with a column with tweets
                    
    quitar_rt: Delete retweets in a dataframe, arguments:
                    1. lista= list of tweets
                    
    tweet_to_vec: Tweet vectorization
                    1. question: string
                    2. embeddings: Dictionary: {key=word, value=vectorization}
                    
    medida_coseno: For each tweet vectorization make cosine distance from the 
                   other tweets
                   1. lista: list of each tweet vectorized
                
    lista_good: Make a symetric matrix where each row consists of the cosine 
                distance between one tweet and the others.
                   1. lista: list of cosines distances
    
    getAnalysis: Clasify polarity
    
"""

def escribir_excel(tab,nombre):
    writer = pd.ExcelWriter(nombre+'.xlsx', engine='xlsxwriter')
    # Convert the dataframe to an XlsxWriter Excel object.
    tab.to_excel(writer, sheet_name='Sheet1')
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()

    
def eliminar_tweets_repetidos(tabla):
    lista_temp=[]
    numeros=[]
    for i in range(0,len(tabla)):
        if len(lista_temp)==0:
            lista_temp.append(tabla.iloc[i][0])
        elif tabla.iloc[i][0] not in lista_temp:
            lista_temp.append(tabla.iloc[i][0])
        else:
            numeros.append(i)
    return(numeros)
            
def quitar_rt(lista):
    lista_final=[]
    for i in range(0,len(lista)):
        pal=lista[i].replace('rt', '')
        lista_final.append(pal)
    return lista_final


def tweet_to_vec(question, embeddings, dim=300):
    """
        question: string
        embeddings: llave=palabra, valor=vectorizacion
        
    """
    result = np.zeros(dim)
    cnt = 0
    # pregunta se convierte en lista
    words = question.split()
    for word in words:
        # Buscamos cada palabra de la lista en el diccionario
        if word in embeddings:
            result += np.array(embeddings[word])
            cnt += 1
    if cnt != 0:
        result /= cnt
    return result


def medida_coseno(lista):
    lista_total=[]
    sum=1
    for i in lista:
        sum=sum+1
        lista_coseno=[]
        for j in lista:
            if (i-j).all():
                a=cosine_similarity([i],[j])
                lista_coseno.append(a)
            else:
                lista_coseno.append(1)
        lista_total.append(lista_coseno)
        print(sum)
    return(lista_total)

def getAnalysis(score):
    if score < 0:
        return "Negative"
    elif score == 0:
        return "Neutral"
    else:
        return "Positive"
    

def lista_good(lista):
    final=[]
    for i in range(0,len(lista)):
        temp=[]
        for j in range(0,len(lista[i])):
            try:
                temp.append(lista[i][j][0][0])
            except:
                temp.append(1)
        temp=np.array(temp)
        final.append(temp)
    return(final)



############################# Start ################################



# Read a database that has two columns: 1. Users, 2. Tweets

tabla=pd.read_excel("usuarios2.xlsx")

# Delete "NA" tweets

nan_value = float("nan")

tabla.replace("", nan_value, inplace=True)

tabla.dropna()

# Delete repeat tweets

num=eliminar_tweets_repetidos(tabla)

tabla_temp=tabla

for i in num:
    tabla_temp=tabla_temp.drop(i)

tabla_temp=tabla_temp.dropna()

############################ clusters analysis  ###################################3

# Read embedding database in this code
wv_embeddings = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=500000)


# transform the column of tweets in dataframe to list
text=tabla_temp['primero'].tolist()


# Delete retweets
lista_v1=quitar_rt(text)

# Make vectorization of each tweet

tweet_vec= []
for tweet in lista_v1:
    # Strip elimina el primer espacio en el string
    question = tweet.strip()
    # Cada pregunta la pasamos al testeo
    answer = tweet_to_vec(question, wv_embeddings)
    # Imprimimos cada pregunta con su vectorizacion 
    tweet_vec.append(answer)
    #print(question2vec_result)


# Make cosine distance

cosenos=medida_coseno(tweet_vec)

# Make a matrix of cosines distance
    
cosenos_final=lista_good(cosenos)

matriz_cosenos=np.stack(cosenos_final,axis=0)


# Cluster graphics 


from sklearn.cluster import SpectralClustering

a=SpectralClustering(8).fit_predict(matriz_cosenos)

import matplotlib.pyplot as plt  


# plot cosine distances matrix

#plt.imshow(matriz_cosenos)
#plt.colorbar()
#plt.show()

# Create a data frame from the matrix

nombres=[]
for i in range(0,len(matriz_cosenos)):
    nombres.append(str(i))
vis=pd.DataFrame(matriz_cosenos,columns=nombres)


# k-means plot

from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns


# Add into the data frame of the cosine distances another column of the cluster of each tweet

clustering_kmeans = KMeans(n_clusters=8, precompute_distances="auto", n_jobs=-1)
vis['clusters'] = clustering_kmeans.fit_predict(vis)

# Write an excel with the results
escribir_excel(vis,"resultados")

#results = pd.DataFrame(reduced_data,columns=['pca1','pca2'])

# Plot k-means clustering

# vis = vis.iloc[:, :-1]
#from sklearn.decomposition import PCA
#pca = PCA(n_components=2)
#principalComponents = pca.fit_transform(vis)
#principalDf = pd.DataFrame(data = principalComponents
#             , columns = ['pca1', 'pca2'])
#vis['clusters'] = clustering_kmeans.fit_predict(vis)


#sns.scatterplot(x="pca1", y="pca2", hue=vis['clusters'], data=principalDf)
#plt.title('K-means Clustering with 2 dimensions')
#plt.show()
    

############################### Sentimental analysis #####################################


# Read the previous results
tabla=pd.read_excel("resultados.xlsx")

# Read again the first table of users
usu=pd.read_excel("usuarios2.xlsx")

# Rename columns

tab_revisada["usuarios"]=usu[["users"]]

tab_revisada["tweets"]=usu[["tweets"]]

tabla_2["clusters"]=tabla["clusters"]

# Create a column with the tweet polarity

tabla_2["polaridad"]=tabla_2["tweets"].apply(lambda x:TextBlob(x).sentiment.polarity)

#  Create a column with the tweet subjectivity

tabla_2["subjetividad"]=tabla_2["tweets"].apply(lambda x:TextBlob(x).sentiment.subjectivity)


# Words cloud of each cluster

tab1=tabla_2[tabla_2["clusters"]==0]
allWords = ' '.join([twts for twts in tab1['tweets']])
wordCloud1 = WordCloud(width = 500, height = 300, random_state = 21, max_font_size = 119,background_color="white").generate(allWords)
plt.imshow(wordCloud1, interpolation = "bilinear")
plt.axis('off')
plt.show()

tab2=tabla_2[tabla_2["clusters"]==1]
allWords = ' '.join([twts for twts in tab2['tweets']])
wordCloud2 = WordCloud(width = 500, height = 300, random_state = 21, max_font_size = 119,background_color="white").generate(allWords)
plt.imshow(wordCloud2, interpolation = "bilinear")
plt.axis('off')
plt.show()

tab3=tabla_2[tabla_2["clusters"]==2]
allWords = ' '.join([twts for twts in tab3['tweets']])
wordCloud3 = WordCloud(width = 500, height = 300, random_state = 21, max_font_size = 119,background_color="white").generate(allWords)
plt.imshow(wordCloud3, interpolation = "bilinear")
plt.axis('off')
plt.show()

tab4=tabla_2[tabla_2["clusters"]==3]
allWords = ' '.join([twts for twts in tab4['tweets']])
wordCloud4 = WordCloud(width = 500, height = 300, random_state = 21, max_font_size = 119,background_color="white").generate(allWords)
plt.imshow(wordCloud4, interpolation = "bilinear")
plt.axis('off')
plt.show()


# Only one plot with all clouds

tab4=tab4.sort_values(by=['polaridad'])
bu=tab4[0:100]

lista_cloud=[wordCloud1,wordCloud2,wordCloud3,wordCloud4]

fig, ax = plt.subplots(2, 2)

ax[0][0].imshow(lista_cloud[3], interpolation = "bilinear")
ax[0][0].set_title('Cluster 1')
ax[0][0].set_yticklabels([])
ax[0][0].set_xticklabels([])
ax[0][1].imshow(lista_cloud[1], interpolation = "bilinear")
ax[0][1].title.set_text('Cluster 2')
ax[0][1].set_yticklabels([])
ax[0][1].set_xticklabels([])
ax[1][0].imshow(lista_cloud[2], interpolation = "bilinear")
ax[1][0].title.set_text('Cluster 3')
ax[1][0].set_yticklabels([])
ax[1][0].set_xticklabels([])
ax[1][1].imshow(lista_cloud[0], interpolation = "bilinear")   
ax[1][1].title.set_text('Cluster 4')
ax[1][1].set_yticklabels([])
ax[1][1].set_xticklabels([])
plt.show() 


fig.savefig("exp1.pdf", bbox_inches='tight')

    
# Make plots of polarity

tab4['Analysis'] = tab4['polaridad'].apply(getAnalysis)
# base
tab4['Analysis'].value_counts().plot(kind = 'pie')
plt.show()

tab3['Analysis'] = tab3['polaridad'].apply(getAnalysis)
# base
tab3['Analysis'].value_counts().plot(kind = 'pie')
plt.show()


tab2['Analysis'] = tab2['polaridad'].apply(getAnalysis)
# base
tab2['Analysis'].value_counts().plot(kind = 'pie')
plt.show()


tab1['Analysis'] = tab1['polaridad'].apply(getAnalysis)
# base
tab1['Analysis'].value_counts().plot(kind = 'pie')
plt.show()


labels=["neutral","negative","positive"]

labels2=["negative","neutral","positive"]

fig, ax = plt.subplots(2, 2)

ax[0][0].pie(tab1['Analysis'].value_counts(),labels=labels,colors=["blue","green","orange"])
ax[0][0].set_title('Cluster 1')
ax[0][1].pie(tab2['Analysis'].value_counts(),labels=labels2,colors=["green","blue","orange"])
ax[0][1].title.set_text('Cluster 2')
ax[1][0].pie(tab3['Analysis'].value_counts(),labels=labels2,colors=["green","blue","orange"])
ax[1][0].set_title('Cluster 3')
ax[1][1].pie(tab4['Analysis'].value_counts(),labels=labels,colors=["blue","green","orange"])
ax[1][1].set_title('Cluster 4')
plt.show() 



a=tabla.sort_values(by=['polaridad'])

escribir_excel(a, "polaridad")



## Select aggresive users ##


tabla_nodos=pd.read_excel("Nodos.xlsx")

lista=[]
for i in range(0,len(tabla_nodos)):
    if any(tabla.usuarios==tabla_nodos.Id[i]):
        ind=tabla[tabla["usuarios"]==tabla_nodos.Id[i]].index[0]
        lista.append([i,ind])
        
# agrego al dato de los clusters+
for i in range(0,32):
    tabla_nodos["Cluster"][lista[i][0]]=3
    tabla_nodos
    
    
escribir_excel(tabla_nodos, "Nodos_nuevo")




