## Steps to run the project

### Before running the code it is necessary to download these files
- Download google embedding

https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit

- Dowload validation, test_embeddings, prepared_test, prepared_train, test, train. 

https://drive.google.com/drive/folders/1sRGbLqDEGDoOVlayHdgSqRWhdbjpDMTs?usp=sharing


### Preprocessing is located in the project folder in the NLP_similarity_models.ipynb file.

- Download the repository and save it somewhere on your computer. Then get the location and write it inside the os.chdir and os.environ.

The first step in preprocessing was to remove URLs, mentions and hashtags. The second step was to convert the letters of all the tweets to lowercase. Then, emoticons were replaced by its meaning in words through the use of the Python library emoji. Finally, empty rows were removed from all tweets. Next, the tweets were translated from spanish to english using Google Api Services. The purpose of this translation was to uniform the language to the one used by Google News Embedding, in order to be able to vectorize the Tweets.

Additionally, the following methods are employed to determine the performance of each model: hitsCount and dcgScore. These methods define the percentage of success of being able to rank a question as positive or negative regarding the similarity with a given question.

### Clusters and Analysis

- Run the procesamiento_final code and change the location of the folder and have the relevant files in the same folder.
