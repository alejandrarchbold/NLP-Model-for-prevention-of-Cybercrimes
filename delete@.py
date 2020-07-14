import re,string

#regular expresion operatiors

def strip_links(text):
    link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links         = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')    
    return text

def strip_all_entities(text):
    entity_prefixes = ['@','#']
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


tests = [
    "@peter I really love that shirt at #Macy. http://bet.ly//WjdiW4",
    "@shawn Titanic tragedy could have been prevented Economic Times: Telegraph.co.ukTitanic tragedy could have been preve... http://bet.ly/tuN2wx",
    "I am at Starbucks http://4sh.com/samqUI (7419 3rd ave, at 75th, Brooklyn)",
]

for t in tests:
    print(strip_all_entities(strip_links(t)))
    
    


from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer(strip_handles=True)
tweet_text = "@abcd @pqrs NoSQL introduction - w3resource http://bit.ly/1ngHC5F  #nosql #database #webdev"
print("\nOriginal Tweet:")
print(tweet_text)
result = tknzr.tokenize(tweet_text)
print("\nTokenize a twitter text:")
print(result)


import emoji
text = "hola carlos 🔥 🔥ghdjf"
text = emoji.demojize(text, delimiters=("", "")) + " "

from googletrans import Translator

translator = Translator()

result = translator.translate(text, src='es')

print(result.text)