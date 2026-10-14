import nltk
import ssl
from ssl import _create_unverified_context
from gensim import corpora
from pprint import pprint  # pretty-printer
import re

from nltk.corpus import stopwords

try:
    _create_unverified_https_context = _create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


nltk.download('stopwords')

documents = [
    "Human machine survey computer interface interface eps time for lab abc computer applications user149",
    "A survey of user149 opinion of computer system user149 response time computer user149 interface interface",
    "The EPS user149 users interfaces interface human interface computer human management system user149",
    "System and human interface interface engineering testing of EPS computer user149",
    "Relation of users perceived response time to error measurement trees",
    "The generation of random binary unordered paths minors user149 user149 computer",
    "The intersection graph of paths in trees paths trees",
    "Graph minors IV Widths of trees and well quasi ordering graph paths",
    "Graph minors A tree paths binary trees graphs",
]

# remove common words and tokenize
stoplist = stopwords.words('english')
texts = [
    [word for word in document.lower().split() if word not in stoplist]
    for document in documents
]

print("Tokens of each document:")
pprint(texts)

# create mapping keyword-id
dictionary = corpora.Dictionary(texts)

print()
print("Mapping keyword-id:")
pprint(dictionary.token2id)

# create the vector for each doc
model_bow = [dictionary.doc2bow(text) for text in texts]

id2token = dict(dictionary.items())


def convert(match):
    return dictionary.id2token[int(match.group(0)[0:-1])]


print()
print("Vectors for documents (the positions with zeros are not shown):")
for doc in model_bow:
    print(re.sub("[0-9]+,", convert, str(doc)))
