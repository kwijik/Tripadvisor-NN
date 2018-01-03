
from sklearn.feature_extraction.text import CountVectorizer
#from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag


class InputRep:
    """A simple feature extractor and vectorizer"""

    def __init__(self):
       # sw = open("./resources/en_negation_words.txt").readline()
        with open("./resources/stopwords_all1_en.txt") as f:
            sw = f.readlines()
        sw = [line.rstrip() for line in sw]
        # self.negations = sorted(set(lines))
        self.stopset = sorted(set(sw))
        self.lemmatizer = WordNetLemmatizer()
        self.max_features = 4000
        #lines = open("./resources/en_negation_words.txt").readline()
        #with open("./resources/en_negation_words.txt") as f:
        #    lines = f.readlines()
       # lines = [line.rstrip for line in self.lines]
        #self.negations = sorted(set(lines))
        # create the vectorizer
        self.vectorizer = CountVectorizer(
            max_features= self.max_features,
            strip_accents=None,
            analyzer="word",
            tokenizer=self.mytokenize,
            stop_words=None,
            ngram_range= (1, 3),
            binary=False,
            preprocessor=None
        )
        #print(self.negations)

    def mytokenize(self, text):
        map = {
            'JJ': 'a',
            'JJR': 'a',
            'JJS': 's',
            'NN': 'n',
            'NNS': 'n',
            'NNP': 'n',
            'NNPS': 'n',
            'VB': 'v',
            'VBD': 'v',
            'VBG': 'v',
            'VBN': 'v',
            'VBP': 'v',
            'VBZ': 'v',
            'RB': 'r',
            'RBR': 'r',
            'RBS': 'r',
            'WRB': 'r',
        }

        """Customized tokenizer.
        Here you can add other linguistic processing and generate more normalized features
        """
        lemmatiser = WordNetLemmatizer()
        tokens = word_tokenize(text)


        tokens = [t.lower() for t in tokens]
        #tokens = [t for t in tokens if (t not in self.stopset or t in self.negations)]
        tokens = [t for t in tokens if t not in self.stopset]

        tokens_pos = pos_tag(tokens)

        result = []

        for t in tokens_pos:
            if t[0] not in self.stopset:
                if (t[1] in map):
                    result.append(lemmatiser.lemmatize(t[0], pos=map[t[1]]))
                else:
                    result.append(t[0])

        return result

    def fit(self, train_texts, unlabeled=None):
        # fit to train corpus
        self.vectorizer.fit(train_texts)
        # print(self.vectorizer.get_feature_names()) # to manually check if the tokens are reasonable

    def get_vects(self, texts):
        '''
        Tokenizes and creates a BoW vector.
        :param texts: A list of strings each string representing a text.
        :return: X: A sparse csr matrix of TFIDF or Count -weighted ngram counts.
        '''
        X = self.vectorizer.transform(texts)
        return X.toarray()
