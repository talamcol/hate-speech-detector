import pickle
import random
import re
import sys

from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

random.seed(42)
STOPWORDS = stopwords.words('english')  # most common word in english data should not be considered during evaluation
STEMMER = SnowballStemmer("english")  # transformer for word forms


def read_edited_texts_from_file():
    texts = pickle.load(open('X.pkl', 'rb'))
    labels = pickle.load(open('Y.pkl', 'rb'))
    print(len(texts), 'data points read')
    # Counter(labels) creates a dict with how many 0 and how many 1 are in the array
    for label, count_ in Counter(labels).items():
        print(label, ':', 'Amount: ', count_, 'Percentage: ', round(100 * (count_ / len(texts)), 2), '%')
    return texts, labels


def determine_hate_by_text(text):
    command_line_input_processed = process_input(text)
    command_line_input_as_tfmatrix = tfidfizer.transform(command_line_input_processed)
    predicted_label = clf.predict(command_line_input_as_tfmatrix)
    for index in range(0, len(predicted_label)):
        if predicted_label[index] == 0:
            return "No hate"
        else:
            return "Hate"


def process_input(text):
    text.lower()
    text_as_array = text.split()
    text_as_array = [w for w in text_as_array if not w in STOPWORDS]
    text_as_array = [w for w in text_as_array if not re.sub('\'\.,', '', w).isdigit()]
    text_as_array = ' '.join([STEMMER.stem(word) for word in text_as_array])
    return [text_as_array]


def do_downsampling(X_tfidf_matrix, Y):
    X_tfidf_matrix, X_, Y, Y_ = train_test_split(X_tfidf_matrix, Y,
                                                 test_size=0.8, random_state=42,
                                                 stratify=Y)
    print('Downsampled data shape:', X_tfidf_matrix.shape)
    return X_tfidf_matrix, X, Y, Y_


if __name__ == '__main__':
    print('Loading data...', file=sys.stderr)
    X, Y = read_edited_texts_from_file()
    print('Vectorizing your data with TFIDF...', file=sys.stderr)
    tfidfizer = TfidfVectorizer(max_features=4000, min_df=3, max_df=0.9,
                                strip_accents='unicode', use_idf=1, smooth_idf=1,
                                sublinear_tf=1, ngram_range=(1, 2))
    X_tfidf_matrix = tfidfizer.fit_transform(X)
    print('Data shape:', X_tfidf_matrix.shape)
    do_downsample = False
    if do_downsample:  # Only take 20% of the data
        X_tfidf_matrix, X_, Y, Y_ = do_downsampling(X_tfidf_matrix, Y)

    print('Train model with logical regression...', file=sys.stderr)
    clf = LogisticRegression(C=2, random_state=3, class_weight='balanced', solver='liblinear')
    X_train, X_test, Y_train, Y_test = train_test_split(X_tfidf_matrix, Y, test_size=0.2, random_state=42, stratify=Y)
    clf.fit(X_train, Y_train)

    # after training the model, here comes the CLI
    print("-----------------------------------")
    print("    COMMAND LINE HATE SPEECH TOOL")
    print("-----------------------------------")
    user_text = input("Enter a text to evaluate. \n")
    print(determine_hate_by_text(user_text))

    # more text input
    go_on = True
    while go_on:
        user_text = input("Enter another text or stop the CLI exit \n")
        if user_text == "exit":
            go_on = False
        else:
            print(determine_hate_by_text(user_text))

    print("----------------------------------------------------------")
    print("Bye!")