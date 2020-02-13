from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def build_lda(data, LDA_config):
    num_of_topic=LDA_config['max_features']
    count_feature=LDA_config['features']
    max_iter=LDA_config['max_iter']
    vec = CountVectorizer(ngram_range=LDA_config['ngram'])

    transformed_data = vec.fit_transform(data)
    feature_names = vec.get_feature_names()
    lda = LatentDirichletAllocation(n_components=num_of_topic, max_iter=max_iter,learning_method='online', random_state=0)
    lda.fit(transformed_data)

    lda_matrix=lda.fit_transform(transformed_data)

    return lda, lda_matrix, feature_names

def display_word_distribution(model, feature_names, n_word):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        words = []
        for i in topic.argsort()[:-n_word - 1:-1]:
            words.append(feature_names[i])
        print(words)


def lda(data,LDA_config):
    lda_model, lda_matrix, feature_names = build_lda(data,LDA_config)
    print("Displaying sample of first 5 words")
    display_word_distribution(model=lda_model, feature_names=feature_names,n_word=5)
    print(feature_names)

    print(lda_matrix.shape)

    return lda_matrix


if __name__ == '__main__':
    data = ['This is the first document. brown adament',
            'This document is the second document. brown adament',
            'And this is the third one. adore red red brown',
            'Is this the first document adament red red?'
            ]
    lsi_config = {
        'features': 30,
        'max_iter':10,
        'max_features': 4,
        'ngram':(1,1)
    }
    lda(data, lsi_config)
