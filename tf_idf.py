import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet
import numpy as np
import random
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# 定义同义词替换函数
def synonym_replacement(sentence, num_replacements):
    words = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(words)

    # 计算 TF-IDF
    tfidf = TfidfVectorizer(tokenizer=nltk.word_tokenize, stop_words='english')
    tfidf_matrix = tfidf.fit_transform([sentence])
    dense = tfidf_matrix.todense()
    dense_list = dense.tolist()
    tfidf_scores = {word: dense_list[0][i] for word, i in tfidf.vocabulary_.items()}
    
    # 选择替换候选词
    sorted_words = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
    candidates = [word for word, score in sorted_words[:num_replacements]]

    # 找到同义词并替换
    replaced = 0
    for word, pos in pos_tags:
        if replaced >= num_replacements:
            break
        synonyms = get_synonyms(word, pos)
        if word in candidates and synonyms:
            synonym = random.choice(synonyms)
            sentence = sentence.replace(word, synonym, 1)
            replaced += 1

    return sentence

# 获取同义词
def get_synonyms(word, pos_tag):
    pos = pos_tag[0].lower()
    if pos in ['a', 's']:  # 形容词
        pos = wordnet.ADJ
    elif pos == 'v':  # 动词
        pos = wordnet.VERB
    elif pos == 'n':  # 名词
        pos = wordnet.NOUN
    elif pos == 'r':  # 副词
        pos = wordnet.ADV
    else: 
        return []

    synonyms = set()
    for syn in wordnet.synsets(word, pos=pos):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    return list(synonyms)

# 应用数据增强
def augment_data(example):
    augmented_text = synonym_replacement(example['article'], 5)  # 假设替换5个单词
    return {"article": example["article"], "augmented_article": augmented_text}

# 使用示例
sample_sentence = "The quick brown fox jumps over the lazy dog."
augmented_sentence = augment_data({"article": sample_sentence})["augmented_article"]
print(augmented_sentence)

