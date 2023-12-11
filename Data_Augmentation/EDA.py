import nltk
import random
from nltk.corpus import wordnet
from datasets import load_dataset

#确保安装了所需的库
#pip install datasets nltk

# 初始化 NLTK
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

def synonym_replacement(sentence, n):
    """ 对句子进行同义词替换 """
    words = nltk.word_tokenize(sentence)
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word.isalnum()]))
    random.shuffle(random_word_list)
    num_replaced = 0

    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    if num_replaced == 0:
        return sentence

    sentence = ' '.join(new_words)
    return sentence

def get_synonyms(word):
    """ 获取词的同义词 """
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char.isalnum() or char == ' '])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

# 加载数据集
dataset = load_dataset("cnn_dailymail", '3.0.0', split='train')

# 应用EDA
for sample in dataset.take(5):  # 仅对前5个样本进行操作
    article = sample['article']
    # 对文章中的每个句子应用同义词替换
    sentences = nltk.sent_tokenize(article)
    augmented_sentences = [synonym_replacement(sentence, 1) for sentence in sentences]
    augmented_article = ' '.join(augmented_sentences)
    print("Original:", article)
    print("Augmented:", augmented_article)
    print("\n")

