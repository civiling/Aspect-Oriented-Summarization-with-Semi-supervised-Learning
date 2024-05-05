import random
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet
from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
def synonym_replacement(sentence, num_replacements=5):
    words = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(words)

    tfidf = TfidfVectorizer(tokenizer=nltk.word_tokenize, stop_words='english')
    tfidf_matrix = tfidf.fit_transform([sentence])
    dense = tfidf_matrix.todense()
    dense_list = dense.tolist()
    tfidf_scores = {word: dense_list[0][i] for word, i in tfidf.vocabulary_.items()}

    sorted_words = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
    candidates = [word for word, score in sorted_words[:num_replacements]]

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

def get_synonyms(word, pos_tag):
    pos = pos_tag[0].lower()
    if pos in ['a', 's']:
        pos = wordnet.ADJ
    elif pos == 'v':
        pos = wordnet.VERB
    elif pos == 'n':
        pos = wordnet.NOUN
    elif pos == 'r':
        pos = wordnet.ADV
    else: 
        return []

    synonyms = set()
    for syn in wordnet.synsets(word, pos=pos):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    return list(synonyms)

# 加载数据集
dataset = load_dataset("cnn_dailymail", "3.0.0")

def augment_data(example):
    augmented_text = synonym_replacement(example['article'])
    return {"article": example["article"], "augmented_article": augmented_text}

dataset = dataset.map(augment_data)

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')


def compute_loss(model, batch, return_outputs=False):
    inputs = tokenizer(batch["article"], return_tensors="pt", padding=True, truncation=True, max_length=1024)
    augmented_inputs = tokenizer(batch["augmented_article"], return_tensors="pt", padding=True, truncation=True, max_length=1024)

    labels = tokenizer(batch["highlights"], return_tensors="pt", padding=True, truncation=True, max_length=128).input_ids
    labels = np.where(labels != tokenizer.pad_token_id, labels, -100)

    outputs = model(**inputs, labels=labels)
    augmented_outputs = model(**augmented_inputs, labels=labels)

    consistency_loss = torch.mean(torch.abs(outputs.logits - augmented_outputs.logits))
    total_loss = outputs.loss + augmented_outputs.loss + consistency_loss
    return (total_loss, outputs) if return_outputs else total_loss
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    # 使用数据集的一个子集
    #train_dataset = dataset["train"].select(range(1000)),
    #eval_dataset = dataset["validation"].select(range(500)),
    compute_loss=compute_loss
)

trainer.train()

# 评估模型性能
results = trainer.evaluate()
print("Evaluation results:", results)

def generate_summary(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# 选择一个样本文本
sample_article = dataset["validation"][0]["article"]
generated_summary = generate_summary(sample_article, tokenizer, model)

print("Original Article:\n", sample_article)
print("\nGenerated Summary:\n", generated_summary)

# 保存模型
model_path = "./bart_summary_model"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# 在未来的会话中加载模型
model = BartForConditionalGeneration.from_pretrained(model_path)
tokenizer = BartTokenizer.from_pretrained(model_path)

