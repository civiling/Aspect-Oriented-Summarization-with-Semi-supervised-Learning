import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW  # 从 PyTorch 中导入 AdamW
from transformers import BartTokenizer, BartForConditionalGeneration
from datasets import load_dataset
import nltk
import random
from nltk.corpus import wordnet


# EDA 相关函数
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

# 初始化 NLTK
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

# 数据集类
class CNNDMDataset(Dataset):
    def __init__(self, tokenizer, original_articles, augmented_articles, summaries):
        self.tokenizer = tokenizer
        self.original_articles = original_articles
        self.augmented_articles = augmented_articles
        self.summaries = summaries

    def __len__(self):
        return len(self.original_articles)

    def __getitem__(self, idx):
        original_article = self.original_articles[idx]
        augmented_article = self.augmented_articles[idx]
        summary = self.summaries[idx]

        original_inputs = self.tokenizer(original_article, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        augmented_inputs = self.tokenizer(augmented_article, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        targets = self.tokenizer(summary, return_tensors='pt', padding='max_length', truncation=True, max_length=128)

        # Flatten the output
        original_inputs = {k: v.squeeze() for k, v in original_inputs.items()}
        augmented_inputs = {k: v.squeeze() for k, v in augmented_inputs.items()}
        targets = targets['input_ids'].squeeze()

        return original_inputs, augmented_inputs, targets

# 加载数据集
dataset = load_dataset("cnn_dailymail", '3.0.0', split='train')
original_articles = [sample['article'] for sample in dataset]
summaries = [sample['highlights'] for sample in dataset]

# 使用EDA处理数据
augmented_articles = [synonym_replacement(article, 1) for article in original_articles]

# 初始化模型和分词器
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')


# 创建数据加载器
train_dataset = CNNDMDataset(tokenizer, original_articles, augmented_articles, summaries)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)  # 使用 PyTorch 的 AdamW

model.train()
for epoch in range(1):  # 示例：1个训练周期
    for original_inputs, augmented_inputs, targets in train_loader:
        # 将数据移至相应设备
        original_inputs = {k: v.to(device) for k, v in original_inputs.items()}
        augmented_inputs = {k: v.to(device) for k, v in augmented_inputs.items()}
        targets = targets.to(device)

        # 前向传播
        #outputs = model(**original_inputs, labels=targets['input_ids'])
        #augmented_outputs = model(**augmented_inputs, labels=targets['input_ids'])
        outputs = model(**original_inputs, labels=targets)
        loss_1 = outputs.loss
        augmented_outputs = model(**augmented_inputs, labels=targets)
        augmented_loss = augmented_outputs.loss
        # 计算一致性损失
        consistency_loss = torch.nn.functional.mse_loss(outputs.logits, augmented_outputs.logits)



        # 计算损失
        loss = loss_1 + augmented_loss + consistency_loss  # 这里可以添加一致性损失

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch}, Loss: {loss.item()}")

# 保存模型
model.save_pretrained("/Users/deng/PycharmProjects/NLP")
