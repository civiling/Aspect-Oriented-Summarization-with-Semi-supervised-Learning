from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 加载预训练的BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def bert_encode(text, tokenizer, model):
    """ 将文本编码为BERT的嵌入表示 """
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # 使用平均池化作为句子的表示

def get_relevance(query, sentence, tokenizer, model):
    """ 计算查询和句子的相关性 """
    query_emb = bert_encode(query, tokenizer, model).detach().numpy()
    sentence_emb = bert_encode(sentence, tokenizer, model).detach().numpy()
    relevance = cosine_similarity(query_emb, sentence_emb)  # 使用余弦相似度
    return relevance

# 示例文本
news = """New Year's Day earthquake rattles Japan, killing at least 6 people. Catch up here
Japan was shaken by a 7.5 magnitude earthquake Monday that has left at least six people dead and another two seriously injured, according to officials.
The quake struck at 4:10 p.m. local time at a depth of 10 kilometers (6 miles) in the Noto Peninsula of Ishikawa prefecture, according to the United States Geological Survey.
The quake collapsed buildings, caused fires and triggered tsunami alerts as far away as eastern Russia, prompting orders for residents to evacuate affected coastal areas of Japan.
Here's what to know:
* 		Tsunami warnings: After the earthquake, authorities issued tsunami warnings to residents of Japan's west coast. Those warnings have since been downgraded to advisories. Tsunami warnings are issued when waves are expected to be up to 3 meters (9.8 feet). Tsunami waves of around 1.2 meters (3.9 feet) were reported in Wajima City, Japanese public broadcaster NHK said.
* 		Aftershocks to continue: According to the United States Geological Survey, at least 31 smaller aftershocks were reported near the region where the earthquake struck. The agency said aftershocks could continue for days to months to follow.
* 		Train passengers trapped: At least 1,400 passengers are stranded inside high-speed bullet trains more than 10 hours after the earthquake shook the region, Japan's public broadcaster NHK reported.
* 		Damage to infrastructure: The earthquake on Monday sliced through highways in central Japan, collapsed buildings, caused blazes and disrupted communications. At least 33,000 households were affected by power outages, said Japanese Chief Cabinet Secretary Hayashi Yoshimasa, according to NHK.
* 		Rescue and recovery efforts: At least 8,500 military personnel are on standby to help with emergency efforts following the quake, said Japan's Defense Minister Minoru Kihara. Health officials in the city of Suzu said some doctors could not treat wounded patients because damaged roads mean they are unable to travel to work.
* 		US support: The Biden administration is in touch with Japanese officials, and the United States "stands ready to provide any necessary assistance for the Japanese people," according to a statement.
"""
query = "What is the situation of aftershocks?"

# 预处理和分句
sentences = news.split('.')  # 简单的分句，根据需要调整

# 计算每个句子的相关性
relevance_scores = [get_relevance(query, sent, tokenizer, model) for sent in sentences]

# 选择最相关的句子
top_sentence = sentences[np.argmax(relevance_scores)]

# 输出摘要
print("Relevance Sentence:", top_sentence)

