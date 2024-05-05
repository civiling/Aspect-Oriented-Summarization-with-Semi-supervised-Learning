from datasets import load_dataset
import requests
import hashlib
import random
import json

# 百度翻译 API 函数
def baidu_translate(text, from_lang, to_lang, appid, secretKey):
    base_url = "http://api.fanyi.baidu.com/api/trans/vip/translate"
    salt = random.randint(32768, 65536)
    sign = hashlib.md5((appid + text + str(salt) + secretKey).encode()).hexdigest()
    params = {
        'q': text,
        'from': from_lang,
        'to': to_lang,
        'appid': appid,
        'salt': salt,
        'sign': sign,
    }
    response = requests.get(base_url, params=params)
    result = response.json()

    # 检查响应是否包含 'trans_result'
    if 'trans_result' in result:
        return result['trans_result'][0]['dst']
    else:
        # 打印错误信息
        print("Error in translation API response:", result)
        return ""


# 回译函数
def back_translate(texts, appid, secretKey):
    translated_texts = []
    for text in texts:
        # 英文到法文
        text_fr = baidu_translate(text, 'en', 'fra', appid, secretKey)
        # 法文到英文
        text_en = baidu_translate(text_fr, 'fra', 'en', appid, secretKey)
        translated_texts.append(text_en)
    return translated_texts

# 百度翻译 API 凭证
appid = '20231027001861483'
secretKey = 'XOncfPViJzyF9VDfvpvl'

# 加载 CNN/DailyMail 数据集的一个子集
dataset = load_dataset("cnn_dailymail", "3.0.0", split='train[:1%]')
texts = [article['article'] for article in dataset]  # 提取文章文本

# 选择部分文本进行回译
sample_texts = texts[:2]  # 例如选择前10篇文章
back_translated_texts = back_translate(sample_texts, appid, secretKey)

for original, back_translated in zip(sample_texts, back_translated_texts):
    print("Original:", original)
    print("Back-translated:", back_translated)
    print("\n")

