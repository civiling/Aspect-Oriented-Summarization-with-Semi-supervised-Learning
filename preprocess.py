from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset
import torch
# 加载数据集
dataset = load_dataset("cnn_dailymail", "3.0.0")

# 使用更小的模型
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

# ...[数据准备代码]...
def process_data_to_model_inputs(batch):
    # 将文章和摘要编码
    inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=1024)
    outputs = tokenizer(batch["highlights"], padding="max_length", truncation=True, max_length=128)

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["labels"] = outputs.input_ids

    return batch

# 对数据集进行预处理
dataset = dataset.map(process_data_to_model_inputs, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 调整训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,  # 减少训练周期
    per_device_train_batch_size=1,  # 减小批次大小
    per_device_eval_batch_size=1,
    logging_dir='./logs',
    logging_steps=10
)

# 创建 Trainer 实例
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"]
)

# 训练模型
trainer.train()

# 评估模型
results = trainer.evaluate()
print(results)

# 使用模型进行摘要
def generate_summary(text):
    inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# 示例文本
sample_text = dataset['validation'][0]['article']
print(generate_summary(sample_text))

