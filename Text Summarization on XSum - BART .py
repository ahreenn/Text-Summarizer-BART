#!/usr/bin/env python
# coding: utf-8

# # Load XSum Dataset

# In[1]:


import datasets
from datasets import load_dataset
import evaluate
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer
import torch
import numpy as np
import nltk


# In[2]:


raw_datasets = datasets.load_dataset("xsum", split="train[:2000]")


# In[3]:


metric = evaluate.load("rouge")


# In[4]:


raw_datasets


# In[5]:


raw_datasets[0]


# In[6]:


metric


# # Model and Tokenizer

# In[7]:


model_name ="facebook/bart-large-cnn"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# tokenization
encoder_max_length = 256  # demo
decoder_max_length = 64


# # Preprocessing the data

# In[8]:


train_data_txt, validation_data_txt = raw_datasets.train_test_split(test_size=0.1).values()


# In[9]:


def batch_tokenize_preprocess(batch, tokenizer, max_source_length, max_target_length):
    source, target = batch["document"], batch["summary"]
    source_tokenized = tokenizer(
        source, padding="max_length", truncation=True, max_length=max_source_length
    )
    target_tokenized = tokenizer(
        target, padding="max_length", truncation=True, max_length=max_target_length
    )

    batch = {k: v for k, v in source_tokenized.items()}
    # Ignore padding in the loss
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in l]
        for l in target_tokenized["input_ids"]
    ]
    return batch


train_data = train_data_txt.map(
    lambda batch: batch_tokenize_preprocess(
        batch, tokenizer, encoder_max_length, decoder_max_length
    ),
    batched=True,
    remove_columns=train_data_txt.column_names,
)

validation_data = validation_data_txt.map(
    lambda batch: batch_tokenize_preprocess(
        batch, tokenizer, encoder_max_length, decoder_max_length
    ),
    batched=True,
    remove_columns=validation_data_txt.column_names,
)


# # Metrics

# In[10]:


nltk.download("punkt", quiet=True)


# In[11]:


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


# In[12]:


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract a few results from ROUGE
    result = {key: value * 100 for key, value in result.items()}

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


# # Training arguments

# In[13]:


training_args = Seq2SeqTrainingArguments(
    output_dir="nlp_summarization_project",
    num_train_epochs=1,  # demo
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=4,  # demo
    per_device_eval_batch_size=4,
    learning_rate=3e-05,
    warmup_steps=500,
    weight_decay=0.1,
    label_smoothing_factor=0.1,
    predict_with_generate=True,
    logging_dir="logs",
    logging_steps=50,
    save_total_limit=3,
    push_to_hub=True,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_data,
    eval_dataset=validation_data,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


# # Train

# In[14]:


trainer.evaluate()


# In[15]:


trainer.train()


# # Evaluate the model after fine-tuning

# In[16]:


trainer.evaluate()


# In[19]:


trainer.push_to_hub()


# # Evaluation

# In[17]:


# Generate summaries from the fine-tuned model and compare them with those generated from the original, pre-trained one.


# In[42]:


from transformers import pipeline

summarizer = pipeline("summarization", model="airinkonno/nlp_summarization_project")


# In[45]:


text = "Elon Musk's brain-chip firm says it has received approval from the US Food and Drugs Administration (FDA) to conduct its first tests on humans. The Neuralink implant company wants to help restore vision and mobility to people by linking brains to computers. It says it does not have immediate plans to start recruiting participants. Mr Musk's previous ambitions to begin tests came to nothing. The FDA said it acknowledged Neuralink's announcement. An earlier bid by Neuralink to win FDA approval was rejected on safety grounds, according to a report in March by the Reuters news agency that cited multiple current and former employees."


# In[46]:


summarizer(text)


# In[ ]:




