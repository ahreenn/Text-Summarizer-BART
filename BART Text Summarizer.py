#!/usr/bin/env python
# coding: utf-8

# In[2]:


import datasets
from datasets import load_dataset
import evaluate
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer
import torch
import numpy as np
import nltk


# In[7]:


from transformers import pipeline

summarizer = pipeline("summarization", model="airinkonno/nlp_summarization_project")


# In[9]:


text = "Elon Musk's brain-chip firm says it has received approval from the US Food and Drugs Administration (FDA) to conduct its first tests on humans. The Neuralink implant company wants to help restore vision and mobility to people by linking brains to computers. It says it does not have immediate plans to start recruiting participants. Mr Musk's previous ambitions to begin tests came to nothing. The FDA said it acknowledged Neuralink's announcement. An earlier bid by Neuralink to win FDA approval was rejected on safety grounds, according to a report in March by the Reuters news agency that cited multiple current and former employees. Neuralink hopes to use its microchips to treat conditions such as paralysis and blindness, and to help certain disabled people use computers and mobile technology. The chips - which have been tested in monkeys - are designed to interpret signals produced in the brain and relay information to devices via Bluetooth. Experts have cautioned that Neuralink's brain implants will require extensive testing to overcome technical and ethical challenges if they are to become widely available."


# In[10]:


summarizer(text)


# In[ ]:




