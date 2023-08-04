from sentence_transformers import models, losses, datasets
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

import logging
from datetime import datetime
import csv
from MultiDatasetDataLoader import MultiDatasetDataLoader
import pandas as pd
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

num_epochs = 10
model_name = '/mnt/vdb1/BERT_training/sbert_approach/ckpt_concat_unique_mask/2023-07-06_13-26-34/checkpoint-630000'
pth_tokenizer = "/mnt/vdb1/BERT_training/sbert_approach/ckpt_concat_unique_mask/2023-07-06_13-26-34"
dataset_path = "/mnt/vdb1/true_false_match_836k.csv"
batch_size_pairs = 380
batch_size_triplets = 250
max_seq_length = 512
use_amp = True                  #Set to False, if you use a CPU or your GPU does not support FP16 operations
evaluation_steps = 500
warmup_steps = 500

# Save path of the model
model_save_path = '/mnt/vdb1/BERT_training/sbert_approach/matcher/output/training_paraphrases_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Load the model
## SentenceTransformer model
word_embedding_model = models.Transformer(model_name, tokenizer_name_or_path=pth_tokenizer, max_seq_length=max_seq_length)



pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
# Our training loss
train_loss = losses.MultipleNegativesRankingLoss(model)



#Read STSbenchmark dataset and use it as development set
logging.info("Read train dev datasets")
dev_samples = []
train_samples = []

#reader = csv.DictReader(dataset_path, delimiter=';', quoting=csv.QUOTE_NONE)
#for row in reader:
df = pd.read_csv(dataset_path, sep=';')
df = df[:1000]
for index, row in df.iterrows():
    #print(row)
    if row['split'] == 'dev':
        score = float(row['score']) 
        dev_samples.append(InputExample(texts=[row['offer'], row['model']], label=score))
    #else:
        score = float(row['score']) 
        train_samples.append(InputExample(texts=[row['offer'], row['model']], label=score, guid = None))

dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')
train_dataloader = MultiDatasetDataLoader([train_samples], batch_size_pairs=batch_size_pairs, batch_size_triplets=batch_size_triplets)

# Configure the training
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=evaluation_steps,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          use_amp=use_amp,
          checkpoint_path=model_save_path,
          checkpoint_save_steps=1000,
          checkpoint_save_total_limit=num_epochs
          )