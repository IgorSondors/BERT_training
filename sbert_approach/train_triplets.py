import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from sentence_transformers import models, losses
from sentence_transformers import SentenceTransformer, InputExample

from torch.utils.data import DataLoader
from sentence_transformers import losses
import pandas as pd
import numpy as np
import random
import torch


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_data(pth_csv, batch_size_triplets):
    train = pd.read_csv(pth_csv, sep=';')
    print(f"load_data: {len(train)} train triplets")
    train_samples = []

    for index, row in train.iterrows():
        train_samples.append(InputExample(texts=[row['name'], row['true_match'], row['false_match']]))
    return DataLoader(train_samples, shuffle=True, batch_size=batch_size_triplets), len(train)

def detaloader_test():
    name = "Apple iPhone 14 Plus 128 ГБ (MQ4H3J/A) BLUE"
    true_match = "Apple iPhone 14 Plus 128Gb"
    false_match = "Apple iPhone 14 Plus 256Gb"
    train_samples = []
    train_samples.append(InputExample(texts=[name, true_match, false_match]))
    train_samples.append(InputExample(texts=[name, true_match, false_match]))
    return DataLoader(train_samples, shuffle=True, batch_size=batch_size_triplets), 2

def train(model, train_dataloader, num_epochs, evaluation_steps, model_save_path, save_steps, use_amp):
    print("train start")
    train_loss = losses.TripletLoss(model=model)

    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1) #10% of train data

    callback = []
    model.fit(train_objectives=[(train_dataloader, train_loss)],
          #evaluator=test_dataloader,
          epochs=num_epochs,
          evaluation_steps=evaluation_steps,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          use_amp=use_amp,
          checkpoint_path=model_save_path,
          checkpoint_save_steps=save_steps,
          checkpoint_save_total_limit=num_epochs,
          #callback=callback
          )
    
    print("train finish")

def load_model(model_name, pth_tokenizer, max_seq_length):
    print("load_model")
    word_embedding_model = models.Transformer(model_name, tokenizer_name_or_path=pth_tokenizer, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model

if __name__ == "__main__":

    model_name = "/mnt/vdb1/BERT_training/sbert_approach/mlm_weights/experiment_part_of_word_masking/18_categories/2023-09-05_16-13-54/checkpoint-700000"
    pth_tokenizer = "/mnt/vdb1/BERT_training/sbert_approach/mlm_weights/experiment_part_of_word_masking/18_categories/2023-09-05_16-13-54"
    model_save_path = '/mnt/vdb1/BERT_training/sbert_approach/matcher/rubert-tiny2-18_categories-fit_loop'

    dataset_path = "/mnt/vdb1/Datasets/18_categories/triplets_train_18_categories.csv"

    # device = "cuda"
    num_epochs = 30
    batch_size_triplets = 450
    # initial_lr = 1e-3
    # max_grad_norm = 1.0
    max_seq_length = 512
    use_amp = True                  #Set to False, if you use a CPU or your GPU does not support FP16 operations
    evaluation_steps = 2000
    
    # seed_everything(42)
    train_dataloader, dataset_len = load_data(dataset_path, batch_size_triplets)
    save_steps = dataset_len//batch_size_triplets # сохраняем раз в эпоху

    print(f"model_name = {model_name}\npth_tokenizer = {pth_tokenizer}\ndataset_path = {dataset_path}\nmodel_save_path = {model_save_path}\n")
    print(f"num_epochs = {num_epochs}\nbatch_size_triplets = {batch_size_triplets}\nmax_seq_length = {max_seq_length}\nuse_amp = {use_amp}\nevaluation_steps = {evaluation_steps}\nsave_steps = {save_steps}\n\n")
    
    model = load_model(model_name, pth_tokenizer, max_seq_length)

    train(model, train_dataloader, num_epochs, evaluation_steps, model_save_path, save_steps, use_amp)