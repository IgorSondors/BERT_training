from sentence_transformers import models, losses
from sentence_transformers import SentenceTransformer, InputExample

from datetime import datetime
from torch.utils.data import DataLoader
from sentence_transformers import losses

import pandas as pd

def load_data(pth_csv, batch_size_triplets):
    train = pd.read_csv(pth_csv, sep=';')
    print(f"load_data: {len(train)} train triplets")
    train_samples = []

    for index, row in train.iterrows():
        train_samples.append(InputExample(texts=[row['name'], row['true_match'], row['false_match']]))
    return DataLoader(train_samples, shuffle=True, batch_size=batch_size_triplets)

def train(model, train_dataloader, num_epochs, evaluation_steps, model_save_path, save_steps, use_amp):
    print("train start")
    train_loss = losses.TripletLoss(model=model)

    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1) #10% of train data

    model.fit(train_objectives=[(train_dataloader, train_loss)],
          #evaluator=test_dataloader,
          epochs=num_epochs,
          evaluation_steps=evaluation_steps,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          use_amp=use_amp,
          checkpoint_path=model_save_path,
          checkpoint_save_steps=save_steps,
          checkpoint_save_total_limit=num_epochs
          )
    
    print("train finish")

def load_model(model_name, pth_tokenizer, max_seq_length):
    print("load_model")
    word_embedding_model = models.Transformer(model_name, tokenizer_name_or_path=pth_tokenizer, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model

if __name__ == "__main__":
    num_epochs = 50
    model_name = "/mnt/vdb1/BERT_training/sbert_approach/mlm_weights/experiment_part_of_word_masking/2023-07-14_14-52-52/checkpoint-360000"
    pth_tokenizer = "/mnt/vdb1/BERT_training/sbert_approach/mlm_weights/experiment_part_of_word_masking/2023-07-14_14-52-52"
    dataset_path = "/mnt/vdb1/Datasets/triplets_train_aug.csv"
    # dataset_path = "/mnt/vdb1/Datasets/triplets_train.csv"
    # dataset_path = "/mnt/vdb1/Datasets/triplets_test.csv"
    model_save_path = '/mnt/vdb1/BERT_training/sbert_approach/matcher/rubert-tiny2-aug'+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    batch_size_triplets = 450
    max_seq_length = 512
    use_amp = True                  #Set to False, if you use a CPU or your GPU does not support FP16 operations
    evaluation_steps = 500
    warmup_steps = 500
    save_steps = 5000

    print(f"model_name = {model_name}\npth_tokenizer = {pth_tokenizer}\ndataset_path = {dataset_path}\nmodel_save_path = {model_save_path}\n")
    print(f"num_epochs = {num_epochs}\nbatch_size_triplets = {batch_size_triplets}\nmax_seq_length = {max_seq_length}\nuse_amp = {use_amp}\nevaluation_steps = {evaluation_steps}\nwarmup_steps = {warmup_steps}\nsave_steps = {save_steps}\n\n")


    train_dataloader = load_data(dataset_path, batch_size_triplets)

    model = load_model(model_name, pth_tokenizer, max_seq_length)

    train(model, train_dataloader, num_epochs, evaluation_steps, model_save_path, save_steps, use_amp)