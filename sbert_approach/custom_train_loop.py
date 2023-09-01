from sentence_transformers import models, losses, InputExample, SentenceTransformer
from transformers import AdamW, get_linear_schedule_with_warmup
from sentence_transformers.util import batch_to_device
from torch.utils.data import DataLoader
from tqdm.autonotebook import trange
import torch.nn as nn
import pandas as pd
import torch
import wandb
import time
import os

def load_model(model_name, pth_tokenizer, max_seq_length):
    print("load_model")
    word_embedding_model = models.Transformer(model_name, tokenizer_name_or_path=pth_tokenizer, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model

def load_data(dataset_path, batch_size):
    train = pd.read_csv(dataset_path, sep=';')
    print(f"load_data: {len(train)} train triplets")
    train_samples = []
    for index, row in train.iterrows():
        train_samples.append(InputExample(texts=[row['name'], row['true_match'], row['false_match']]))
    return DataLoader(train_samples, shuffle=True, batch_size=batch_size)

def train_loop(model_name, pth_tokenizer, model_save_path, dataset_path, device, num_epochs, batch_size, initial_lr, max_grad_norm, max_seq_length):
    # Initialize WandB
    wandb.init(project="triplet-loss-experiment2", entity="igor-sondors")
    wandb.config.update({
        "model_name": model_name,
        "pth_tokenizer": pth_tokenizer,
        "model_save_path": model_save_path,
        "dataset_path": dataset_path,
        "device": device,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": initial_lr,
        "max_grad_norm": max_grad_norm,
        "max_seq_length": max_seq_length
    })

    model = load_model(model_name, pth_tokenizer, max_seq_length).to(device)
    data_loader = load_data(dataset_path, batch_size)
    data_loader.collate_fn = model.smart_batching_collate

    train_loss = losses.TripletLoss(model=model)

    optimizer = AdamW(model.parameters(), lr=initial_lr, weight_decay=0.01)  # Using AdamW optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=initial_lr)

    steps_per_epoch = len(data_loader)
    total_steps = steps_per_epoch * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    print("train_loop")
    for epoch in trange(num_epochs, desc="Epoch"):
        data_iterator = iter(data_loader)
        total_loss = 0
        epoch_time = time.time()
        train_loss.zero_grad()
        train_loss.train()
        for _ in trange(steps_per_epoch, desc="Iteration"):
            data = next(data_iterator)
            features, labels = data
            labels = labels.to(device)
            features = list(map(lambda batch: batch_to_device(batch, device), features))

            loss_value = train_loss(features, labels)
            total_loss += loss_value.item()
            loss_value.backward()

            torch.nn.utils.clip_grad_norm_(train_loss.parameters(), max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        avg_loss = total_loss / steps_per_epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}, epoch time: {round(time.time() - epoch_time, 2)} sec')
        wandb.log({"epoch": epoch, "avg_loss": avg_loss, "learning_rate": optimizer.param_groups[0]['lr']})
        model.save(os.path.join(model_save_path, f"epoch_{epoch+1}"))
        # wandb.save(model_save_path)

if __name__ == "__main__":
    model_name = "/mnt/vdb1/BERT_training/sbert_approach/mlm_weights/experiment_part_of_word_masking/2023-07-14_14-52-52/checkpoint-360000"
    pth_tokenizer = "/mnt/vdb1/BERT_training/sbert_approach/mlm_weights/experiment_part_of_word_masking/2023-07-14_14-52-52"
    model_save_path = '/mnt/vdb1/BERT_training/sbert_approach/matcher/rubert-tiny2-custom_train_loop'

    # dataset_path = "/mnt/vdb1/Datasets/triplets_train_aug_clear.csv"
    # dataset_path = "/mnt/vdb1/Datasets/triplets_train_aug.csv"
    dataset_path = "/mnt/vdb1/Datasets/triplets_train.csv"
    # dataset_path = "/mnt/vdb1/Datasets/triplets_test.csv"

    device = "cuda"
    num_epochs = 30
    batch_size = 400
    initial_lr = 1e-3
    max_grad_norm = 1.0
    max_seq_length = 512

    train_loop(model_name, pth_tokenizer, model_save_path, dataset_path, device, num_epochs, batch_size, initial_lr, max_grad_norm, max_seq_length)