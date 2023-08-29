from sentence_transformers import models
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
import pandas as pd
import torch
import wandb
import time

class TripletDataset(Dataset):
    def __init__(self, anchor_inputs, positive_inputs, negative_inputs):
        self.anchor_inputs = anchor_inputs
        self.positive_inputs = positive_inputs
        self.negative_inputs = negative_inputs
        
    def __len__(self):
        return len(self.anchor_inputs)
    
    def __getitem__(self, idx):
        anchor = self.anchor_inputs[idx]
        positive = self.positive_inputs[idx]
        negative = self.negative_inputs[idx]
        return anchor, positive, negative

class CustomTripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(CustomTripletLoss, self).__init__()
        self.margin = margin
        self.distance = nn.PairwiseDistance(p=2)
        
    def forward(self, anchor, positive, negative):
        # Calculate distances
        distance_pos = self.distance(anchor, positive)
        distance_neg = self.distance(anchor, negative)
        
        # Calculate triplet loss
        loss = torch.relu(distance_pos - distance_neg + self.margin)
        return loss.mean()

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()
        
def load_model(model_name, pth_tokenizer, max_seq_length):
    print("load_model")
    word_embedding_model = models.Transformer(model_name, tokenizer_name_or_path=pth_tokenizer, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model

def load_data(dataset_path):
    # Create an instance of the TripletDataset
    print("load_data")
    train_df = pd.read_csv(dataset_path, sep=';')
    train_data = TripletDataset(train_df['name'], train_df['true_match'], train_df['false_match'])
    return DataLoader(train_data, batch_size=batch_size, shuffle=True)

def train_loop(num_epochs, model_save_path, model_name, pth_tokenizer, dataset_path, batch_size, initial_lr, max_seq_length):
    model = load_model(model_name, pth_tokenizer, max_seq_length).to("cuda")
    data_loader = load_data(dataset_path)
    print("train_loop")

    # Initialize WandB
    wandb.init(project="triplet-loss-experiment2", entity="igor-sondors")
    wandb.config.update({
        "model_name": model_name,
        "dataset_path": dataset_path,
        "model_save_path": model_save_path,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": initial_lr,
        "max_seq_length": max_seq_length
    })

    train_loss = TripletLoss()#CustomTripletLoss()
    optimizer = AdamW(model.parameters(), lr=initial_lr)  # Using AdamW optimizer
    
    # Initialize the learning rate scheduler
    total_steps = len(data_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    wandb.watch(model, log="all")

    for epoch in range(num_epochs):
        # print(f"epoch: {epoch}")
        model.train()
        total_loss = 0
        batch_counter = 0
        epoch_time = time.time()
        for batch in data_loader:
            batch_counter += 1
            # batch_time = time.time()
            optimizer.zero_grad()

            anchor_batch, positive_batch, negative_batch = batch
            anchor_embeddings = model.encode(anchor_batch, convert_to_tensor=True, batch_size=len(anchor_batch))
            positive_embeddings = model.encode(positive_batch, convert_to_tensor=True, batch_size=len(positive_batch))
            negative_embeddings = model.encode(negative_batch, convert_to_tensor=True, batch_size=len(negative_batch))
            
            anchor_embeddings.requires_grad = True
            positive_embeddings.requires_grad = True
            negative_embeddings.requires_grad = True

            loss = train_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
            # print(f"Batch [{batch_counter}/{len(data_loader)}], Loss: {loss:.4f}, batch time: {round(time.time() - batch_time, 2)} sec")

            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate
            
            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        model.save(model_save_path+f"/epoch_{epoch}")
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}, epoch time: {round(time.time() - epoch_time, 2)} sec')
        print("_"*40,"\n")
        wandb.log({"epoch": epoch, "avg_loss": avg_loss, "learning_rate": optimizer.param_groups[0]['lr']})
    
    # Save the model
    model.save(model_save_path)
    wandb.save(model_save_path)

if __name__ == "__main__":
    max_seq_length = 512
    initial_lr = 0.001#2e-5
    batch_size = 1000
    num_epochs = 30
    model_name = "/mnt/vdb1/BERT_training/sbert_approach/mlm_weights/experiment_part_of_word_masking/2023-07-14_14-52-52/checkpoint-360000"
    pth_tokenizer = "/mnt/vdb1/BERT_training/sbert_approach/mlm_weights/experiment_part_of_word_masking/2023-07-14_14-52-52"
    dataset_path = "/mnt/vdb1/Datasets/triplets_train_aug_clear.csv"
    # dataset_path = "/mnt/vdb1/Datasets/triplets_train_aug.csv"
    # dataset_path = "/mnt/vdb1/Datasets/triplets_train.csv"
    # dataset_path = "/mnt/vdb1/Datasets/triplets_test.csv"
    model_save_path = '/mnt/vdb1/BERT_training/sbert_approach/matcher/rubert-tiny2-custom'
    
    train_loop(num_epochs, model_save_path, model_name, pth_tokenizer, dataset_path, batch_size, initial_lr, max_seq_length)