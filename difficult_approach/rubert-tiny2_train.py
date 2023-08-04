from transformers import BertForPreTraining, BertTokenizerFast, BertConfig, DataCollatorForWholeWordMask

import torch
import torch.nn as nn

from collections import Counter
from tqdm.auto import tqdm, trange

import pandas as pd
import time
import gc

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def get_mask_labels(input_ids):
    data_collator = DataCollatorForWholeWordMask(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    mask_labels = []
    for e in input_ids:
        ref_tokens = []
        for idx in e:
            token = tokenizer._convert_id_to_token(idx)
            ref_tokens.append(token)
        mask_labels.append(data_collator._whole_word_mask(ref_tokens))
    ml = torch.tensor(mask_labels)
    inputs, labels = data_collator.torch_mask_tokens(input_ids, ml)
    return inputs, labels

def preprocess_inputs(inputs):
    inputs['input_ids'], inputs['labels'] = get_mask_labels(inputs['input_ids'])
    return {k: v.to(model.device) for k, v in inputs.items()}

def get_mlm_loss(inputs, outputs):
    return nn.CrossEntropyLoss()(
        outputs.prediction_logits.view(-1, model.config.vocab_size),
        inputs['labels'].view(-1)
    )

##############################################################

def upd_small_model(base_model, NEW_MODEL_NAME, df):
    print("Сохраняем копию модели для обучения")
    tok = BertTokenizerFast.from_pretrained(base_model)
    cnt_ru = Counter()
    for text in tqdm(df.name):
        cnt_ru.update(tok(text)['input_ids'])
        
    resulting_vocab = {
        tok.vocab[k] for k in tok.special_tokens_map.values()
    }
    for k, v in cnt_ru.items():
        if v >= 5 or k <= 3_000:
            resulting_vocab.add(k)

    resulting_vocab = sorted(resulting_vocab)
    print(len(resulting_vocab))   

    tok.save_pretrained(NEW_MODEL_NAME)
    new_tokenizer = BertTokenizerFast.from_pretrained(NEW_MODEL_NAME)

    small_config = BertConfig(
        emb_size=312,
        hidden_size=312,
        intermediate_size=600,
        max_position_embeddings=512,
        num_attention_heads=12,
        num_hidden_layers=3,
        vocab_size=new_tokenizer.vocab_size,
    )

    small_model = BertForPreTraining(small_config)
    small_model.save_pretrained(NEW_MODEL_NAME)

    #Выкачиваем веса из большой модели для инициализации
    big_model = BertForPreTraining.from_pretrained(base_model)
    # copy input embeddings
    small_model.bert.embeddings.word_embeddings.weight.data = big_model.bert.embeddings.word_embeddings.weight.data[resulting_vocab, :312].clone()
    small_model.bert.embeddings.position_embeddings.weight.data = big_model.bert.embeddings.position_embeddings.weight.data[:, :312].clone()
    # copy output embeddings
    small_model.cls.predictions.decoder.weight.data = big_model.cls.predictions.decoder.weight.data[resulting_vocab, :312].clone()
    small_model.save_pretrained(NEW_MODEL_NAME)
    

def training_loop(model, tokenizer, df, batch_size, accumulation_steps, lr, epochs):
    
    save_steps = int(8192 / batch_size)
    window = int(1024 / batch_size * 4)
    print('window steps', window, 'save steps', save_steps)
    ewms = [0] * 20

    tq = trange(int(df.shape[0] * epochs / batch_size))

    optimizer = torch.optim.Adam(
        params=model.parameters(), 
        lr=lr)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=1765)

    model.train()

    for i in tq:
        bb = df.name.sample(batch_size)

        inputs_ru = preprocess_inputs(tokenizer(bb.tolist(), return_tensors='pt', padding=True, truncation=True))
        outputs_ru = model(**inputs_ru, output_hidden_states=True)
        
        losses = [
            get_mlm_loss(inputs_ru, outputs_ru)
        ]
        loss = sum(losses)
        loss.backward()


        w = 1 / min(i+1, window)
        ewms = [ewm * (1-w) + loss.item() * w for ewm, loss in zip(ewms, [loss] + losses)]
        desc = 'loss: ' + ' '.join(['{:2.2f}'.format(l) for l in ewms]) + '|{:2.1e}'.format(optimizer.param_groups[0]['lr'])
        tq.set_description(desc)

        if i % accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            
            optimizer.zero_grad()
            cleanup()
        
        if i % window == 0 and i > 0:
            print(desc)
            # cleanup()

        if i % save_steps == 0 and i > 0:
            model.save_pretrained(NEW_MODEL_NAME+'_ckpt')
            tokenizer.save_pretrained(NEW_MODEL_NAME+'_ckpt')
            print('saving...', i, optimizer.param_groups[0]['lr'])

if __name__=="__main__":

    NEW_MODEL_NAME = '/mnt/vdb1/BERT_training/rubert-tiny2-price'
    base_model = '/mnt/vdb1/BERT_training/rubert-tiny2'
    corpus_path = '/mnt/vdb1/14_categories_balanced.csv'

    batch_size = 32  
    accumulation_steps = 4  # эта штука реально помогает, когда обучение подзастряло. А ещё ускоряет!
    lr = 1e-5
    epochs = 50

    print(f"batch_size = {batch_size}\naccumulation_steps = {accumulation_steps}\nlr = {lr}\nepochs = {epochs}")
    df = pd.read_csv(corpus_path, sep=';')
    df = df.drop(columns=['category_id', 'model_id', 'attrs', 'price', 'description'])
    df = df.drop(columns=['external_category', 'external_brand', 'external_type'])

    upd_small_model(base_model, NEW_MODEL_NAME, df)# сохр модель которую буду обучать

    model = BertForPreTraining.from_pretrained(NEW_MODEL_NAME, ignore_mismatched_sizes=True).cuda()
    tokenizer = BertTokenizerFast.from_pretrained(NEW_MODEL_NAME)

    start_time = time.time()
    training_loop(model, tokenizer, df, batch_size, accumulation_steps, lr, epochs)
    time_spent = time.time() - start_time
    print(f"Обучение завершено\nвремя: {time_spent}\nвремя на одну эпоху: {time_spent/epochs}\n")
    print(f"batch_size = {batch_size}\naccumulation_steps = {accumulation_steps}\nlr = {lr}\nepochs = {epochs}")

    