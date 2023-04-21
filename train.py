import os
import sys
import json
import time
import pickle
from tqdm import tqdm
from glob import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import RobertaForSequenceClassification
from torch.utils.data import DataLoader
from data import LabeledDataset, UnlabeledDataset


CHECKPOINT = 'roberta-large'
NUM_LABELS = 10 # Number of labels in Yahoo
PAD_token = 1 # RoBERTa 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GPU = torch.cuda.get_device_name() if torch.cuda.is_available() else None

def get_unsupervised_loss(model, unlabeled_batch, unsupervised_criterion, config):

    input_ids = unlabeled_batch[0].to(DEVICE)
    attention_mask = unlabeled_batch[1].to(DEVICE)
    aug_input_ids = unlabeled_batch[2].to(DEVICE)
    aug_attention_mask = unlabeled_batch[3].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        prob = F.softmax(outputs.logits, dim = -1) # KLdiv target

        # confidence-based masking
        unsup_loss_mask = torch.max(prob, dim=-1)[0] > config['uda_confidence_thresh']
        unsup_loss_mask = unsup_loss_mask.type(torch.float32).to(DEVICE)

    aug_outputs = model(aug_input_ids, attention_mask=aug_attention_mask)

    # softmax temperature controlling
    uda_softmax_temp = config['uda_softmax_temp'] if config['uda_softmax_temp'] > 0 else 1.
    aug_log_prob = F.log_softmax(aug_outputs.logits / uda_softmax_temp, dim = -1)

    loss = torch.sum(unsupervised_criterion(aug_log_prob, prob), dim=-1)
    loss = torch.sum(loss * unsup_loss_mask, dim=-1) / torch.max(torch.sum(unsup_loss_mask, dim=-1), torch.tensor([1.]).to(DEVICE))

    return loss

# TSA
def get_tsa_thresh(schedule, current_epoch, total_epochs, start, end = 1):
    training_progress = torch.tensor(float(current_epoch) / float(total_epochs))
    if schedule == 'linear_schedule':
        threshold = training_progress
    elif schedule == 'exp_schedule':
        scale = 5
        threshold = torch.exp((training_progress - 1) * scale)
    elif schedule == 'log_schedule':
        scale = 5
        threshold = 1 - torch.exp((-training_progress) * scale)
    output = threshold * (end - start) + start
    return output.to(DEVICE)

def get_supervised_loss(model, labeled_batch, supervised_criterion, current_epoch, config):

    input_ids = labeled_batch[0].to(DEVICE)
    attention_mask = labeled_batch[1].to(DEVICE)
    labels = labeled_batch[2].to(DEVICE)

    outputs = model(input_ids, attention_mask=attention_mask)
    loss = supervised_criterion(outputs.logits, labels)

    tsa_thresh = get_tsa_thresh(config['tsa'], current_epoch, config['epochs'], start=1./outputs.logits.shape[-1])
    smaller_than_threshold = torch.exp(-loss) <= tsa_thresh 

    loss_mask = torch.ones_like(labels, dtype=torch.float32) * smaller_than_threshold.type(torch.float32)
    loss = torch.sum(loss * loss_mask, dim=-1) / torch.max(torch.sum(loss_mask, dim=-1), torch.tensor([1.]).to(DEVICE))

    return loss

def train_epoch(model, labeled_train_loader, unlabeled_train_loader, optimizer, 
                supervised_criterion, unsupervised_criterion, epoch, config):

    model.train()

    final_losses, sup_losses, unsup_losses = [], [], []
    for unlabeled_batch in tqdm(unlabeled_train_loader):

        labeled_batch = next(labeled_train_loader)

        optimizer.zero_grad()

        supervised_loss = get_supervised_loss(model, labeled_batch, supervised_criterion,
                                              epoch, config)
        unsupervised_loss = get_unsupervised_loss(model, unlabeled_batch, unsupervised_criterion, config)

        loss = supervised_loss + (config['uda_coeff'] * unsupervised_loss)

        final_losses.append(loss)
        sup_losses.append(supervised_loss)
        unsup_losses.append(unsupervised_loss)
        
        loss.backward()
        optimizer.step()

    return final_losses, sup_losses, unsup_losses

def evaluate(model, val_loader, criterion):

    model.eval()
    val_loss, val_acc = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch[0].to(DEVICE)
            attention_mask = batch[1].to(DEVICE)
            labels = batch[2].to(DEVICE)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += criterion(outputs.logits, labels).item()
            val_acc += (outputs.logits.argmax(dim=1) == labels).sum().item()

    return val_loss/len(val_loader), val_acc/len(val_loader.dataset)

def train(model, labeled_train_loader, unlabeled_train_loader, val_loader, config, results_path):

    # Define the optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=5e-6)
    supervised_criterion = nn.CrossEntropyLoss(reduction='none')
    unsupervised_criterion = nn.KLDivLoss(reduction='none')
    val_criterion = nn.CrossEntropyLoss()

    with open(os.path.join(results_path, "history.pkl"), 'wb') as f:
        logs = {'final_losses': [],
                'supervised_losses': [],
                'unsupervised_losses': [],
                'val_losses': [],
                'val_accuracy': []}
        pickle.dump(logs, f)


    model.to(DEVICE)
    for epoch in range(config['epochs']):

        final_losses, sup_losses, unsup_losses = train_epoch(model, labeled_train_loader, unlabeled_train_loader, optimizer, 
                                                             supervised_criterion, unsupervised_criterion, epoch + 1, config)
        
        
        # evaluate the model on the validation set
        val_loss, val_acc = evaluate(model, val_loader, val_criterion)

        logs = pickle.load(open(os.path.join(results_path, "history.pkl"), 'rb'))
        logs['final_losses'].extend(final_losses)
        logs['supervised_losses'].extend(sup_losses)
        logs['unsupervised_losses'].extend(unsup_losses)
        logs['val_losses'].append(val_loss)
        logs['val_accuracy'].append(val_acc)
        pickle.dump(logs, open(os.path.join(results_path, "history.pkl"), 'wb'))

        torch.save(model.state_dict(), os.path.join(results_path, f"epoch_{epoch + 1}.pt"))

        # print(f'Epoch {epoch+1}: val_loss={val_loss:.4f}, val_acc={val_acc:.4f}')

def main(config_name):

    with open(os.path.join("configs", f"{config_name}.json"), "r") as f:
        config = json.load(f)

    # Load the pretrained model
    model = RobertaForSequenceClassification.from_pretrained(CHECKPOINT, num_labels = NUM_LABELS)

    # Indices name
    train_labeled_idx_name = config['train_labeled_idx_name'] # "msr_30_equal_100_labeled_indices.pt"
    train_unlabeled_idx_name = config['train_unlabeled_idx_name'] # "msr_30_idr_5_50000_unlabeled_indices.pt"

    # Creating training and validation datasets
    labeled_train = LabeledDataset(train_labeled_idx_name)
    unlabeled_train = UnlabeledDataset(train_unlabeled_idx_name)

    val_dataset = LabeledDataset()

    # Creating Dataloaders 
    labeled_batch_size = get_batch_size(len(labeled_train), config)
    labeled_train_loader = DataLoader(labeled_train, 
                                      batch_size = labeled_batch_size, 
                                      shuffle = True,
                                      collate_fn = collate_batch)
    labeled_train_loader = repeat_dataloader(labeled_train_loader)
    
    unlabeled_batch_size = get_batch_size(len(unlabeled_train), config)
    unlabeled_train_loader = DataLoader(unlabeled_train, 
                                        batch_size = unlabeled_batch_size, 
                                        shuffle = True,
                                        collate_fn = collate_batch)
    
    val_loader = DataLoader(val_dataset, 
                            batch_size = config['val_batch_size'], 
                            shuffle=False,
                            collate_fn = collate_batch)
    
    config_results_path = os.path.join("results", config_name)
    os.makedirs(config_results_path, exist_ok=True)

    num_results = len(glob(os.path.join(config_results_path, f"run_*")))
    run_results_path = os.path.join(config_results_path, f"run_{num_results + 1}")
    os.makedirs(run_results_path)

    logger = open(os.path.join(run_results_path,'std.log'),'w')

    logger.write(f"Labeled Batch Size = {labeled_batch_size}\n")

    num_labeled_one_epoch = labeled_batch_size * (len(unlabeled_train) // unlabeled_batch_size) / len(labeled_train)
    logger.write(f"\nNumber of epochs through labeled data = {config['epochs'] * num_labeled_one_epoch}")

    logger.write(f"\nUnlabeled Batch Size = {unlabeled_batch_size}")
    logger.write(f"\nNumber of epochs through labeled data = {config['epochs']}\n")

        
    # train the model
    start_time = time.time()
    train(model, labeled_train_loader, unlabeled_train_loader, val_loader, config, run_results_path,logger)
    end_time = time.time()

    training_time_hours, training_time_minutes = divmod(end_time - start_time, 3600)

    logger.write(f"\nFinished training in: {int(training_time_hours)} hours, {int(training_time_minutes)} minutes")
    logger.close()
    
def collate_batch(batch):
    """
    Labeled batch: input_ids, attention_mask, labels
    Unlabeled batch: input_ids, attention_mask, aug_input_ids, aug_attention_mask
    """
    if len(batch[0]) == 3:

        input_ids, attention_mask, labels = [], [], []
        for (_input, _mask, _label) in batch:
            input_ids.append(_input)
            attention_mask.append(_mask)
            labels.append(_label)
        
        input_ids = pad_sequence(input_ids, batch_first = True, padding_value = PAD_token)
        attention_mask = pad_sequence(attention_mask, batch_first = True, padding_value = 0)
                
        return input_ids, attention_mask, torch.tensor(labels)
    
    else:

        input_ids, attention_mask, aug_input_ids, aug_attention_mask = [], [], [], []
        for (_input, _mask, _aug_input, _aug_mask) in batch:
            input_ids.append(_input)
            attention_mask.append(_mask)
            aug_input_ids.append(_aug_input)
            aug_attention_mask.append(_aug_mask)
        
        input_ids = pad_sequence(input_ids, batch_first = True, padding_value = PAD_token)
        attention_mask = pad_sequence(attention_mask, batch_first = True, padding_value = 0)
        aug_input_ids = pad_sequence(aug_input_ids, batch_first = True, padding_value = PAD_token)
        aug_attention_mask = pad_sequence(aug_attention_mask, batch_first = True, padding_value = 0)
                
        return input_ids, attention_mask, aug_input_ids, aug_attention_mask

def get_batch_size(num_samples, config):

    if GPU:
        if GPU == 'Quadro RTX 8000': config['max_batch_size'] = 8
        elif GPU == 'NVIDIA A100-SXM4-80GB': config['max_batch_size'] = 12
    
    batch_size = min(num_samples//config['steps_per_epoch'], config['max_batch_size'])
    batch_size = max(batch_size, config['min_batch_size'])
    return batch_size

def repeat_dataloader(dataloader):
    
    while True:
        for x in dataloader:
            yield x
    
if __name__ == '__main__':
    
    main(sys.argv[1])
