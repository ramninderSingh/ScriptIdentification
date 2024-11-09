from sklearn.model_selection import train_test_split
from datasets import DatasetDict,Dataset,ClassLabel
from evaluate import load as load_metric
from EDA import TrainEDA
from torch.utils.data import DataLoader, random_split, ConcatDataset
from transformers import AutoImageProcessor,ViTForImageClassification,TrainingArguments,Trainer
from config import train_config as config
import os
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
# from tensorboard.backend.event_processing import event_accumulator
from math import ceil


def transform(example_batch):
    transform = transforms.Resize((224, 224)) 

    resized_images = [transform(x.convert("RGB")) for x in example_batch['image']]

    inputs=processor(resized_images,return_tensors='pt')

    inputs['label']=example_batch['label']

    return inputs


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }

def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


model_name='google/vit-base-patch16-224-in21k'

processor=AutoImageProcessor.from_pretrained(model_name,use_fast=True)





data_folders = {
    'hindi': config['hindi_path_real'],
    'english': config['english_path_real'],
    'gujarati': config['gujarati_path_real'],
    'punjabi':config['punjabi_path_real'],
    'assamese':config['assamese_path_real'],
    'bengali':config['bengali_path_real'],
    'kannada':config['kannada_path_real'],
    'malayalam':config['malayalam_path_real'],
    'marathi':config['marathi_path_real'],
    'odia':config['odia_path_real'],
    'tamil':config['tamil_path_real'],
    'telugu':config['telugu_path_real'],


}

print(data_folders.keys())

class_labels = ClassLabel(names=list(data_folders.keys()))  



all_images = []
for label, folder in data_folders.items():
    all_images.extend(TrainEDA(folder,label,max_images=config['max_images_real']))



dataset = Dataset.from_list(all_images)
dataset = dataset.cast_column("label", class_labels)


ds = dataset.train_test_split(test_size=0.2, stratify_by_column="label")



dataset_dict = DatasetDict({
    'train': ds["train"],
    'val': ds["test"]
})


print(dataset_dict)
print(dataset_dict['train'][0])  
print(dataset_dict['val'][0])  


labels=dataset_dict["train"].features['label']
print(labels)




prepared_ds=dataset_dict.with_transform(transform)

metric=load_metric("accuracy")



model=ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=len(labels.names),
    id2label={str(i): c for i, c in enumerate(labels.names)},
    label2id={c:str(i) for i,c in enumerate(labels.names)},
    ignore_mismatched_sizes=True
)


root_dir = config['checkpoints_dir']  # Path where all config files and checkpoints will be saved
training_args = TrainingArguments(
  output_dir=root_dir,
  per_device_train_batch_size=16,
  evaluation_strategy="epoch",
  save_strategy="epoch",
  fp16=True,
  num_train_epochs=config['epoch'],
  learning_rate=2e-4,
  save_total_limit=1,
  remove_unused_columns=False,
  push_to_hub=False,
  report_to='tensorboard',
  logging_dir='./logs', 
  load_best_model_at_end=True,
)

steps_per_epoch = ceil(len(prepared_ds["train"]) / training_args.per_device_train_batch_size)
training_args.logging_steps = steps_per_epoch  # Set to log at end of each epoch


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["val"],
    tokenizer=processor,
)



save_dir = os.path.join(config['checkpoints_dir'],'best_model')

train_results = trainer.train()
trainer.save_model(save_dir)  # Save the best model
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()




train_loss = []
eval_loss = []
epochs = []

for log in trainer.state.log_history:
    if 'loss' in log.keys() and 'epoch' in log.keys():
        train_loss.append(log['loss'])
        epochs.append(log['epoch'])
        print(log['epoch'])
    if 'eval_loss' in log.keys():
        eval_loss.append(log['eval_loss'])


min_len = min(len(epochs), len(train_loss), len(eval_loss))
epochs = epochs[:min_len]
train_loss = train_loss[:min_len]
eval_loss = eval_loss[:min_len]



# Plotting the loss vs. epochs
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, label="Training Loss", marker='o')
plt.plot(epochs, eval_loss, label="Validation Loss", marker='o')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs. Epochs")
plt.legend()
plt.grid(True)
output_path=f"{root_dir}/loss_vs_epoch.png"
plt.savefig(output_path,format='png')
