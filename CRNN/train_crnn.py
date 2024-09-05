import os
import torch
import logging
from torch.utils.data import DataLoader, random_split, ConcatDataset
import torch.optim as optim
from torch.nn import CTCLoss
from EDA import ImageDataset1, ImageDataset2, AugmentedDataset
from model import CRNN
from config import train_config as config
from torchvision import transforms
import matplotlib.pyplot as plt
    

def train_batch(crnn, data, optimizer, criterion, device):
    crnn.train()
    images, targets = [d.to(device) for d in data]


    logits_seq = crnn(images)   
    logits = logits_seq[-1]

    loss = criterion(logits, targets)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(crnn.parameters(), 5)  # Gradient clipping with 5
    optimizer.step()

    _, preds = torch.max(logits, 1)
    corrects = torch.sum(preds == targets).item()
    accuracy = corrects / len(targets)

    
    return loss.item(), accuracy

 




def validate(crnn, valid_loader, criterion, device):
    crnn.eval()
    tot_valid_loss = 0
    tot_valid_count = 0
    correct_predictions = 0

    with torch.no_grad():
        for valid_data in valid_loader:
            images, targets = [d.to(device) for d in valid_data]
            logits_seq = crnn(images)   # [seq_length, batch_size, num_classes]

            logits = torch.mean(logits_seq, dim=0)  # Averaging across sequences

            loss = criterion(logits, targets)
            batch_size = images.size(0)

            tot_valid_loss += loss.item() * batch_size
            tot_valid_count += batch_size

            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == targets).sum().item()

    avg_valid_loss = tot_valid_loss / tot_valid_count
    avg_valid_acc = correct_predictions / tot_valid_count * 100
    return avg_valid_loss, avg_valid_acc

def main():
    epochs = config['epochs']
    train_batch_size = config['train_batch_size']
    eval_batch_size = config['eval_batch_size']
    lr = config['lr']
    show_interval = config['show_interval']
    cpu_workers = config['cpu_workers']

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    # Initialize datasets with the new classes
    # hindi_syn = ImageDataset1(config['hindi_path_syn'], label=0, max_images=config['max_images_syn'], transform=None)
    # english_syn = ImageDataset1(config['english_path_syn'], label=1, max_images=config['max_images_syn'], transform=None)
    # guj_syn = ImageDataset2(config['gujarati_path_syn'], label=2, max_images=config['max_images_syn'], transform=None)
    hindi_real = ImageDataset2(config['hindi_path_real'], label=0, max_images=config['max_images_real'], transform=None)
    english_real = ImageDataset2(config['english_path_real'], label=1, max_images=config['max_images_real'], transform=None)
    # guj_real = ImageDataset2(config['gujarati_path_real'], label=2, max_images=config['max_images_real'], transform=None)
    pun_real = ImageDataset2(config['punjabi_path_real'], label=2, max_images=config['max_images_real'], transform=None)

    # Combine datasets
    
    full_dataset = ConcatDataset([hindi_real, english_real,pun_real])
    # full_dataset = ConcatDataset([hindi_syn,english_syn,hindi_real,english_real])
    # full_dataset = ConcatDataset([hindi_real,english_real,guj_real,hindi_syn,guj_syn,english_syn])

    

# 
    # Split dataset into training and validation sets
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Define image transforms
    train_transforms = transforms.Compose([
        transforms.Resize((32, 64)),
        transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((32, 64)),
        transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply transformations to datasets
    train_dataset = AugmentedDataset(train_dataset, augmentations=None, post_transforms=train_transforms)
    val_dataset = AugmentedDataset(val_dataset, post_transforms=val_transforms)

    # DataLoaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=cpu_workers,
    )
    valid_loader = DataLoader(
        dataset=val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=cpu_workers,
    )

    num_class = config['classes']  # 3 classes for script identification
    # passing 3 channels
    crnn = CRNN(3, 32, 64, num_class,  
                map_to_seq_hidden=config['map_to_seq_hidden'],
                rnn_hidden=config['rnn_hidden'],
                leaky_relu=config['leaky_relu'])


    crnn.to(device)

    optimizer = optim.RMSprop(crnn.parameters(), lr=lr)
    # criterion = CTCLoss(reduction='sum', zero_infinity=True)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    
    best_valid_loss = float('inf')
    patience = config['patience']
    patience_counter = 0
    i = 1

    
    train_losses = []
    valid_losses = []
    train_accuracies = []   
    valid_accuracies = []

    for epoch in range(1, epochs + 1):
        print(f'epoch: {epoch}')
        tot_train_loss = 0.
        tot_train_acc = 0.
        tot_train_count = 0
        i = 0  # Initialize batch index

        for train_data in train_loader:
            loss, accuracy = train_batch(crnn, train_data, optimizer, criterion, device)
            train_size = len(train_data[0])

            tot_train_loss += loss
            tot_train_acc += accuracy * train_size
            tot_train_count += train_size

            if i % show_interval == 0:
                print(f'train_batch_loss[{i}]: {loss}')

            i += 1

        train_loss = tot_train_loss / len(train_loader)
        train_accuracy = 100 * tot_train_acc / tot_train_count
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        print(f'train_loss: {train_loss}, train_accuracy: {train_accuracy}')

        # Validation
        avg_valid_loss, avg_valid_acc = validate(crnn, valid_loader, criterion, device)
        print(f'valid_evaluation: loss={avg_valid_loss}, acc={avg_valid_acc}')

        valid_losses.append(avg_valid_loss)
        valid_accuracies.append(avg_valid_acc)

        # Check for early stopping
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            patience_counter = 0  # Reset counter if validation loss improves
            # Optionally, save the best model
            # torch.save(crnn.state_dict(), 'best_crnn_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered. Stopping training.")
                break  # Exit the loop early
            
    # Save the final model
    final_model_path = os.path.join(config['checkpoints_dir'], 'crnn_real(2)_t2.pt')
    torch.save(crnn.state_dict(), final_model_path)
    print('Saved final model at', final_model_path)




    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    loss_plot_path = os.path.join(config['checkpoints_dir'], 'loss_real_1.png')
    plt.savefig(loss_plot_path)
    plt.close()

    # Plotting and saving the validation accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(valid_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    acc_plot_path = os.path.join(config['checkpoints_dir'], 'accu_real_1.png')
    plt.savefig(acc_plot_path)
    plt.close()




if __name__ == '__main__':
    main()
