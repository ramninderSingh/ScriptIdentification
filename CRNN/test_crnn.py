import torch
from EDA import ImageDataset1, ImageDataset2, AugmentedDataset
from torch.utils.data import DataLoader, random_split, ConcatDataset
from config import test_config as config
from model import CRNN
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, f1_score



def test_model(crnn, test_loader, device):
    crnn.eval()
    correct_predictions = 0
    total_count = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for test_data in test_loader:
            images, targets = [d.to(device) for d in test_data]
            logits_seq = crnn(images)
            logits = torch.mean(logits_seq, dim=0)

            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            correct_predictions += (predictions == targets).sum().item()

            total_count += images.size(0)

    avg_test_acc = correct_predictions / total_count

    # precision = precision_score(all_targets, all_predictions, average='weighted')
    # recall = recall_score(all_targets, all_predictions, average='weighted')
    # f1 = f1_score(all_targets, all_predictions, average='weighted')

    precision = precision_score(all_targets, all_predictions, average='macro')
    recall = recall_score(all_targets, all_predictions, average='macro')
    f1 = f1_score(all_targets, all_predictions, average='macro')

    # # # class wise
    precision_c = precision_score(all_targets, all_predictions, average=None)
    recall_c = recall_score(all_targets, all_predictions, average=None)
    f1_c = f1_score(all_targets, all_predictions, average=None)

    print(precision_c)
    print(recall_c)
    print(f1_c)

    print(f'Average Test Accuracy: {avg_test_acc * 100:.2f}%')
    print(f'Precision: {precision * 100:.2f}%')
    print(f'Recall: {recall * 100:.2f}%')
    print(f'F1 Score: {f1 * 100:.2f}%')

    return avg_test_acc, precision, recall, f1



    return avg_test_acc



def main():
    cpu_workers = config['cpu_workers']
 
    batch_size=config['test_batch_size']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    transform = transforms.Compose([
    transforms.Resize((32, 64)),
    transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    ])


    # hindi_train_real = ImageDataset2(config['hindi_path_real_train'], label=0, max_images=config['max_images'], transform=transform)
    # english_train_real = ImageDataset2(config['english_path_real_train'], label=1, max_images=config['max_images'], transform=transform)
    # guj_train_real = ImageDataset2(config['gujarati_path_real_train'], label=2, max_images=config['max_images'], transform=transform)
    # pun_train_real = ImageDataset2(config['punjabi_path_real_train'], label=2, max_images=config['max_images'], transform=transform)

    hindi_real = ImageDataset2(config['hindi_path'], label=0, max_images=config['max_images'], transform=transform)
    english_real = ImageDataset2(config['english_path'], label=1, max_images=config['max_images'], transform=transform)
    pun_real = ImageDataset2(config['punjabi_path'], label=2, max_images=config['max_images'], transform=transform)
    # guj_real = ImageDataset2(config['gujarati_path'], label=2, max_images=config['max_images'], transform=transform)

    test_dataset = ConcatDataset([hindi_real,english_real,pun_real])
    # test_dataset = ConcatDataset([hindi_real,english_real,pun_real,hindi_train_real,english_train_real,pun_train_real])
    


    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=cpu_workers)

    num_class = config['classes']
    crnn = CRNN(3, 32, 64, num_class,   
                map_to_seq_hidden=config['map_to_seq_hidden'],
                rnn_hidden=config['rnn_hidden'],
                leaky_relu=config['leaky_relu'])


    crnn.load_state_dict(torch.load(config['reload_model'],map_location=device))
    
    crnn.to(device)


    test_model(crnn, test_loader, device)

    



if __name__ == '__main__':
    main()
