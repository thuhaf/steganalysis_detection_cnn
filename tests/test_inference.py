import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import os
from tqdm import tqdm

from datasets import ImageStegDataset
from models import get_model

CONFIG = {
    'test_cover_dir': './data/test/cover',
    'test_stego_dir': './data/test/stego',
    'model_name': 'deep_steg_cnn',
    'checkpoint_path': './checkpoints/best_model.pth',
    'batch_size': 32,
    'image_size': 256,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def load_trained_model(config):
    model = get_model(config['model_name'], num_classes=2, pretrained=False)
    
    if os.path.exists(config['checkpoint_path']):
        checkpoint = torch.load(config['checkpoint_path'], map_location=config['device'])
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        raise FileNotFoundError(f"Checkpoint not found at {config['checkpoint_path']}")
    
    model.to(config['device'])
    model.eval()
    return model

def get_test_dataloader(config):
    test_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
    ])

    test_dataset = ImageStegDataset(
        cover_dir=config['test_cover_dir'],
        stego_dir=config['test_stego_dir'],
        transform=test_transform,
        split='test'
    )
    
    return DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=2
    )

def evaluate_model(model, dataloader, device):
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return np.array(all_labels), np.array(all_preds)

def visualize_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Cover', 'Stego'],
                yticklabels=['Cover', 'Stego'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()

def main():
    try:
        model = load_trained_model(CONFIG)
    except Exception as e:
        print(e)
        return

    test_loader = get_test_dataloader(CONFIG)
    y_true, y_pred = evaluate_model(model, test_loader, CONFIG['device'])
    
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Cover', 'Stego']))
    
    visualize_confusion_matrix(y_true, y_pred)
    print("Confusion Matrix saved to confusion_matrix.png")

if __name__ == '__main__':
    main()
