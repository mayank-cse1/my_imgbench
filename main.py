from my_imgbench import eda, models, train_eval
import torch.nn as nn
import torch.optim as optim

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_dir, test_dir = "data/train", "data/test"
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

train_ds = datasets.ImageFolder(train_dir, transform=transform)
test_ds = datasets.ImageFolder(test_dir, transform=transform)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# EDA
eda.plot_class_distribution(train_ds)
eda.show_sample_images(train_ds)

# # Model training and evaluation loop
for arch_fn in [models.BasicCNN]:
    print(f"Training {arch_fn.__name__}")
    model = arch_fn(num_classes=len(train_ds.classes)) if arch_fn == models.BasicCNN else arch_fn(len(train_ds.classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    train_eval.train_model(model, {'train': train_loader, 'test': test_loader}, criterion, optimizer, device)
    train_eval.evaluate_model(model, test_loader, device, train_ds.classes)
