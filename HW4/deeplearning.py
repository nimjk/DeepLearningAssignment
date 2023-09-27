# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import DataParallel
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import timm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
# %%
high_dir = '/Users/nimjk/OneDrive/Desktop/Ryzen/mltest/High'
low_dir = '/Users/nimjk/OneDrive/Desktop/Ryzen/mltest/Low'
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
criterion = nn.CrossEntropyLoss()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
# %%
HQF_train= datasets.ImageFolder(root=high_dir + '/f2f_data/train', transform=transform)
HQF_test = datasets.ImageFolder(root=high_dir + '/f2f_data/test', transform=transform)
HQF_val = datasets.ImageFolder(root=high_dir + '/f2f_data/val', transform=transform)

batch_size = 32
HQF_train_loader = torch.utils.data.DataLoader(HQF_train, batch_size=batch_size, shuffle=True, num_workers=4)
HQF_test_loader = torch.utils.data.DataLoader(HQF_test, batch_size=batch_size, shuffle=False, num_workers=4)
HQF_val_loader = torch.utils.data.DataLoader(HQF_val, batch_size=batch_size, shuffle=False, num_workers=4)
# %%
LQF_train= datasets.ImageFolder(root=low_dir + '/f2f_data/train', transform=transform)
LQF_test = datasets.ImageFolder(root=low_dir + '/f2f_data/test', transform=transform)
LQF_val = datasets.ImageFolder(root=low_dir + '/f2f_data/val', transform=transform)

batch_size = 32
LQF_train_loader = torch.utils.data.DataLoader(LQF_train, batch_size=batch_size, shuffle=True, num_workers=4)
LQF_test_loader = torch.utils.data.DataLoader(LQF_test, batch_size=batch_size, shuffle=False, num_workers=4)
LQF_val_loader = torch.utils.data.DataLoader(LQF_val, batch_size=batch_size, shuffle=False, num_workers=4)
# %%
HQN_train= datasets.ImageFolder(root=high_dir + '/nt_data/train', transform=transform)
HQN_test = datasets.ImageFolder(root=high_dir + '/nt_data/test', transform=transform)
HQN_val = datasets.ImageFolder(root=high_dir + '/nt_data/val', transform=transform)

batch_size = 32
HQN_train_loader = torch.utils.data.DataLoader(HQN_train, batch_size=batch_size, shuffle=True, num_workers=4)
HQN_test_loader = torch.utils.data.DataLoader(HQN_test, batch_size=batch_size, shuffle=False, num_workers=4)
HQN_val_loader = torch.utils.data.DataLoader(HQN_val, batch_size=batch_size, shuffle=False, num_workers=4)
# %%
LQN_train= datasets.ImageFolder(root=low_dir + '/nt_data/train', transform=transform)
LQN_test = datasets.ImageFolder(root=low_dir + '/nt_data/test', transform=transform)
LQN_val = datasets.ImageFolder(root=low_dir + '/nt_data/val', transform=transform)

batch_size = 32
LQN_train_loader = torch.utils.data.DataLoader(LQN_train, batch_size=batch_size, shuffle=True, num_workers=4)
LQN_test_loader = torch.utils.data.DataLoader(LQN_test, batch_size=batch_size, shuffle=False, num_workers=4)
LQN_val_loader = torch.utils.data.DataLoader(LQN_val, batch_size=batch_size, shuffle=False, num_workers=4)
# %%
model_ef = timm.create_model('efficientnet_b0', num_classes=2, pretrained=True).to(device)
x = torch.randn(32, 3, 224, 224).to(device)
optimizer_ef = optim.Adam(model_ef.parameters(),lr=0.001)
model_ef(x).shape
# %%
model_ef=DataParallel(model_ef)
# %%
model_x = timm.create_model('legacy_xception', num_classes=2, pretrained=True).to(device)
x = torch.randn(32, 3, 224, 224).to(device)
optimizer_x = optim.Adam(model_x.parameters(),lr=0.001)
model_x(x).shape

# %%
def calculate_accuracy(outputs, targets):
    _, predicted = outputs.max(1)
    total = targets.size(0)
    correct = predicted.eq(targets).sum().item()
    accuracy = 100.0 * correct / total
    return accuracy

def calculate_performance_metrics(targets, predictions):
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='weighted')
    recall = recall_score(targets, predictions, average='weighted')
    f1 = f1_score(targets, predictions, average='weighted')
    return accuracy, precision, recall, f1

def train_and_evaluate(model, train_loader, test_loader, validate_loader, criterion, optimizer, device, num_epochs):
    model.to(device)
    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0        

        for inputs, targets in train_loader:
            inputs, targets = torch.tensor(inputs).unsqueeze(0), torch.tensor(targets).unsqueeze(0)
            inputs, targets = inputs.clone().detach().to(device), targets.clone().detach().to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_total += targets.size(0)
            train_correct += (outputs.argmax(dim=1) == targets).sum().item()

        train_accuracy = 100.0 * train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        for inputs, targets in validate_loader:
            inputs, targets = torch.tensor(inputs).unsqueeze(0), torch.tensor(targets).unsqueeze(0)
            inputs, targets = inputs.clone().detach().to(device), targets.clone().detach().to(device)

            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            val_loss += loss.item()
            val_total += targets.size(0)
            val_correct += (outputs.argmax(dim=1) == targets).sum().item()

        val_accuracy = 100.0 * val_correct / val_total

        print(f"Epoch {epoch + 1}/{num_epochs}: "
              f"Train Loss: {train_loss/train_total:.4f} "
              f"Train Accuracy: {train_accuracy:.2f}% "
              f"Val Loss: {val_loss/val_total:.4f} "
              f"Val Accuracy: {val_accuracy:.2f}%")
        
        train_lost = train_loss / train_total
        train_acc = 100.0 * train_correct / train_total
        val_lost = val_loss / val_total
        val_acc = 100.0 * val_correct / val_total
        train_losses.append(train_lost)
        train_accuracies.append(train_acc)
        val_losses.append(val_lost)
        val_accuracies.append(val_acc)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_model.pt")

    # Load the best model and evaluate on the test set
    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()
    test_targets = []
    test_predictions = []
    test_correct = 0
    test_total = 0

    for inputs, targets in test_loader:
        inputs, targets = torch.tensor(inputs).unsqueeze(0), torch.tensor(targets).unsqueeze(0)
        inputs, targets = inputs.clone().detach().to(device), targets.clone().detach().to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = outputs.max(1)
        
        test_targets.extend(targets.tolist())
        test_predictions.extend(predicted.tolist())

        test_total += targets.size(0)
        test_correct += (outputs.argmax(dim=1) == targets).sum().item()

    test_accuracy = 100.0 * test_correct / test_total

    test_targets = torch.tensor(test_targets).to(device)
    test_predictions = torch.tensor(test_predictions).to(device)

    test_accuracy, test_precision, test_recall, test_f1 = calculate_performance_metrics(test_targets.cpu().numpy(), test_predictions.cpu().numpy())

    print(f"Test Accuracy: {test_accuracy:.2f}")
    print(f"Test Precision: {test_precision:.2f}")
    print(f"Test Recall: {test_recall:.2f}")
    print(f"Test F1 Score: {test_f1:.2f}")

    print(f"\nBest Validation Accuracy: {best_val_accuracy:.2f}%")
    # Plotting the loss curves and accuracies
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label="Training Accuracy")
    plt.plot(range(1, num_epochs + 1), val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()
# %%
train_losses.clear()
train_accuracies.clear()
val_accuracies.clear()
val_losses.clear()
# %%
num_epoch = 30
train_and_evaluate(model_ef, HQF_train, HQF_val, HQF_test, criterion, optimizer_ef, device, num_epoch)
train_losses.clear()
train_accuracies.clear()
val_accuracies.clear()
val_losses.clear()
train_and_evaluate(model_ef, HQN_train, HQN_val, HQN_test, criterion, optimizer_ef, device, num_epoch)
train_losses.clear()
train_accuracies.clear()
val_accuracies.clear()
val_losses.clear()
train_and_evaluate(model_ef, LQF_train, LQF_val, LQF_test, criterion, optimizer_ef, device, num_epoch)
train_losses.clear()
train_accuracies.clear()
val_accuracies.clear()
val_losses.clear()
train_and_evaluate(model_ef, LQN_train, LQN_val, LQN_test, criterion, optimizer_ef, device, num_epoch)
train_losses.clear()
train_accuracies.clear()
val_accuracies.clear()
val_losses.clear()

# %%
train_and_evaluate(model_x, HQF_train, HQF_val, HQF_test, criterion, optimizer_ef, device, num_epoch)
train_losses.clear()
train_accuracies.clear()
val_accuracies.clear()
val_losses.clear()
train_and_evaluate(model_x, HQN_train, HQN_val, HQN_test, criterion, optimizer_ef, device, num_epoch)
train_losses.clear()
train_accuracies.clear()
val_accuracies.clear()
val_losses.clear()
train_and_evaluate(model_x, LQF_train, LQF_val, LQF_test, criterion, optimizer_ef, device, num_epoch)
train_losses.clear()
train_accuracies.clear()
val_accuracies.clear()
val_losses.clear()
train_and_evaluate(model_x, LQN_train, LQN_val, LQN_test, criterion, optimizer_ef, device, num_epoch)
train_losses.clear()
train_accuracies.clear()
val_accuracies.clear()
val_losses.clear()
# %%
