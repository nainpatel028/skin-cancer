import torch
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def train(model, optimizer, criterion, train_dl, valid_dl, epochs, save_dir="models"):
    result = []
    val_res = []

    # Create directory to save models if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    for e in range(epochs):
        print(f"Training the Epoch: {e + 1}")
        model.train()
        total_loss = 0
        total = 0
        correct = 0
        
        for X_train, y_train in train_dl:
            X_train, y_train = X_train.to(device).float(), y_train.to(device).long()
            
            y_pred = model(X_train)
            if isinstance(y_pred, tuple):  # In case of Inception
                y_pred = y_pred[0]
                
            loss = criterion(y_pred, y_train)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            _, predicted = torch.max(y_pred.data, 1)
            total += y_train.size(0)
            correct += (predicted == y_train).sum().item()
            
        acc_train = 100 * correct / total
        _, val_acc = valid(model, valid_dl, device)
        
        print(f'Epoch: [{e+1}/{epochs}], Loss: {total_loss / total:.4f}, Train Acc: {acc_train:.2f}, Val Acc: {val_acc:.2f}')
        result.append(acc_train)
        val_res.append(val_acc)
        
        # Save model state and optimizer state for this epoch
        torch.save({
            'epoch': e + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_accuracy': acc_train,
            'val_accuracy': val_acc
        }, os.path.join(save_dir, f'model_epoch_{e+1}.pth'))
    
    # Save results to files
    np.savetxt('result.csv', np.array(result), fmt='%.2f', delimiter=',')
    np.savetxt('val_result.csv', np.array(val_res), fmt='%.2f', delimiter=',')


# Validation Function
def valid(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for sample, target in val_loader:
            sample = sample.to(device).float()
            target = target.to(device).long()
            outputs = model(sample)
            _, predicted = torch.max(outputs.data, 1)
            
            total += target.size(0)
            correct += (predicted == target).sum().item()

    val_accuracy = 100 * correct / total
    return [], val_accuracy


def plot():
    train_data = np.loadtxt('result.csv', delimiter=',', ndmin=1)
    val_data = np.loadtxt('val_result.csv', delimiter=',', ndmin=1)

    plt.figure()
    plt.plot(range(1, len(train_data) + 1), train_data, color='blue', label='Train')
    plt.plot(range(1, len(val_data) + 1), val_data, color='red', label='Validation')
    plt.legend()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Train and Validation Accuracy', fontsize=16)
    plt.savefig('plot.png')

    plt.show()


def preds(Model, test_dl, label_encoder, device):
    # Example: Getting a Batch of Data
    images, targets = next(iter(test_dl))  
    indices = np.arange(len(images))
    np.random.shuffle(indices)

    fig, axes = plt.subplots(8, 2, figsize=(15, 24))  # 16 rows and 2 columns (image and bar graph).
    axes = axes.ravel()

    # Set the background color to white for all subplots.
    for ax in axes:
        ax.set_facecolor('white')

    # Set the background color to white for the entire graph.
    fig.patch.set_facecolor('white')

    # Inverting the LabelEncoder dictionary to map integers back to labels
    int_to_label = {i: label for i, label in enumerate(label_encoder.classes_)}

    # Calculation of precision
    correct_predictions = {label: 0 for label in int_to_label.values()}
    total_predictions = {label: 0 for label in int_to_label.values()}


    # Limit the number of iterations to the quantity of available subplots
    for i in range(min(len(indices), len(axes) // 2)):
        idx = indices[i]
        image, actual_target = images[idx], targets[idx]
        image_tensor = image.to(device).unsqueeze(0)
        output = Model(image_tensor)
        _, prediction = torch.max(output, 1)

        predicted_label = int_to_label[prediction.cpu().item()]  # Convert to class name
        actual_label = int_to_label[actual_target.item()]
        total_predictions[actual_label] += 1
        if predicted_label == actual_label:
            correct_predictions[predicted_label] += 1

        # Convert the image tensor for plotting
        image = denormalize(image, mean, std)
        image = image.permute(1, 2, 0)  # Change dimensions from CxHxW to HxWxC
        image = image.cpu().numpy()

        # Show image
        axes[2 * i].imshow(image)
        axes[2 * i].set_title(f"Actual Label: {actual_label}", color='black')
        axes[2 * i].axis('off')

        # Display a bar graph showing probabilities
        probabilities = torch.nn.functional.softmax(output, dim=1).cpu().detach().numpy().flatten()
        bar_colors = ['#45c5e4'] * len(probabilities)  # Default colors
        # Change bar colors based on conditions
        if predicted_label == actual_label:
            bar_colors[np.argmax(probabilities)] = 'aquamarine'  # correct prediction
        else:
            bar_colors[np.argmax(probabilities)] = 'red'  # incorrect prediction
            bar_colors[list(int_to_label.values()).index(actual_label)] = 'steelblue'  # Correct label

        axes[2 * i + 1].bar(int_to_label.values(), probabilities, color=bar_colors)
        axes[2 * i + 1].set_ylim(0, 1)
        axes[2 * i + 1].set_title(f"Probabilities for {predicted_label}", color='black')

        # Configure colors of the axes and annotations
        axes[2 * i + 1].tick_params(axis='x', colors='black')
        axes[2 * i + 1].tick_params(axis='y', colors='black')
        axes[2 * i + 1].spines['bottom'].set_color('black')
        axes[2 * i + 1].spines['top'].set_color('black') 
        axes[2 * i + 1].spines['right'].set_color('black')
        axes[2 * i + 1].spines['left'].set_color('black')
        axes[2 * i + 1].yaxis.label.set_color('black')
        axes[2 * i + 1].xaxis.label.set_color('black')

    # Adjust the design and show the graph
    plt.savefig('output.png',transparent=True)
    plt.tight_layout()
    plt.show()
