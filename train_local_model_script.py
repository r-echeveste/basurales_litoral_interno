import sys, os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset, WeightedRandomSampler
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from torchvision import transforms

project_directory = "./"
print(f"Current working directory: {os.getcwd()}")

sys.path.append(os.path.join(project_directory, 'lombardia_model'))

# Importing model definition
from architecture.resnet50_fpn import Net

# Decide if rebalancing based on the target class will be used to train
rebalance = True

# Decide if loading the xval indices else randomly selecting the train, val and test indices
load_indices = True

print("Loading Lombardia model...")
# Loading state dictionary
STATE_DICT_PATH = "lombardia_model/weights/checkpoint.pth"

# Creating an instance of the model
model = Net(num_classes=1)

# Loading the weights into the model
model.load_state_dict(torch.load(STATE_DICT_PATH, map_location=torch.device('cpu')))

# Freeze all layers except the last two fc layers
for name, param in model.named_parameters():
    if not ('fc' in name or 'classifier' in name): # Freeze if not an fc layer in classifier
        param.requires_grad = False
    else:
        param.requires_grad = True

# Print trainable parameters to verify
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

# Output location

if (rebalance):
    print("We will train rebalancing the dataset...")
    output_dir = "litoral_model/rebalance"
else:
    print("We will train without rebalancing the dataset...")
    output_dir = "litoral_model/no_rebalance"

os.makedirs(output_dir, exist_ok=True)

print("Loading Data...")

# Loading labels
db_parana = pd.read_csv("labels/parana_labels.csv")
db_rosario = pd.read_csv("labels/rosario_labels.csv")
db_santafe = pd.read_csv("labels/santafe_labels.csv")

train_val_transform = transforms.Compose([
            transforms.Resize(800), # Images are resized to 800x800 using the same resolution as inTorres et al. (2023)
            transforms.RandomHorizontalFlip(),  # Random horizontal flips as in Torres et al. (2023)
            transforms.RandomRotation(90), # Random 90-degree rotations as in Torres et al. (2023)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])  # ImageNet normalization
            ])

test_transform = transforms.Compose([
            transforms.Resize(800), # Images are resized to 800x800 using the same resolution as inTorres et al. (2023)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])  # ImageNet normalization
            ])


class CustomImageDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform):
        """
        Args:
            dataframe (pandas.DataFrame): DataFrame containing image file names and labels.
            img_dir (str): Directory containing the images.
        """
        self.img_labels = dataframe[['file_name', 'etiqueta']]
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label

# If load_indices is True, load the indices from the CSV files
if load_indices:
    
    print("Loading indices from CSV files...")

    location_indices = "litoral_model/no_rebalance"
    
    db_parana = pd.read_csv("labels/parana_labels.csv")
    db_rosario = pd.read_csv("labels/rosario_labels.csv")
    db_santafe = pd.read_csv("labels/santafe_labels.csv")

    list_test_images = pd.read_csv(location_indices + "/test_images.csv")
    list_val_images = pd.read_csv(location_indices + "/val_images.csv")
    list_train_images = pd.read_csv(location_indices + "/train_images.csv")

    # Loading the images used for train, val and test    
    db_parana_train = db_parana[db_parana['file_name'].isin(list_train_images['file_name'])]
    db_parana_val = db_parana[db_parana['file_name'].isin(list_val_images['file_name'])]
    db_parana_test = db_parana[db_parana['file_name'].isin(list_test_images['file_name'])]

    db_rosario_train = db_rosario[db_rosario['file_name'].isin(list_train_images['file_name'])]
    db_rosario_val = db_rosario[db_rosario['file_name'].isin(list_val_images['file_name'])]
    db_rosario_test = db_rosario[db_rosario['file_name'].isin(list_test_images['file_name'])]

    db_santafe_train = db_santafe[db_santafe['file_name'].isin(list_train_images['file_name'])]
    db_santafe_val = db_santafe[db_santafe['file_name'].isin(list_val_images['file_name'])]
    db_santafe_test = db_santafe[db_santafe['file_name'].isin(list_test_images['file_name'])]

    parana_dataset = CustomImageDataset(db_parana, 'images/parana/patches_parana',test_transform)
    rosario_dataset = CustomImageDataset(db_rosario, 'images/rosario/patches_ros',test_transform)
    santafe_dataset = CustomImageDataset(db_santafe, 'images/santa_fe/patches_sfe',test_transform)

    parana_test_dataset = CustomImageDataset(db_parana_test, 'images/parana/patches_parana',test_transform)
    parana_val_dataset = CustomImageDataset(db_parana_val, 'images/parana/patches_parana',train_val_transform)
    parana_train_dataset = CustomImageDataset(db_parana_train, 'images/parana/patches_parana',train_val_transform)

    rosario_test_dataset = CustomImageDataset(db_rosario_test, 'images/rosario/patches_ros',test_transform)
    rosario_val_dataset = CustomImageDataset(db_rosario_val, 'images/rosario/patches_ros',train_val_transform)
    rosario_train_dataset = CustomImageDataset(db_rosario_train, 'images/rosario/patches_ros',train_val_transform)

    santafe_test_dataset = CustomImageDataset(db_santafe_test, 'images/santa_fe/patches_sfe',test_transform)
    santafe_val_dataset = CustomImageDataset(db_santafe_val, 'images/santa_fe/patches_sfe',train_val_transform)
    santafe_train_dataset = CustomImageDataset(db_santafe_train, 'images/santa_fe/patches_sfe',train_val_transform)

    # Saving images used for train, val and test in the output directory

    list_train_images.to_csv(os.path.join(output_dir, 'train_images.csv'), index=False)
    list_val_images.to_csv(os.path.join(output_dir, 'val_images.csv'), index=False)
    list_test_images.to_csv(os.path.join(output_dir, 'test_images.csv'), index=False)

    print(f"Paraná: Train - {len(parana_train_dataset)}, Val - {len(parana_val_dataset)}, Test - {len(parana_test_dataset)}")
    print(f"Rosario: Train - {len(rosario_train_dataset)}, Val - {len(rosario_val_dataset)}, Test - {len(rosario_test_dataset)}")
    print(f"Santa Fe: Train - {len(santafe_train_dataset)}, Val - {len(santafe_val_dataset)}, Test - {len(santafe_test_dataset)}")

    # Combine train datasets
    train_dataset = ConcatDataset([parana_train_dataset, rosario_train_dataset, santafe_train_dataset])

    # Combine val datasets
    val_dataset = ConcatDataset([parana_val_dataset, rosario_val_dataset, santafe_val_dataset])

    # Combine test datasets
    test_dataset = ConcatDataset([parana_test_dataset, rosario_test_dataset, santafe_test_dataset])

    print(f"Combined Train Dataset Size: {len(train_dataset)}")
    print(f"Combined Val Dataset Size: {len(val_dataset)}")
    print(f"Combined Test Dataset Size: {len(test_dataset)}")

else:
    parana_dataset = CustomImageDataset(db_parana, 'images/parana/patches_parana',test_transform)
    rosario_dataset = CustomImageDataset(db_rosario, 'images/rosario/patches_ros',test_transform)
    santafe_dataset = CustomImageDataset(db_santafe, 'images/santa_fe/patches_sfe',test_transform)
    
    # Separating each dataset into train val test

    # Split ratios
    train_ratio = 0.6
    val_ratio = 0.20
    test_ratio = 0.20

    # Split parana_dataset
    parana_train_size = int(train_ratio * len(parana_dataset))
    parana_val_size = int(val_ratio * len(parana_dataset))
    parana_test_size = len(parana_dataset) - parana_train_size - parana_val_size
    parana_train_dataset, parana_val_dataset, parana_test_dataset = random_split(
        parana_dataset, [parana_train_size, parana_val_size, parana_test_size]
    )

    # Split rosario_dataset
    rosario_train_size = int(train_ratio * len(rosario_dataset))
    rosario_val_size = int(val_ratio * len(rosario_dataset))
    rosario_test_size = len(rosario_dataset) - rosario_train_size - rosario_val_size
    rosario_train_dataset, rosario_val_dataset, rosario_test_dataset = random_split(
        rosario_dataset, [rosario_train_size, rosario_val_size, rosario_test_size]
    )

    # Split santafe_dataset
    santafe_train_size = int(train_ratio * len(santafe_dataset))
    santafe_val_size = int(val_ratio * len(santafe_dataset))
    santafe_test_size = len(santafe_dataset) - santafe_train_size - santafe_val_size
    santafe_train_dataset, santafe_val_dataset, santafe_test_dataset = random_split(
        santafe_dataset, [santafe_train_size, santafe_val_size, santafe_test_size]
    )

    print(f"Paraná: Train - {len(parana_train_dataset)}, Val - {len(parana_val_dataset)}, Test - {len(parana_test_dataset)}")
    print(f"Rosario: Train - {len(rosario_train_dataset)}, Val - {len(rosario_val_dataset)}, Test - {len(rosario_test_dataset)}")
    print(f"Santa Fe: Train - {len(santafe_train_dataset)}, Val - {len(santafe_val_dataset)}, Test - {len(santafe_test_dataset)}")

    # Combine train datasets
    train_dataset = ConcatDataset([parana_train_dataset, rosario_train_dataset, santafe_train_dataset])

    # Combine val datasets
    val_dataset = ConcatDataset([parana_val_dataset, rosario_val_dataset, santafe_val_dataset])

    # Combine test datasets
    test_dataset = ConcatDataset([parana_test_dataset, rosario_test_dataset, santafe_test_dataset])

    print(f"Combined Train Dataset Size: {len(train_dataset)}")
    print(f"Combined Val Dataset Size: {len(val_dataset)}")
    print(f"Combined Test Dataset Size: {len(test_dataset)}")

    # Saving images used for train, val and test

    train_image_filenames = []
    train_image_filenames.extend(parana_dataset.img_labels.iloc[parana_train_dataset.indices]['file_name'].tolist())
    train_image_filenames.extend(rosario_dataset.img_labels.iloc[rosario_train_dataset.indices]['file_name'].tolist())
    train_image_filenames.extend(santafe_dataset.img_labels.iloc[santafe_train_dataset.indices]['file_name'].tolist())
    train_images_df = pd.DataFrame({'file_name': train_image_filenames})
    train_images_df.to_csv(os.path.join(output_dir, 'train_images.csv'), index=False)

    val_image_filenames = []
    val_image_filenames.extend(parana_dataset.img_labels.iloc[parana_val_dataset.indices]['file_name'].tolist())
    val_image_filenames.extend(rosario_dataset.img_labels.iloc[rosario_val_dataset.indices]['file_name'].tolist())
    val_image_filenames.extend(santafe_dataset.img_labels.iloc[santafe_val_dataset.indices]['file_name'].tolist())
    val_images_df = pd.DataFrame({'file_name': val_image_filenames})
    val_images_df.to_csv(os.path.join(output_dir, 'val_images.csv'), index=False)

    test_image_filenames = []
    test_image_filenames.extend(parana_dataset.img_labels.iloc[parana_test_dataset.indices]['file_name'].tolist())
    test_image_filenames.extend(rosario_dataset.img_labels.iloc[rosario_test_dataset.indices]['file_name'].tolist())
    test_image_filenames.extend(santafe_dataset.img_labels.iloc[santafe_test_dataset.indices]['file_name'].tolist())
    test_images_df = pd.DataFrame({'file_name': test_image_filenames})
    test_images_df.to_csv(os.path.join(output_dir, 'test_images.csv'), index=False)

# Training the model

print("Training the model...")

batch_size = 16

# Create data loaders

if (rebalance):
    # Calculating sample weights
    all_labels_train = []
    for dataset in train_dataset.datasets:  # Access individual datasets in ConcatDataset
        all_labels_train.extend(dataset.img_labels['etiqueta'].tolist())  # Assuming 'etiqueta' is the label column

    class_counts_train = np.bincount(all_labels_train)
    class_weights_train = 1. / torch.tensor(class_counts_train, dtype=torch.float)
    sample_weights_train = [class_weights_train[label] for label in all_labels_train]
    sample_weights_train = torch.tensor(sample_weights_train, dtype=torch.float)
    sample_weights_train = sample_weights_train / sample_weights_train.sum()  # Normalize weights

    all_labels_val = []
    for dataset in val_dataset.datasets:  # Access individual datasets in ConcatDataset
        all_labels_val.extend(dataset.img_labels['etiqueta'].tolist())  # Assuming 'etiqueta' is the label column
    class_counts_val = np.bincount(all_labels_val)
    class_weights_val = 1. / torch.tensor(class_counts_val, dtype=torch.float)
    sample_weights_val = [class_weights_val[label] for label in all_labels_val]
    sample_weights_val = torch.tensor(sample_weights_val, dtype=torch.float)
    sample_weights_val = sample_weights_val / sample_weights_val.sum()  # Normalize weights

    # Creating the WeightedRandomSampler
    train_sampler = WeightedRandomSampler(weights=sample_weights_train,
                                num_samples=len(train_dataset),
                                replacement=True)
    
    val_sampler = WeightedRandomSampler(weights=sample_weights_val,
                                num_samples=len(val_dataset),
                                replacement=True)
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, shuffle=False)
else:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define optimizer and loss function
optimizer = Adam(model.parameters(), lr=0.001)
criterion = BCEWithLogitsLoss()  # For binary classification

# Early stopping parameters
n_epochs = 200
patience = 50  # Same patience as in Torres et al. (2023)
best_val_loss = float('inf')  # Initialize with a very high value
epochs_without_improvement = 0

# Moving model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_loss_history = []
val_loss_history = []

# Training loop
for epoch in range(n_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to the appropriate device (e.g., GPU if available)
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()  # Reset gradients
        output = model(data)  # Forward pass
        loss = criterion(output, target.unsqueeze(1).float())  # Calculate loss
        loss.backward()  # Backpropagate
        optimizer.step()  # Update weights

    # Computing Train and Val loss
    model.eval()
    train_loss = 0
    val_loss = 0

    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += criterion(output, target.unsqueeze(1).float()).item()  # Sum up batch loss

        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target.unsqueeze(1).float()).item()  # Sum up batch loss

    train_loss /= len(train_loader)  # Average validation loss
    val_loss /= len(val_loader)  # Average validation loss

    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)

    print(f'Epoch: {epoch}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}')

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        # You can save the best model here
        torch.save(model.state_dict(), os.path.join(output_dir, 'best_local_model.pth'))
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f'Early stopping triggered after {epoch} epochs.')
            break


loss_history = pd.DataFrame({
    'train_loss': train_loss_history,
    'val_loss': val_loss_history
})

loss_history.to_csv(os.path.join(output_dir, 'loss_history.csv'), index=False)
