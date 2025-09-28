import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from keras import Sequential
from keras.src.layers import RandomZoom
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models, datasets
import matplotlib.pyplot as plt
from sklearn.svm import SVC, SVR
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error
from sklearn.decomposition import PCA
from lifelines import CoxPHFitter
import pandas as pd
import seaborn as sns
from keras.layers import RandomRotation
from keras.models import Sequential

print("Done Test 1")

# Change to Kaggle dataset path
BATCH_SIZE = 16
EPOCHS = 3  # keep small for demo
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Done Test 2")

from torchvision import transforms

# Training data transforms (augmentation + normalization)
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),       # resize images
    transforms.RandomHorizontalFlip(),   # simple augmentation
    transforms.ToTensor(),               # convert to tensor
    transforms.Normalize((0.5,), (0.5,)) # normalize (mean, std)
])

# Test/validation data transforms (just resize + normalize)
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

DATA_DIR = r"C:\Kaggle\Input\augmented-alzheimer-mri-dataset-v2\Alzheimer_Dataset"
train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform_train)
test_ds = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=transform_test)

def data_augmentar():
    data_augmentation = Sequential()
    data_augmentation.add(RandomRotation(factor=(-0.15, 0.15)))
    data_augmentation.add(RandomZoom((-0.3, -0.1)))
    return data_augmentation

data_augmentation = data_augmentar()
assert(data_augmentation.layers[0].name.startswith('random_rotation'))
assert(data_augmentation.layers[0].factor == (-0.15, 0.15))
assert(data_augmentation.layers[1].name.startswith('random_zoom'))
assert(data_augmentation.layers[1].height_factor == (-0.3, -0.1))


def get_dataloaders():
    if os.path.exists(DATA_DIR) and len(os.listdir(DATA_DIR)) > 0:
        # Assume Kaggle dataset in ImageFolder structure
        transform_train = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

        train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform_train)
        test_ds = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=transform_test)
    else:
        # Synthetic data fallback
        class SyntheticMRIDataset(Dataset):
            def __init__(self, n_samples=200, n_classes=4):
                self.X = np.random.rand(n_samples, 1, 128, 128).astype(np.float32)
                self.y = np.random.randint(0, n_classes, size=n_samples)
            def __len__(self): return len(self.X)
            def __getitem__(self, idx):
                return torch.tensor(self.X[idx]), self.y[idx]
        train_ds = SyntheticMRIDataset(200)
        test_ds = SyntheticMRIDataset(80)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader, train_ds, test_ds

train_loader, test_loader, train_ds, test_ds = get_dataloaders()
print("Done Test 3")

class CNNEncoder(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        base_model = models.resnet18(weights=None)
        base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = base_model.fc.in_features
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        self.fc = nn.Linear(num_ftrs, num_classes)
    def forward(self, x):
        feats = self.feature_extractor(x)
        feats = feats.view(feats.size(0), -1)
        out = self.fc(feats)
        return out, feats

model = CNNEncoder(num_classes=4).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
print("Done Test 4")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs, _ = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}")
print("Done Test 5")

model.eval()
embeddings_train, y_train = [], []
embeddings_test, y_test = [], []
with torch.no_grad():
    for imgs, labels in train_loader:
        imgs = imgs.to(DEVICE)
        outputs, feats = model(imgs)
        embeddings_train.append(feats.cpu().numpy())
        y_train.extend(labels)
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        outputs, feats = model(imgs)
        embeddings_test.append(feats.cpu().numpy())
        y_test.extend(labels)

X_train = np.vstack(embeddings_train)
X_test = np.vstack(embeddings_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
print("Done test 6")

svm_clf = SVC(kernel='rbf')
svm_clf.fit(X_train, y_train)
y_pred_svm = svm_clf.predict(X_test)
print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

# SVR for synthetic CDR-SB regression
# Simulate CDR-SB scores: map class labels to severity + noise
cdr_scores_train = y_train + np.random.normal(0, 0.5, size=len(y_train))
cdr_scores_test = y_test + np.random.normal(0, 0.5, size=len(y_test))

svr_reg = SVR(kernel='rbf')
svr_reg.fit(X_train, cdr_scores_train)
cdr_pred = svr_reg.predict(X_test)
print("SVR MAE:", mean_absolute_error(cdr_scores_test, cdr_pred))

#CoxPH survival analysis (synthetic)
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Create synthetic survival data
survival_df = pd.DataFrame(X_train_pca, columns=[f"pc{i}" for i in range(5)])
survival_df['time'] = np.random.uniform(1, 5, size=len(survival_df))
survival_df['event'] = np.random.binomial(1, 0.3, size=len(survival_df))

cox = CoxPHFitter()
cox.fit(survival_df, duration_col='time', event_col='event')
cox.print_summary()

# Confusion matrix for SVM
cm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("SVM Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()

# Plot survival curves
cox.plot()
plt.title("CoxPH Coefficients")
plt.show()

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt

# Let's say X_train and X_test are numeric arrays
np.random.seed(42)
X_train = np.random.rand(100, 10)  # 100 samples, 10 features
X_test = np.random.rand(20, 10)    # 20 samples, 10 features

# PCA transformation
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Creating synthetic survival data
survival_df = pd.DataFrame(X_train_pca, columns=[f"pc{i}" for i in range(5)])
survival_df['time'] = np.random.uniform(1, 5, size=len(survival_df))        # survival time
survival_df['event'] = np.random.binomial(1, 0.3, size=len(survival_df))     # event indicator

cox = CoxPHFitter()
cox.fit(survival_df, duration_col='time', event_col='event')
cox.print_summary()

# Preparing test set for prediction
test_df = pd.DataFrame(X_test_pca, columns=[f"pc{i}" for i in range(5)])

survival_functions = cox.predict_survival_function(test_df)

#Plotting the curves
plt.figure(figsize=(8, 6))
for i in range(min(5, survival_functions.shape[1])):  # plot first 5 test samples
    plt.step(survival_functions.index, survival_functions.iloc[:, i], where="post", label=f"Sample {i}")
plt.xlabel("Time")
plt.ylabel("Survival probability")
plt.title("Predicted Survival Curves (Synthetic Data)")
plt.legend()
plt.show()
