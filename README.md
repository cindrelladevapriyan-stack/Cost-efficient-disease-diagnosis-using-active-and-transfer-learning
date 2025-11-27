# Cost-efficient-disease-diagnosis-using-active-and-transfer-learning
A deep learning project focusing on cost-efficient disease diagnosis using transfer learning and active learning on medical image datasets such as NIH ChestX-ray14.
import os
import random
import warnings
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models

from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    HAS_ML_STRAT = True
except Exception:
    HAS_ML_STRAT = False
    warnings.warn("Install iterative-stratification for better multi-label splits")


SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = min(8, (os.cpu_count() or 1) - 1) or 0

ALL_LABELS = [
    'atelectasis', 'cardiomegaly', 'consolidation', 'edema', 'effusion',
    'emphysema', 'fibrosis', 'hernia', 'infiltration', 'mass', 'nodule',
    'pleural_thickening', 'pneumonia', 'pneumothorax', 'pneumoperitoneum',
    'pneumomediastinum', 'subcutaneous_emphysema', 'tortuous_aorta',
    'aortic_calcification', 'no_finding'
]


def load_metadata(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"Image Index": "image_index", "Finding Labels": "finding_labels"})
    df['finding_labels'] = df['finding_labels'].fillna("No Finding").astype(str)
    df['image_index'] = df['image_index'].astype(str)
    return df

def encode_labels_and_paths(df, base_dir):
    df['finding_labels'] = df['finding_labels'].fillna("No Finding").astype(str)

    
    for label in ALL_LABELS:
        df[label] = df['finding_labels'].apply(
            lambda x: 1.0 if label.replace("_", " ").title() in str(x) else 0.0
        )

    
    df[ALL_LABELS] = df[ALL_LABELS].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)

    df['path'] = df['image_index'].apply(lambda x: os.path.join(base_dir, str(x)))
    df = df[df['path'].apply(os.path.isfile)].reset_index(drop=True)
    return df


def visualize_random_images(df, n=9):
    print("\nShowing sample images...")
    sample = df.sample(n)
    plt.figure(figsize=(10, 10))
    for i, (_, row) in enumerate(sample.iterrows()):
        img = Image.open(row["path"])
        plt.subplot(3, 3, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(row["finding_labels"])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_label_distribution(df):
    print("\nPlotting label distribution...")
    plt.figure(figsize=(12, 6))
    df[ALL_LABELS].sum().sort_values(ascending=False).plot(kind="bar")
    plt.title("Label Frequency Distribution")
    plt.ylabel("Count")
    plt.xticks(rotation=75)
    plt.tight_layout()
    plt.show()


def plot_cooccurrence_heatmap(df):
    print("\nPlotting label co-occurrence heatmap...")
    corr = df[ALL_LABELS].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", annot=False)
    plt.title("Label Co-occurrence Heatmap")
    plt.tight_layout()
    plt.show()


def plot_image_size_distribution(df):
    print("\nPlotting image size distribution...")
    widths, heights = [], []
    for path in df["path"].sample(200):
        img = Image.open(path)
        widths.append(img.size[0])
        heights.append(img.size[1])

    plt.figure(figsize=(10, 5))
    sns.histplot(widths, bins=30, kde=True)
    plt.title("Image Width Distribution")
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.histplot(heights, bins=30, kde=True)
    plt.title("Image Height Distribution")
    plt.show()


def create_train_test(df, test_size=0.2, random_state=SEED):
    X = df.index.values.reshape(-1, 1)
    Y = df[ALL_LABELS].values
    if HAS_ML_STRAT:
        msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(msss.split(X, Y))
    else:
       
        stratify_col = (Y.sum(axis=1) > 0).astype(int)  
        train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=test_size, random_state=random_state, stratify=stratify_col)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, test_df


class ChestXrayDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        labels = torch.tensor(row[ALL_LABELS].values.astype(np.float32))
        return img, labels


class MultiLabelResNet(nn.Module):
    def __init__(self, num_labels=len(ALL_LABELS)):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.classifier = nn.Linear(in_features, num_labels)

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    for imgs, labels in tqdm(loader, desc="Training", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    preds_all, labels_all = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Evaluating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)

            probs = torch.sigmoid(outputs).cpu().numpy()
            preds_all.append(probs)
            labels_all.append(labels.cpu().numpy())

    preds_all = np.vstack(preds_all)
    labels_all = np.vstack(labels_all)


    thresholds = []
    for i in range(labels_all.shape[1]):
        unique_vals = np.unique(labels_all[:, i])

      
        if len(unique_vals) < 2:
            thresholds.append(0.5)
            continue

        best_f1 = 0
        best_thresh = 0.5
        for t in np.linspace(0.1, 0.9, 9):
            f1 = f1_score(labels_all[:, i], preds_all[:, i] > t, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
        thresholds.append(best_thresh)

    thresholds = np.array(thresholds)
    preds_binary = (preds_all > thresholds).astype(int)

    
    f1 = f1_score(labels_all, preds_binary, average="macro", zero_division=0)


    auc_list = []
    for i in range(labels_all.shape[1]):
        unique_vals = np.unique(labels_all[:, i])
        if len(unique_vals) < 2:
            continue  
        try:
            auc_list.append(roc_auc_score(labels_all[:, i], preds_all[:, i]))
        except:
            pass

    auc = float(np.mean(auc_list)) if len(auc_list) > 0 else 0.0

    return total_loss / len(loader.dataset), f1, auc



def active_learning_pipeline(train_df, test_df,
                             rounds=3, query_size=1000, init_size=2000,
                             batch_size=64, image_size=224, epochs=2):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_dataset = ChestXrayDataset(train_df, transform)
    test_dataset = ChestXrayDataset(test_df, transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_idx = np.arange(len(train_dataset))
    np.random.shuffle(all_idx)
    labeled_idx = list(all_idx[:init_size])
    unlabeled_idx = list(all_idx[init_size:])

    model = MultiLabelResNet()
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    loss_history = []
    f1_history = []
    auc_history = []

    for r in range(rounds):
        print(f"\n=== ROUND {r+1}/{rounds} | labeled={len(labeled_idx)} ===")

        train_loader = DataLoader(Subset(train_dataset, labeled_idx),
                                  batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            print(f"Epoch {epoch+1} done, Loss={loss:.4f}")

        val_loss, f1, auc = evaluate(model, test_loader, criterion, device)
        print(f"Eval: loss={val_loss:.4f}  f1={f1:.4f}  auc={auc}")

        loss_history.append(val_loss)
        f1_history.append(f1)
        auc_history.append(auc)

      
        if len(unlabeled_idx) == 0:
            break
        unl_loader = DataLoader(Subset(train_dataset, unlabeled_idx), batch_size=batch_size)

        model.eval()
        scores = []
        with torch.no_grad():
            for imgs, _ in unl_loader:
                imgs = imgs.to(device)
                probs = torch.sigmoid(model(imgs)).cpu().numpy()
                entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1)
                scores.extend(entropy)

        selected_rel = np.argsort(scores)[-query_size:]
        selected_abs = [unlabeled_idx[i] for i in selected_rel]

        labeled_idx.extend(selected_abs)
        unlabeled_idx = [u for u in unlabeled_idx if u not in selected_abs]

   
    plt.figure(figsize=(10, 4))
    plt.plot(loss_history, marker="o")
    plt.title("Validation Loss per Round")
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(f1_history, marker="o")
    plt.title("F1 Score per Round")
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(auc_history, marker="o")
    plt.title("AUC per Round")
    plt.show()

    return loss_history, f1_history, auc_history


if __name__ == "__main__":
    csv_path = ""
    base_dir = ""

    print("Loading metadata...")
    df = load_metadata(csv_path)

    print("Encoding labels + verifying files...")
    df = encode_labels_and_paths(df, base_dir)
    print("Total valid images:", len(df))

    visualize_random_images(df)
    plot_label_distribution(df)
    plot_cooccurrence_heatmap(df)
    plot_image_size_distribution(df)

    train_df, test_df = create_train_test(df)

    loss_hist, f1_hist, auc_hist = active_learning_pipeline(
        train_df, test_df,
        rounds=3, query_size=1000, init_size=2000,
        batch_size=64, image_size=224, epochs=2)

    print("\nFinal Metrics:")
    print("Loss history:", loss_hist)
    print("F1 history:", f1_hist)
    print("AUC history:", auc_hist)
