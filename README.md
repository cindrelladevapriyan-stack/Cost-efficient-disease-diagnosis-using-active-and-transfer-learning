# Cost-efficient-disease-diagnosis-using-active-and-transfer-learning
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
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms, models

from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    HAS_ML_STRAT = True
except Exception:
    HAS_ML_STRAT = False
    warnings.warn("Install iterative-stratification for better multi-label splits (pip install iterative-stratification)")


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
        pretty = label.replace("_", " ").title()
        df[label] = df['finding_labels'].apply(lambda x: 1.0 if pretty in str(x) else 0.0)
    
    df[ALL_LABELS] = df[ALL_LABELS].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
    df['path'] = df['image_index'].apply(lambda x: os.path.join(base_dir, str(x)))
    df = df[df['path'].apply(os.path.isfile)].reset_index(drop=True)
    return df

def visualize_random_images(df, n=9):
    print("\nShowing sample images...")
    sample = df.sample(min(n, len(df)))
    plt.figure(figsize=(10, 10))
    for i, (_, row) in enumerate(sample.iterrows()):
        try:
            img = Image.open(row["path"])
            plt.subplot(3, 3, i + 1)
            plt.imshow(img, cmap="gray")
            plt.title(row["finding_labels"])
            plt.axis("off")
        except Exception as e:
            print("Error loading image:", row["path"], e)
    plt.tight_layout()
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
    def __init__(self, num_labels=len(ALL_LABELS), pretrained=True):
        super().__init__()
        if pretrained:
            base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            base = models.resnet50(weights=None)
        in_features = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.classifier = nn.Linear(in_features, num_labels)

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)


class FocalBCEWithLogits(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        if pos_weight is not None:
            self.register_buffer('pos_weight', pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits, targets):
        if self.pos_weight is not None:
            bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction='none')
        else:
            bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = (1 - pt) ** self.gamma
        loss = self.alpha * focal * bce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def mixup_data(x, y, alpha=0.4):
    """Simple mixup for multi-label (returns mixed inputs, pairs of labels and lambda)"""
    if alpha <= 0:
        return x, y, None
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, (y_a, y_b, lam)

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_one_epoch(model, loader, criterion, optimizer, device, use_mixup=False, mixup_alpha=0.4):
    model.train()
    running_loss = 0.0
    total = 0
    for imgs, labels in tqdm(loader, desc="Training", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        if use_mixup:
            mixed_imgs, mixed_targets = mixup_data(imgs, labels, alpha=mixup_alpha)
            if mixed_targets is None:
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            else:
                y_a, y_b, lam = mixed_targets
                outputs = model(mixed_imgs)
                # if criterion supports mix (like FocalBCE), use mixup_criterion fallback
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)
    return running_loss / max(total, 1)

def predict_probs(model, loader, device):
    model.eval()
    probs_all, labels_all = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Predicting", leave=False):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_all.append(probs)
            labels_all.append(labels.numpy())
    if len(probs_all) == 0:
        return np.zeros((0, len(ALL_LABELS))), np.zeros((0, len(ALL_LABELS)))
    return np.vstack(probs_all), np.vstack(labels_all).astype(np.float32)

def tune_thresholds(probs_val, labels_val, steps=99):
    num_labels = labels_val.shape[1]
    thresholds = np.full(num_labels, 0.5, dtype=np.float32)
    per_class_f1 = np.zeros(num_labels, dtype=np.float32)
    for i in range(num_labels):
        unique_vals = np.unique(labels_val[:, i])
        if len(unique_vals) < 2:
            thresholds[i] = 0.5
            per_class_f1[i] = 0.0
            continue
        best_f1 = 0.0
        best_t = 0.5
        for t in np.linspace(0.01, 0.99, steps):
            f1 = f1_score(labels_val[:, i], (probs_val[:, i] > t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        thresholds[i] = best_t
        per_class_f1[i] = best_f1
    return thresholds, per_class_f1

def evaluate_with_thresholds(probs, labels, thresholds):
    preds = (probs > thresholds).astype(int)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    per_class_f1 = []
    for i in range(labels.shape[1]):
        unique_vals = np.unique(labels[:, i])
        if len(unique_vals) < 2:
            per_class_f1.append(0.0)
        else:
            per_class_f1.append(f1_score(labels[:, i], preds[:, i], zero_division=0))
    aucs = []
    for i in range(labels.shape[1]):
        if len(np.unique(labels[:, i])) < 2:
            continue
        try:
            aucs.append(roc_auc_score(labels[:, i], probs[:, i]))
        except:
            pass
    macro_auc = float(np.mean(aucs)) if len(aucs) > 0 else np.nan
    return float(macro_f1), float(macro_auc), np.array(per_class_f1)


def active_learning_pipeline(pool_df, val_df, test_df,
                             rounds=3, query_size=1000, init_size=5000,
                             batch_size=64, image_size=224, epochs=4,
                             max_unlabeled_to_score=5000,
                             use_focal=True, focal_gamma=2.0,
                             use_mixup=False, mixup_alpha=0.4,
                             device=None):
    """
    pool_df: initial training pool (will be split into labeled/unlabeled)
    val_df: held-out validation (for threshold tuning & model selection)
    test_df: final test set
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    pool_dataset = ChestXrayDataset(pool_df, transform=train_transform)
    val_dataset = ChestXrayDataset(val_df, transform=eval_transform)
    test_dataset = ChestXrayDataset(test_df, transform=eval_transform)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=DEFAULT_NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=DEFAULT_NUM_WORKERS)

    all_idx = np.arange(len(pool_dataset))
    np.random.shuffle(all_idx)
    labeled_idx = list(all_idx[:init_size])
    unlabeled_idx = list(all_idx[init_size:])

    model = MultiLabelResNet()
    model.to(device)

    
    def compute_pos_weight(indices):
        labeled_labels = np.vstack([pool_df.iloc[i][ALL_LABELS].values for i in indices]).astype(np.float32)
        N = labeled_labels.shape[0]
        class_counts = labeled_labels.sum(axis=0)
        neg = (N - class_counts)
        pos = class_counts
        pos_weight = (neg + 1.0) / (pos + 1.0)  
        pos_weight = np.clip(pos_weight, 1e-6, 1e6).astype(np.float32)
        return pos_weight, labeled_labels

    
    pos_weight_np, _ = compute_pos_weight(labeled_idx)
    pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32).to(device)

    
    if use_focal:
        criterion = FocalBCEWithLogits(alpha=1.0, gamma=focal_gamma, pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, verbose=True)

    best_val_f1 = -1.0
    best_state = None

    loss_history = []
    f1_history = []
    auc_history = []

    for r in range(rounds):
        print(f"\n=== ROUND {r+1}/{rounds} | labeled={len(labeled_idx)} ===")

        
        pos_weight_np, labeled_labels_matrix = compute_pos_weight(labeled_idx)
        pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32).to(device)
        if use_focal:
            criterion = FocalBCEWithLogits(alpha=1.0, gamma=focal_gamma, pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        
        class_freq = (labeled_labels_matrix.sum(axis=0) / (labeled_labels_matrix.shape[0] + 1e-12)).astype(np.float32)
        inv_class_freq = (1.0 / (class_freq + 1e-6))
        inv_class_freq = inv_class_freq / (np.mean(inv_class_freq) + 1e-12)
        sample_weights_np = (labeled_labels_matrix * inv_class_freq).sum(axis=1)
        sample_weights_np = sample_weights_np + 0.1  
        sample_weights = torch.tensor(sample_weights_np, dtype=torch.float32)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        train_loader = DataLoader(Subset(pool_dataset, labeled_idx),
                                  batch_size=batch_size, sampler=sampler, num_workers=DEFAULT_NUM_WORKERS)

        
        for epoch in range(epochs):
            avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, use_mixup=use_mixup, mixup_alpha=mixup_alpha)
            print(f"Epoch {epoch+1} done, Loss={avg_loss:.4f}")

        
        probs_val, labels_val = predict_probs(model, val_loader, device)
        if probs_val.shape[0] == 0 or labels_val.shape[0] == 0:
            print("Validation set empty -- skipping threshold tuning.")
            thresholds = np.full(len(ALL_LABELS), 0.5, dtype=np.float32)
            val_macro_f1 = 0.0
            val_macro_auc = np.nan
            per_class_f1_val = np.zeros(len(ALL_LABELS), dtype=np.float32)
        else:
            thresholds, per_class_f1_val = tune_thresholds(probs_val, labels_val, steps=99)
            val_macro_f1, val_macro_auc, per_class_f1_metrics = evaluate_with_thresholds(probs_val, labels_val, thresholds)
            val_macro_f1 = float(val_macro_f1)
            val_macro_auc = float(val_macro_auc)
            print(f"Validation: f1={val_macro_f1:.4f} auc={val_macro_auc:.4f}")

        
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_state = {
                'model': model.state_dict(),
                'thresholds': thresholds,
                'pos_weight': pos_weight_np.copy()
            }
            print("Saved new best model (by val F1).")

        scheduler.step(val_macro_f1)

        loss_history.append(avg_loss)
        f1_history.append(val_macro_f1)
        auc_history.append(val_macro_auc)

        
        if len(unlabeled_idx) == 0:
            print("No more unlabeled samples.")
            break

        if (max_unlabeled_to_score is not None) and (len(unlabeled_idx) > max_unlabeled_to_score):
            unlabeled_idx_sample = random.sample(unlabeled_idx, max_unlabeled_to_score)
            print(f"Unlabeled pool: {len(unlabeled_idx)}. Sampling {len(unlabeled_idx_sample)} for scoring to save time.")
        else:
            unlabeled_idx_sample = list(unlabeled_idx)
            print(f"Scoring entire unlabeled pool: {len(unlabeled_idx_sample)} samples.")

        unl_loader = DataLoader(Subset(pool_dataset, unlabeled_idx_sample),
                                batch_size=batch_size, shuffle=False, num_workers=DEFAULT_NUM_WORKERS)

        model.eval()
        scores = []
        with torch.no_grad():
            for imgs, _ in tqdm(unl_loader, desc="Scoring unlabeled", leave=False):
                imgs = imgs.to(device)
                probs = torch.sigmoid(model(imgs)).cpu().numpy()
                probs = np.clip(probs, 1e-8, 1-1e-8)
                entropy = - (probs * np.log(probs) + (1-probs) * np.log(1-probs))
                # sum over labels -> higher means more uncertain across any label
                sample_entropy = np.sum(entropy, axis=1)
                scores.extend(sample_entropy)

        scores = np.array(scores)
        k = min(query_size, len(scores))
        if k == 0:
            print("No unlabeled samples available to query.")
            break

        selected_rel = np.argsort(scores)[-k:]
        selected_abs = [unlabeled_idx_sample[i] for i in selected_rel]

        labeled_idx.extend(selected_abs)
        unlabeled_idx = [u for u in unlabeled_idx if u not in selected_abs]

        print(f"Selected {len(selected_abs)} samples to add to labeled pool. New labeled size: {len(labeled_idx)}")

   
    if best_state is None:
        print("No best model saved, using current model and 0.5 thresholds.")
        final_thresholds = np.full(len(ALL_LABELS), 0.5, dtype=np.float32)
    else:
        model.load_state_dict(best_state['model'])
        final_thresholds = best_state['thresholds']

    probs_test, labels_test = predict_probs(model, test_loader, device)
    if probs_test.shape[0] == 0:
        test_macro_f1 = 0.0
        test_macro_auc = np.nan
        per_class_f1_test = np.zeros(len(ALL_LABELS), dtype=np.float32)
    else:
        test_macro_f1, test_macro_auc, per_class_f1_test = evaluate_with_thresholds(probs_test, labels_test, final_thresholds)
    print("\nFinal test metrics:")
    print(f"Test macro F1: {test_macro_f1:.4f}  AUC: {test_macro_auc:.4f}")
    print("Sample per-class F1 (first 10):", {ALL_LABELS[i]: float(per_class_f1_test[i]) for i in range(min(10, len(ALL_LABELS)))})

    # plots
    plt.figure(figsize=(10, 4))
    plt.plot(loss_history, marker='o')
    plt.title("Training Loss (avg per round)")
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(f1_history, marker='o')
    plt.title("Validation F1 per Round")
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(auc_history, marker='o')
    plt.title("Validation AUC per Round")
    plt.show()

# Inside active_learning_pipeline, before return
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
    "model_state_dict": best_state['model'] if best_state is not None else model.state_dict(),
    "thresholds": best_state['thresholds'] if best_state is not None else np.full(len(ALL_LABELS), 0.5, dtype=np.float32),
    "pos_weight": best_state['pos_weight'] if best_state is not None else pos_weight_np,
    "loss_history": loss_history,
    "val_f1_history": f1_history,
    "val_auc_history": auc_history,
    "ALL_LABELS": ALL_LABELS}

    save_path = os.path.join(save_dir, "final_model_checkpoint.pth")
    torch.save(checkpoint, save_path)
    print(f"\n Training complete. Everything saved to {save_path}")


    return loss_history, f1_history, auc_history, test_macro_f1, test_macro_auc, per_class_f1_test


if __name__ == "__main__":
    csv_path = "/Users/devapriyansahayagoodwin/Documents/cost_final_code.py/Data_Entry_2017_v2020.csv"
    base_dir = "/Users/devapriyansahayagoodwin/Documents/cost_final_code.py/images/"

    print("Loading metadata...")
    df = load_metadata(csv_path)
    print("Encoding labels + verifying files...")
    df = encode_labels_and_paths(df, base_dir)
    print("Total valid images:", len(df))

    
    visualize_random_images(df)
    plot_label_distribution = lambda d: None 
    

    
    train_pool_df, test_df = create_train_test(df, test_size=0.2, random_state=SEED)

    
    val_frac = 0.10
    if HAS_ML_STRAT:
        msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=SEED)
        X = train_pool_df.index.values.reshape(-1,1)
        Y = train_pool_df[ALL_LABELS].values
        tr_idx, val_idx = next(msss.split(X, Y))
        pool_df = train_pool_df.iloc[tr_idx].reset_index(drop=True)
        val_df = train_pool_df.iloc[val_idx].reset_index(drop=True)
    else:
        
        stratify_col = (train_pool_df[ALL_LABELS].values.sum(axis=1) > 0).astype(int)
        tr_idx, val_idx = train_test_split(np.arange(len(train_pool_df)), test_size=val_frac, random_state=SEED, stratify=stratify_col)
        pool_df = train_pool_df.iloc[tr_idx].reset_index(drop=True)
        val_df = train_pool_df.iloc[val_idx].reset_index(drop=True)

    print("Pool size (for active learning):", len(pool_df))
    print("Validation size:", len(val_df))
    print("Test size:", len(test_df))

    
    loss_hist, f1_hist, auc_hist, test_f1, test_auc, per_class_f1_test = active_learning_pipeline(
        pool_df, val_df, test_df,
        rounds=3,              
        query_size=1000,        
        init_size=5000,         
        batch_size=64,
        image_size=224,
        epochs=4,               
        max_unlabeled_to_score=5000,
        use_focal=True,         
        focal_gamma=2.0,
        use_mixup=False,        
        mixup_alpha=0.4,
        device=None
    )

    print("\nFinal Metrics:")
    print("Loss history:", loss_hist)
    print("Val F1 history:", f1_hist)
    print("Val AUC history:", auc_hist)
    print("Test macro F1:", test_f1)
    print("Test macro AUC:", test_auc)
