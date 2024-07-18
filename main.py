import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm

from src.datasets import ThingsMEGDataset
from src.models import BasicConvClassifier
from src.utils import set_seed
#追加
from scipy.signal import resample, butter, filtfilt
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import Dataset, DataLoader
from PIL import Image


#データセットクラスの拡張
class ThingsMEGDataset(Dataset):
    def __init__(self, split, data_dir, feature_path=None):
        self.data, self.labels = self.load_data(split, data_dir, has_labels=(split != 'test'))
        self.subject_idxs = self.load_subject_idxs(split, data_dir)
        
        if feature_path and split != 'test':  # testセットには特徴量を使用しない
            self.clip_features = torch.load(feature_path)
        
        
        # 脳波データの前処理
        self.data = self.preprocess_eeg(self.data)
    
        # クラス数、シーケンス長、チャネル数を設定
        self.num_classes = 1854  # 実際のクラス数を設定
        self.seq_len = self.data.shape[1] 
        self.num_channels = self.data.shape[2] if len(self.data.shape) > 2 else 1

    def load_data(self, split, data_dir, has_labels=True):
        data_path = os.path.join(data_dir, f'{split}_X.pt')
        data = torch.load(data_path)
        
        if has_labels:
            labels_path = os.path.join(data_dir, f'{split}_y.pt')
            if os.path.exists(labels_path):
                labels = torch.load(labels_path)
            else:
                labels = None
        else:
            labels = None
        
        return data, labels
    
    def load_subject_idxs(self, split, data_dir):
        subject_idxs_path = os.path.join(data_dir, f'{split}_subject_idxs.pt')
        subject_idxs = torch.load(subject_idxs_path)
        return subject_idxs
    
    def preprocess_eeg(self, data):
        if isinstance(data, torch.Tensor):
            data = data.numpy()  # テンソルをnumpy配列に変換
        if data.dtype != np.float32 and data.dtype != np.float64:
            data = data.astype(np.float32)  # 浮動小数点型に変換

        # 脳波データの前処理（例：標準化）
        data = (data - np.mean(data, axis=-1, keepdims=True)) / np.std(data, axis=-1, keepdims=True)
        data = torch.from_numpy(data).permute(0, 2, 1)  # 形式を[N, C, W]に変換
                
        return data
    
    def __getitem__(self, idx):
        if hasattr(self, 'clip_features'):
            return self.data[idx], self.labels[idx], self.subject_idxs[idx], self.clip_features[idx]
        else:
            return self.data[idx], self.subject_idxs[idx]  # テストセットの場合はラベルを返さない
    
    def __len__(self):
        return len(self.labels)


class BasicConvClassifier(nn.Module):
    def __init__(self, num_classes, seq_len, num_channels):
        super(BasicConvClassifier, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * seq_len, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CombinedModel(nn.Module):
    def __init__(self, num_classes, seq_len, num_channels, num_subjects, clip_feature_dim=None):
        super(CombinedModel, self).__init__()
        self.eeg_encoder = BasicConvClassifier(num_classes, seq_len, num_channels, num_subjects)
        if clip_feature_dim:
            self.fc_clip = nn.Linear(clip_feature_dim, 512)
            self.fc_combined = nn.Linear(512 + 512, num_classes)
        else:
            self.fc_combined = nn.Linear(512, num_classes)
        self.use_clip_features = clip_feature_dim is not None
    
    def forward(self, eeg_data, subject_idxs, clip_features=None):
        eeg_features = self.eeg_encoder(eeg_data, subject_idxs)
        if self.use_clip_features:
            clip_features = F.relu(self.fc_clip(clip_features))
            combined_features = torch.cat((eeg_features, clip_features), dim=1)
        else:
            combined_features = eeg_features
        output = self.fc_combined(combined_features)
        return output

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}

    train_set = ThingsMEGDataset("train", args.data_dir, 'data/train_clip_features.pt')
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_set = ThingsMEGDataset("val", args.data_dir ,'data/val_clip_features.pt')
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    test_set = ThingsMEGDataset("test", args.data_dir) # テストセットには特徴量を指定しない
    test_loader = DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
        )


    # ------------------
    #       Model
    # ------------------
    model = CombinedModel(
        train_set.num_classes, train_set.seq_len, train_set.num_channels,
        num_subjects=len(np.unique(train_set.subject_idxs)), clip_feature_dim=512
    ).to(args.device)

    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ------------------
    #   Start training
    # ------------------  
    max_val_acc = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(args.device)
      
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        
        model.train()
        for X, y, subject_idxs, clip_features in tqdm(train_loader, desc="Train"):
            X, y, clip_features = X.to(args.device), y.to(args.device), clip_features.to(args.device)
            subject_idxs = subject_idxs.to(args.device)
            
            y_pred = model(X, subject_idxs, clip_features)
            
            loss = F.cross_entropy(y_pred, y)
            train_loss.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())

        model.eval()
        for X, y, subject_idxs, clip_features in tqdm(val_loader, desc="Validation"):
            X, y, clip_features = X.to(args.device), y.to(args.device), clip_features.to(args.device)
            subject_idxs = subject_idxs.to(args.device)
            
            with torch.no_grad():
                y_pred = model(X, subject_idxs, clip_features)
            
            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())

        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})
        
        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = np.mean(val_acc)

    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))

    preds = [] 
    model.eval()
    for X, subject_idxs, clip_features in tqdm(test_loader, desc="Test"):        
        preds.append(model(X.to(args.device), subject_idxs.to(args.device), clip_features.to(args.device)).detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")


if __name__ == "__main__":
    run()
