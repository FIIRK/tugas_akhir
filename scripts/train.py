import random
import torch
import os

from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from pathlib import Path

from scripts.metrics import DiceBCELoss, F1_score, pixel_accuracy
from scripts.myModels import vgg16_transform
from scripts.utils import augment

def get_data_loader(DATA_PATH, DATA_CLASS, TRAIN_FRACTION=0.8, BATCH_SIZE=5):
    NUM_WORKERS = os.cpu_count()
    image_paths = [str(im_path) for im_path in Path(DATA_PATH + '/images/post/').rglob(pattern='*_post_*.png')]

    random.shuffle(image_paths)
    split_index = int(TRAIN_FRACTION * len(image_paths))
    train_paths = image_paths[:split_index]
    test_paths = image_paths[split_index:]

    # Load data
    train_data = DATA_CLASS(image_paths=train_paths,
                            transform=vgg16_transform,
                            augment=augment)

    test_data  = DATA_CLASS(image_paths=test_paths,
                            transform=vgg16_transform)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    return train_dataloader, test_dataloader

def train_model(model, train_dataloader, test_dataloader, LEARNING_RATE=0.01, NUM_EPOCHS=1, save_path='weights.pkl'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dbce_loss = DiceBCELoss(smooth=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    torch.cuda.empty_cache()
    progress = {}

    for epoch in tqdm(range(NUM_EPOCHS)):
        progress[epoch] = {"train_prc":[], "train_rec":[], "train_acc":[], "train_loss":[],
                           "test_prc" :[], "test_rec" :[], "test_acc" :[], "test_loss" :[]}
        print("TRAIN PHASE")
        model.train()

        for batch_idx, (image, masks) in tqdm(enumerate(train_dataloader)):
            image, masks = image.to(device), masks.to(device)
            optimizer.zero_grad()
            output = model(image)
            dice = dbce_loss(output, masks)
            f1s  = F1_score(output, masks)
            pix  = pixel_accuracy(output, masks)
            progress[epoch]["train_prc"].append(f1s[0])
            progress[epoch]["train_rec"].append(f1s[1])
            progress[epoch]["train_acc"].append(pix.item())
            progress[epoch]["train_loss"].append(dice.item())

            if (batch_idx+1) % 40 == 0 or (batch_idx+1) == 1:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Batch [{batch_idx+1:3d}/{len(train_dataloader)}]," \
                      + f" Loss:{dice.item():.4f}, Acc:{pix.item():.4f}," \
                      + f" Prc:{f1s[0]:.4f}, Rec:{f1s[1]:.4f}")

            dice.backward()
            optimizer.step()

        model.eval()
        print("EVAL PHASE")
        with torch.inference_mode():
            val_loss = 0.0
            for idx, (image, masks) in tqdm(enumerate(test_dataloader)):
                image, masks = image.to(device), masks.to(device)
                output = model(image)
                dice = dbce_loss(output, masks)
                f1s = F1_score(output, masks)
                pix = pixel_accuracy(output, masks)
                progress[epoch]["test_prc"].append(f1s[0])
                progress[epoch]["test_rec"].append(f1s[1])
                progress[epoch]["test_acc"].append(pix.item())
                progress[epoch]["test_model"].append(dice.item())

            avg_acc  = sum(progress[epoch]["test_acc"])  / len(progress[epoch]["test_acc"])
            avg_loss = sum(progress[epoch]["test_loss"]) / len(progress[epoch]["test_loss"])
            print(f"Avg Acc: {avg_acc:.4f}, Avg Loss: {avg_loss:.4f}, Prc:{f1s[0]:.4f}, Rec:{f1s[1]:.4f}")
