import os
import cv2
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from albumentations import Compose, RandomScale, RandomSizedCrop, RandomCrop, Resize, \
    HorizontalFlip, Rotate, Lambda, VerticalFlip, Normalize, \
    CenterCrop, Downscale, ElasticTransform, GaussianBlur, GaussNoise, GridDistortion, \
    HueSaturationValue, MedianBlur, MotionBlur, OneOf, PadIfNeeded, \
    RandomBrightnessContrast, ChannelShuffle, RandomBrightness
from torchmetrics import F1Score
from albumentations.pytorch.transforms import ToTensorV2
from seg_hrnet import hrnet
from yacs.config import CfgNode
import matplotlib.pyplot as plt
import wandb

plt.ion()   # interactive mode

SEED = 7777

def dice_coeff(input, target, reduce_batch_first=False, epsilon=1e-6):
    assert input.size() == target.size(), f"input size = {input.size()}, target size = {target.size()}"
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)

    return dice.mean()

def dice_loss(input, target): return 1 - dice_coeff(input, target)

class HandDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir="./eyth_dataset", mode="train", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.sample_paths = open(os.path.join(self.root_dir, f"train-val-test-split/{mode}.txt")).read().split()

    def __getitem__(self, index):
        sample_path = self.sample_paths[index]
        image_path = os.path.join(self.root_dir, 'images', sample_path)
        mask_path = os.path.join(self.root_dir, 'masks', sample_path.replace('jpg', 'png'))
        image_np = cv2.imread(image_path)
        mask_np = cv2.imread(mask_path, 0)
        mask_np[mask_np==255] = 1

        if self.transform is not None:
            transformed = self.transform(image=image_np, mask=mask_np)
            return transformed['image'], transformed['mask']
        return image_np, mask_np

    def __len__(self):
        return len(self.sample_paths)

def build_model(num_classes):
    cfg = CfgNode()
    cfg.MODEL = CfgNode()
    cfg.MODEL.NUM_CLASSES = num_classes
    model = hrnet(cfg)
    return model

def iou_calculation(pred, y, target_class, eps=1e-9):
    pred_mask = pred == target_class
    y_mask = y == target_class
    intersection = pred_mask & y_mask
    union = pred_mask | y_mask

    return intersection.sum() / (union.sum() + eps)

def eval_model(model, dataloader_test):
    model.eval()
    images_miou = []
    with torch.no_grad():
        for step_idx, (batch_inputs, batch_targets) in enumerate(dataloader_test):
            outputs = model(batch_inputs.cuda())
            # TODO: calculate background IOU and hand IOU
            prediction = outputs.argmax(axis=1)
            background_iou = iou_calculation(prediction, torch.unsqueeze(batch_targets.cuda().long(), 0), 0)
            hand_iou = iou_calculation(prediction, torch.unsqueeze(batch_targets.cuda().long(), 0), 1)

            mIOU = (background_iou + hand_iou) / 2
            if step_idx % 30 == 0:
                print(f'step_idx: {step_idx}, mIOU = {mIOU:.2f}, background: {background_iou:.2f}, hand: {hand_iou:.2f}')
            images_miou.append(mIOU)
    accuracy = sum(images_miou)/len(images_miou)
    print(f'Model accuracy: {accuracy:.2f}')
    return accuracy

def train_model(model, criterion, optimizer, scheduler, dataloader_train, dataloader_val, num_epoches):
    # loop epoch
    best_performace = float("inf")
    EarlyStop = 15
    tag = [
           "VerticalFlip",
           "HorizontalFlip",
           "RandomScale",
           "ElasticTransform",
           "HueSaturationValue",
           "ChannelShuffle",
           ]
    wandb.init(project="Hand segmentation",
               entity="stanleyluuuu",
               name="Baseline-BCELoss+DiceLoss",
               config={'learning_rate':1e-2, 'Loss Function':"Cross Entropy", 'n_epochs':num_epoches, 'Early Stop': EarlyStop},
               tags=tag)
    log_dict = {}
    f1 = F1Score(task="binary", num_classes=2).cuda()
    for epoch_idx in range(num_epoches):
        print(f'Epoch: {epoch_idx}, learning rate:', optimizer.param_groups[0]['lr'])
        # loop dataloader_train
        model.train()
        train_loss_total = 0
        for step_idx, (batch_inputs, batch_targets) in enumerate(dataloader_train):
            # TODO implement train step: forward pass, calculate loss, optimize model
            optimizer.zero_grad()

            outputs = model(batch_inputs.cuda())
            
            loss = criterion(outputs, F.one_hot(batch_targets.cuda().long(), 2).permute(0, 3, 1, 2).float())
            # loss = criterion(outputs, batch_targets.cuda().long()) # Cross Entorpy Loss
            loss += dice_loss(F.softmax(outputs, dim=1).float(), F.one_hot(batch_targets.cuda().long(), 2).permute(0, 3, 1, 2).float())

            loss.backward()
            train_loss_total += loss
            
            optimizer.step()
            if step_idx % 10 == 0:
                print(f"Epoch: {epoch_idx}, Step: {step_idx + 1}/{len(dataloader_train)}, Loss: {loss}")
        if scheduler is not None:
            scheduler.step()
        train_loss = train_loss_total / len(dataloader_train)
        log_dict.update([("Train/Loss", train_loss)])
        # loop dataloader_val to calculate val_loss for validation set
        model.eval()
        val_loss_total = 0
        val_score_total = 0
        with torch.no_grad():
            for batch_inputs, batch_targets in dataloader_val:

                # TODO implement valid step: forward pass, calculate loss,
                outputs = model(batch_inputs.cuda())
                loss = criterion(outputs, F.one_hot(batch_targets.cuda().long(), 2).permute(0, 3, 1, 2).float())
                # loss = criterion(outputs, batch_targets.cuda().long()) # Cross Entorpy Loss
                loss += dice_loss(F.softmax(outputs, dim=1).float(), F.one_hot(batch_targets.cuda().long(), 2).permute(0, 3, 1, 2).float())
                score = f1(outputs.argmax(axis=1), batch_targets.cuda().long())

                val_score_total += score
                val_loss_total += loss
        val_loss = val_loss_total / len(dataloader_val)
        val_score = val_score_total / len(dataloader_val)
        val_accuracy = eval_model(model, dataloader_val)

        log_dict.update([("Val/Loss", val_loss), ("Val/Accuracy", val_accuracy), ("Val/F1Score", val_score)])

        wandb.log(log_dict)

        if val_loss < best_performace:
            best_performace = val_loss
            early_stop_count = 0
        else: early_stop_count += 1

        if early_stop_count >= EarlyStop: break

        print(f"Epoch: {epoch_idx}, val_loss = {val_loss}, val_accuracy = {val_accuracy:.2f}")

def set_seed():
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)

if __name__ == "__main__":
    set_seed()
    dataset = HandDataset(mode="train")
    image, mask = dataset[0]
    print("Num of training samples", len(dataset), "\nImage shape:", image.shape, "\nMask shape:", mask.shape)

    max_sample_to_show = 20
    fig, axs = plt.subplots(2,5, sharey=True)
    fig.set_figwidth(20)
    for i in range(max_sample_to_show):
        image, mask = dataset[i]
        image_rbg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axs[0, i%5].imshow(Image.fromarray(image_rbg))  # we convert numpy image to Pillow Image, in order to plot them
        axs[1, i%5].imshow(Image.fromarray(mask))
        print(f"mask {i} unique values:", np.unique(mask))
        if i % 5 == 4:
            plt.show()
            if i < max_sample_to_show - 1:
                fig, axs = plt.subplots(2, 5, sharey=True)
                fig.set_figwidth(20)

    transform_test = Compose([Normalize(), ToTensorV2()])
    # TODO define preprocessing for training set
    transform_train = Compose([Normalize(),
                               RandomScale(scale_limit=[0.5, 2]),
                               HorizontalFlip(p=0.5),
                               VerticalFlip(p=0.5),
                               ElasticTransform(p=0.5),
                               RandomCrop(200, 350),
                               HueSaturationValue(),
                               ChannelShuffle(),
                               ToTensorV2(),])
    
    dataset_train = HandDataset(mode="train", transform=transform_train)
    dataset_val = HandDataset(mode="val", transform=transform_test)
    dataset_test = HandDataset(mode="train", transform=transform_test)

    # TODO: set the number of classes
    num_classes = 2

    model = build_model(num_classes=num_classes)
    model.cuda()

    # we try to feed a validation image into the model
    x, y = dataset_val[0]
    outputs = model(torch.stack([x]).cuda())[0]
    print("model outputs:", x.shape, y.shape, outputs.shape)

    # define pytorch dataloader
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=3, num_workers=2)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, num_workers=2)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, num_workers=2)

    # TODO define loss function
    criterion = nn.BCEWithLogitsLoss()

    # TODO define optimizer to optimize model weight
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

    # TODO define scheduler to update learning rate
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # TODO define number of epoch to train
    num_epoches = 80

    # train the model
    train_model(model, criterion, optimizer, scheduler, dataloader_train, dataloader_val, num_epoches)
    # calculate model accuracy on test set
    eval_model(model, dataloader_test)

    # visualize predicted hang segmentation mask

    dataset = HandDataset(mode="test")
    for i in range(50):
        fig, axs = plt.subplots(1, 3, sharey=True)
        fig.set_figwidth(20)
        image, mask = dataset[i]

        # TODO: inference image and get the predicted_mask
        input = transform_test(image=image)["image"]
        input = torch.unsqueeze(input.cuda(), 0)

        output = model(input)
        prediction = output.argmax(axis=1).cpu().numpy()
        predicted_mask = np.uint8(np.squeeze(prediction))

        image_rbg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print("Image", i)
        axs[0].imshow(Image.fromarray(image_rbg))  # we convert numpy image to Pillow Image, in order to plot them
        axs[1].imshow(Image.fromarray(mask))
        axs[2].imshow(Image.fromarray(predicted_mask))
        plt.show()