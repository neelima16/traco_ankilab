import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from sklearn.model_selection import train_test_split

# ─── CONFIG ─────────────────────────────────────────
IMAGE_DIR = '/home/hpc/tovl/tovl104v/traco_ankilab/batch_processed_dataset/merged_label_images'
ANN_DIR   = '/home/hpc/tovl/tovl104v/traco_ankilab/batch_processed_dataset/merged_label_annotations'
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── STEP 1: Build Class Mapping ──────────────────────
class_names = set()
for fname in os.listdir(ANN_DIR):
    if fname.endswith('.xml'):
        root = ET.parse(os.path.join(ANN_DIR, fname)).getroot()
        for obj in root.findall('object'):
            class_names.add(obj.find('name').text)

class_names = sorted(class_names)
class_to_id = {name: idx + 1 for idx, name in enumerate(class_names)}
id_to_class = {v: k for k, v in class_to_id.items()}
NUM_CLASSES = len(class_names) + 1  # include background

print(f"📦 Detected classes: {class_to_id} (Total classes including background: {NUM_CLASSES})")

# ─── STEP 2: Dataset Definition ──────────────────────
class HexbugDataset(Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_file = self.files[idx]
        ann_file = img_file.replace('.jpg', '.xml')
        img = Image.open(os.path.join(IMAGE_DIR, img_file)).convert('RGB')
        orig_w, orig_h = img.size

        boxes, labels = [], []
        root = ET.parse(os.path.join(ANN_DIR, ann_file)).getroot()
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(class_to_id[obj.find('name').text])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        if self.transforms:
            img, boxes = self.transforms(img, boxes, orig_w, orig_h)

        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([idx])}
        return F.to_tensor(img), target

# ─── STEP 3: Transforms & Collate ───────────────────
def resize_transform(img, boxes, w, h):
    img = F.resize(img, [300, 300])
    scale_x, scale_y = 300.0 / w, 300.0 / h
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y
    return img, boxes

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return tuple(zip(*batch))

# ─── STEP 4: Prepare Data Loaders ───────────────────
all_imgs = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
train_imgs, val_imgs = train_test_split(all_imgs, test_size=0.2, random_state=42)

train_set = HexbugDataset(train_imgs, transforms=resize_transform)
val_set   = HexbugDataset(val_imgs, transforms=resize_transform)

train_loader = DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_set, batch_size=8, shuffle=False, collate_fn=collate_fn)

print(f"🗂️ Train samples: {len(train_set)}, Val samples: {len(val_set)}")

# ─── STEP 5: Build Model ────────────────────────────
print("📦 Loading SSD model...")

ssd_pretrained = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
model = ssd300_vgg16(weights=None, num_classes=NUM_CLASSES)
model.backbone.load_state_dict(ssd_pretrained.backbone.state_dict())
model = model.to(DEVICE)

# ─── STEP 6: Optimizer Setup ────────────────────────
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=1e-3, momentum=0.9, weight_decay=5e-4)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# ─── STEP 7: Resume Checkpoint if Exists ────────────
checkpoint_path = 'ssd300_multiclass_final_50.pth'
start_epoch = 1
best_val = float('inf')

if os.path.exists(checkpoint_path):
    print(f"🔄 Resuming from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    best_val = checkpoint.get('best_val_loss', float('inf'))
    start_epoch = checkpoint['epoch'] + 1

# ─── STEP 8: Train & Validate ───────────────────────
def train_one_epoch():
    model.train()
    total = 0
    for imgs, tgts in train_loader:
        imgs = [i.to(DEVICE) for i in imgs]
        tgts = [{k: v.to(DEVICE) for k, v in t.items()} for t in tgts]
        loss_dict = model(imgs, tgts)
        losses = sum(loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        total += losses.item()
    return total / len(train_loader)

def validate():
    model.train()  # Use train mode for loss computation
    total = 0
    with torch.no_grad():
        for imgs, tgts in val_loader:
            imgs = [i.to(DEVICE) for i in imgs]
            tgts = [{k: v.to(DEVICE) for k, v in t.items()} for t in tgts]
            loss_dict = model(imgs, tgts)
            total += sum(loss_dict.values()).item()
    model.eval()
    return total / len(val_loader)

# ─── STEP 9: Training Loop ───────────────────────────
for epoch in range(start_epoch, 201):
    train_loss = train_one_epoch()
    val_loss = validate()
    lr_scheduler.step()
    print(f"📈 Epoch {epoch:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'best_val_loss': best_val,
            'num_classes': NUM_CLASSES
        }, 'ssd300_multiclass_best_50.pth')
        print("✅ Saved best model.")

# ─── STEP 10: Save Final Model ───────────────────────
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
    'best_val_loss': best_val,
    'num_classes': NUM_CLASSES
}, 'ssd300_multiclass_final_50.pth')
print("🎉 Training complete! Best val loss:", best_val)
