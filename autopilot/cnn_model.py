import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models import (
    MobileNet_V2_Weights,
    VGG16_Weights,
    mobilenet_v2,
    vgg16,
)

BACKBONE_SPECS = {
    "vgg16": {
        "builder": lambda: vgg16(weights=VGG16_Weights.IMAGENET1K_V1),
        "weights": VGG16_Weights.IMAGENET1K_V1,
        "flatten_dim": 512 * 7 * 7,
        "pool": False,
    },
    "mobilenet_v2": {
        "builder": lambda: mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1),
        "weights": MobileNet_V2_Weights.IMAGENET1K_V1,
        "flatten_dim": 1280,
        "pool": True,
    },
}


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class BackboneFeatureExtractor:
    def __init__(self, backbone_name, device=None):
        if backbone_name not in BACKBONE_SPECS:
            raise ValueError(f"Unknown backbone '{backbone_name}'. Options: {list(BACKBONE_SPECS)}")
        spec = BACKBONE_SPECS[backbone_name]
        self.backbone_name = backbone_name
        self.device = device or default_device()
        self.model = spec["builder"]().features.to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.transform = spec["weights"].transforms()
        self.pool = spec["pool"]
        self.flatten_dim = spec["flatten_dim"]

    @torch.no_grad()
    def extract(self, images, batch_size=32):
        chunks = []
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            tensors = torch.stack(
                [self.transform(Image.fromarray(img)) for img in batch]
            ).to(self.device)
            out = self.model(tensors)
            if self.pool:
                out = F.adaptive_avg_pool2d(out, 1)
            chunks.append(out.flatten(1).cpu())
        return torch.cat(chunks, dim=0)


class ClassifierHead(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=100, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class CNNClassifierAdapter:
    def __init__(self, extractor: BackboneFeatureExtractor, head: ClassifierHead, class_names):
        self.extractor = extractor
        self.head = head
        self.class_names = class_names
        self.device = extractor.device

    def predict_proba_images(self, images):
        self.head.eval()
        with torch.no_grad():
            features = self.extractor.extract(images).to(self.device)
            logits = self.head(features)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        return proba

    def save(self, path):
        torch.save(
            {
                "head_state": self.head.state_dict(),
                "class_names": self.class_names,
                "backbone": self.extractor.backbone_name,
                "flatten_dim": self.extractor.flatten_dim,
            },
            path,
        )

    @classmethod
    def load(cls, path, device=None):
        device = device or default_device()
        payload = torch.load(path, map_location=device, weights_only=False)
        extractor = BackboneFeatureExtractor(payload["backbone"], device)
        head = ClassifierHead(payload["flatten_dim"], len(payload["class_names"])).to(device)
        head.load_state_dict(payload["head_state"])
        head.eval()
        return cls(extractor, head, payload["class_names"])


def train_cnn(
    x_train,
    y_train,
    x_test,
    class_names,
    backbone="vgg16",
    epochs=15,
    batch_size=16,
    learning_rate=1e-3,
    device=None,
):
    device = device or default_device()
    extractor = BackboneFeatureExtractor(backbone, device)

    train_features = extractor.extract(x_train).to(device)
    test_features = extractor.extract(x_test).to(device)
    train_labels = torch.as_tensor(y_train, dtype=torch.long, device=device)

    head = ClassifierHead(extractor.flatten_dim, len(class_names)).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    num_samples = train_features.shape[0]
    for _ in range(epochs):
        head.train()
        permutation = torch.randperm(num_samples)
        for start in range(0, num_samples, batch_size):
            idx = permutation[start : start + batch_size]
            if idx.numel() <= 1:
                continue
            optimizer.zero_grad()
            outputs = head(train_features[idx])
            loss = criterion(outputs, train_labels[idx])
            loss.backward()
            optimizer.step()

    head.eval()
    with torch.no_grad():
        y_pred = head(test_features).argmax(dim=1).cpu().numpy()

    adapter = CNNClassifierAdapter(extractor, head, class_names)
    return adapter, y_pred
