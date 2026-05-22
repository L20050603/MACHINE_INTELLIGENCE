import argparse
import io
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from .mnist_downloader import ensure_mnist_available


# ──────────────────────────────────────────────
# Conditional Batch Normalization (P2)
# ──────────────────────────────────────────────

class ConditionalBatchNorm1d(nn.Module):
    """BatchNorm1d modulated by class label: gamma(y) * BN(x) + beta(y)."""

    def __init__(self, num_features, num_classes, embed_dim=16):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.embed = nn.Embedding(num_classes, embed_dim)
        self.gamma = nn.Linear(embed_dim, num_features)
        self.beta = nn.Linear(embed_dim, num_features)
        # start as identity: gamma=1, beta=0
        nn.init.constant_(self.gamma.weight, 0.0)
        nn.init.constant_(self.gamma.bias, 1.0)
        nn.init.constant_(self.beta.weight, 0.0)
        nn.init.constant_(self.beta.bias, 0.0)

    def forward(self, x, labels):
        out = self.bn(x)
        gamma = self.gamma(self.embed(labels))
        beta = self.beta(self.embed(labels))
        return gamma * out + beta


# ──────────────────────────────────────────────
# Generator (P2: ConditionalBatchNorm)
# ──────────────────────────────────────────────

class ConditionalGenerator(nn.Module):
    def __init__(self, noise_dim=96, num_classes=10, embed_dim=16):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, embed_dim)

        self.fc1 = nn.Linear(noise_dim + embed_dim, 256)
        self.cbn1 = ConditionalBatchNorm1d(256, num_classes)
        self.fc2 = nn.Linear(256, 512)
        self.cbn2 = ConditionalBatchNorm1d(512, num_classes)
        self.fc3 = nn.Linear(512, 1024)
        self.cbn3 = ConditionalBatchNorm1d(1024, num_classes)
        self.fc4 = nn.Linear(1024, 28 * 28)

    def forward(self, noise, labels):
        x = torch.cat([noise, self.label_embed(labels)], dim=1)
        x = F.leaky_relu(self.cbn1(self.fc1(x), labels), 0.2, inplace=True)
        x = F.leaky_relu(self.cbn2(self.fc2(x), labels), 0.2, inplace=True)
        x = F.leaky_relu(self.cbn3(self.fc3(x), labels), 0.2, inplace=True)
        x = self.fc4(x)
        return torch.tanh(x).view(-1, 1, 28, 28)


# ──────────────────────────────────────────────
# Discriminator (P1: Spectral Normalization)
# ──────────────────────────────────────────────

class AuxiliaryDiscriminator(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(1, 32, 4, 2, 1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(32, 64, 4, 2, 1))
        self.bn = nn.BatchNorm2d(64)
        self.fc1 = nn.utils.spectral_norm(nn.Linear(64 * 7 * 7, 256))
        self.validity = nn.utils.spectral_norm(nn.Linear(256, 1))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, images):
        feat = F.leaky_relu(self.conv1(images), 0.2, inplace=True)
        feat = F.leaky_relu(self.bn(self.conv2(feat)), 0.2, inplace=True)
        feat = feat.flatten(1)
        feat = F.leaky_relu(self.fc1(feat), 0.2, inplace=True)
        return self.validity(feat), self.classifier(feat)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _to_model_space(images):
    return images * 2.0 - 1.0


def _to_image_space(images):
    return (images + 1.0) / 2.0


def load_generator(checkpoint_path, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    noise_dim = checkpoint.get("noise_dim", 96)
    generator = ConditionalGenerator(noise_dim=noise_dim).to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    return generator, noise_dim, device


def load_acgan(checkpoint_path, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    noise_dim = checkpoint.get("noise_dim", 96)
    generator = ConditionalGenerator(noise_dim=noise_dim).to(device)
    discriminator = AuxiliaryDiscriminator().to(device)
    generator.load_state_dict(checkpoint["generator"])
    discriminator.load_state_dict(checkpoint["discriminator"])
    generator.eval()
    discriminator.eval()
    return generator, discriminator, noise_dim, device


# ──────────────────────────────────────────────
# Training (P0: hinge loss, label smoothing, class weight)
# ──────────────────────────────────────────────

def train_acgan(
    data_dir="./data",
    output_dir="./generated",
    epochs=3,
    batch_size=128,
    noise_dim=96,
    lr=0.0002,
    sample_every=1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(_to_model_space)])
    ensure_mnist_available(data_dir, transform=transform)
    loader = DataLoader(
        datasets.MNIST(data_dir, train=True, download=False, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    generator = ConditionalGenerator(noise_dim=noise_dim).to(device)
    discriminator = AuxiliaryDiscriminator().to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(1, epochs + 1):
        g_loss_total = 0.0
        d_loss_total = 0.0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            batch = images.size(0)

            # ── generate fakes ──
            noise = torch.randn(batch, noise_dim, device=device)
            fake_labels = torch.randint(0, 10, (batch,), device=device)
            fake_images = generator(noise, fake_labels)

            # ── train discriminator ──
            opt_d.zero_grad()

            real_validity, real_class = discriminator(images)
            fake_validity, fake_class = discriminator(fake_images.detach())

            # P0: hinge adversarial loss
            d_adv_real = F.relu(1.0 - real_validity).mean()
            d_adv_fake = F.relu(1.0 + fake_validity).mean()

            # P1: label smoothing on classifier
            d_cls_real = F.cross_entropy(real_class, labels, label_smoothing=0.1)
            d_cls_fake = F.cross_entropy(fake_class, fake_labels, label_smoothing=0.1)

            d_loss = d_adv_real + d_adv_fake + d_cls_real + d_cls_fake
            d_loss.backward()
            opt_d.step()

            # ── train generator ──
            opt_g.zero_grad()

            validity, class_logits = discriminator(fake_images)

            # P0: hinge generator loss
            g_adv = -validity.mean()
            # P0: triple class loss weight to force label respect
            g_cls = F.cross_entropy(class_logits, fake_labels)
            g_loss = g_adv + 3.0 * g_cls

            g_loss.backward()
            opt_g.step()

            g_loss_total += g_loss.item()
            d_loss_total += d_loss.item()

        print(
            f"Epoch {epoch:02d} | "
            f"G loss={g_loss_total / len(loader):.4f} | D loss={d_loss_total / len(loader):.4f}"
        )
        if epoch % sample_every == 0:
            save_digit_grid(generator, output / f"acgan_epoch_{epoch:02d}.png", noise_dim=noise_dim, device=device)

    torch.save(
        {
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "noise_dim": noise_dim,
            "epochs": epochs,
        },
        output / "acgan_mnist.pth",
    )
    return generator


# ──────────────────────────────────────────────
# Generation utilities (unchanged API)
# ──────────────────────────────────────────────

@torch.no_grad()
def save_digit_grid(generator, path, noise_dim=96, device=None):
    device = device or next(generator.parameters()).device
    generator.eval()
    labels = torch.arange(10, device=device).repeat_interleave(8)
    noise = torch.randn(len(labels), noise_dim, device=device)
    images = _to_image_space(generator(noise, labels)).clamp(0, 1)
    save_image(images, path, nrow=8)
    generator.train()


@torch.no_grad()
def generate_numpy_samples(
    checkpoint_path,
    per_class=100,
    noise_dim=None,
    device=None,
    min_class_confidence=0.0,
    oversample_factor=6,
):
    generator, discriminator, checkpoint_noise_dim, device = load_acgan(checkpoint_path, device=device)
    noise_dim = noise_dim or checkpoint_noise_dim

    all_images = []
    all_labels = []
    for digit in range(10):
        accepted = []
        attempts = 0
        max_attempts = max(1, oversample_factor) * per_class
        while sum(chunk.size(0) for chunk in accepted) < per_class and attempts < max_attempts:
            batch_size = min(256, max(per_class, per_class - sum(chunk.size(0) for chunk in accepted)))
            labels = torch.full((batch_size,), digit, dtype=torch.long, device=device)
            noise = torch.randn(batch_size, noise_dim, device=device)
            generated = generator(noise, labels)

            if min_class_confidence > 0:
                _, class_logits = discriminator(generated)
                class_probs = F.softmax(class_logits, dim=1)
                keep = (class_probs.argmax(dim=1) == digit) & (class_probs[:, digit] >= min_class_confidence)
                generated = generated[keep]

            if generated.numel() > 0:
                accepted.append(_to_image_space(generated).clamp(0, 1).cpu())
            attempts += batch_size

        if accepted:
            images = torch.cat(accepted, dim=0)[:per_class]
        else:
            labels = torch.full((per_class,), digit, dtype=torch.long, device=device)
            noise = torch.randn(per_class, noise_dim, device=device)
            images = _to_image_space(generator(noise, labels)).clamp(0, 1).cpu()

        if images.size(0) < per_class:
            print(f"Warning: only accepted {images.size(0)}/{per_class} synthetic samples for digit {digit}")

        all_images.append(images.numpy().reshape(images.size(0), -1))
        all_labels.append(torch.full((images.size(0),), digit).numpy())
    return all_images, all_labels


@torch.no_grad()
def generate_digit_tensors(checkpoint_path, digit=0, count=6, device=None):
    generator, noise_dim, device = load_generator(checkpoint_path, device=device)
    labels = torch.full((count,), int(digit), dtype=torch.long, device=device)
    noise = torch.randn(count, noise_dim, device=device)
    return _to_image_space(generator(noise, labels)).clamp(0, 1).cpu()


@torch.no_grad()
def generate_digit_png(checkpoint_path, digit=0, count=6):
    images = generate_digit_tensors(checkpoint_path, digit=digit, count=count)
    buffer = io.BytesIO()
    save_image(images, buffer, format="PNG", nrow=count)
    buffer.seek(0)
    return buffer.read()


def parse_args():
    parser = argparse.ArgumentParser(description="Train an ACGAN to generate label-controlled MNIST samples.")
    base_dir = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-dir", default=str(base_dir / "data"))
    parser.add_argument("--output-dir", default=str(base_dir / "artifacts" / "generated"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--noise-dim", type=int, default=96)
    parser.add_argument("--lr", type=float, default=0.0002)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_acgan(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        noise_dim=args.noise_dim,
        lr=args.lr,
    )