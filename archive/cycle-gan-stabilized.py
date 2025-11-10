
import itertools
import pickle
import random
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

# ------------------------------
# Device & Seeds
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(88)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(88)
np.random.seed(88)
random.seed(88)

# ------------------------------
# Dataset
# ------------------------------
def get_transform_rainfall(grayscale=False, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

class RainfallDataset(Dataset):
    def __init__(self, pickle_file_A, pickle_file_B, input_nc, output_nc) -> None:
        super().__init__()
        self.dir_A = pickle_file_A
        self.dir_B = pickle_file_B
        with open(self.dir_A, "rb") as f:
            data_A = pickle.load(f)
            self.A_dates = list(data_A.keys())
            self.A_images = list(data_A.values())
        with open(self.dir_B, "rb") as f:
            data_B = pickle.load(f)
            self.B_dates = list(data_B.keys())
            self.B_images = list(data_B.values())
        self.A_size = len(self.A_images)
        self.B_size = len(self.B_images)
        self.transform_A = get_transform_rainfall(grayscale=(input_nc == 1))
        self.transform_B = get_transform_rainfall(grayscale=(output_nc == 1))

    def __len__(self):
        # use max as in original, but guard indexing by modulo to avoid IndexError
        return max(self.A_size, self.B_size)

    def __getitem__(self, index):
        idx_A = index % self.A_size
        idx_B = index % self.B_size

        A_image = self.A_images[idx_A]
        B_image = self.B_images[idx_B]

        # Assuming A_image is an array-like; A_image[2] was used originally. Keep same if present.
        A_arr = A_image[2] # if isinstance(A_image, (list, tuple)) and len(A_image) > 2 else A_image
        B_arr = B_image

        A_img = Image.fromarray(A_arr)
        B_img = Image.fromarray(B_arr)

        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        return A, B

# ------------------------------
# Norm Layer Helper (InstanceNorm per CycleGAN)
# ------------------------------
def get_norm(channels):
    return nn.InstanceNorm2d(channels, affine=False, track_running_stats=False)

# ------------------------------
# Resnet Block (with InstanceNorm)
# ------------------------------
class ResnetBlock(nn.Module):
    """Define a Resnet block with InstanceNorm and ReflectionPad as in CycleGAN"""
    def __init__(self, dim, padding_type="reflect", use_dropout=True, use_bias=True):
        super().__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), get_norm(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), get_norm(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, X):
        return X + self.conv_block(X)

# ------------------------------
# Generators (Resnet-based)
# ------------------------------
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, ngf=64, n_blocks=6, padding_type="reflect", use_dropout=True, use_bias=True):
        super().__init__()
        assert(n_blocks >= 0)
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 get_norm(ngf),
                 nn.ReLU(True)]

        # downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      get_norm(ngf * mult * 2),
                      nn.ReLU(True)]

        # resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type, use_dropout, use_bias)]

        # upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2,
                                         padding=1, output_padding=1, bias=use_bias),
                      get_norm(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, X):
        return self.model(X)

# ------------------------------
# Discriminator (PatchGAN 70x70)
# ------------------------------
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, use_bias=True):
        super().__init__()
        kw, padw = 4, 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                    nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                         get_norm(ndf * nf_mult),
                         nn.LeakyReLU(0.2, True)]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                     get_norm(ndf * nf_mult),
                     nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, X):
        return self.model(X)

# ------------------------------
# GAN Loss
# ------------------------------
class GANLoss(nn.Module):
    def __init__(self, gan_mode="lsgan", target_real_label=1.0, target_fake_label=0.0):
        super().__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ["wgangp"]:
            self.loss = None
        else:
            raise NotImplementedError("gan mode %s not implemented" % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        target_tensor = self.real_label if target_is_real else self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        if self.gan_mode in ["lsgan", "vanilla"]:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == "wgangp":
            loss = -prediction.mean() if target_is_real else prediction.mean()
        return loss

# ------------------------------
# Image Pool (Replay Buffer)
# ------------------------------
class ImagePool:
    def __init__(self, size=50):
        self.size = size
        self.pool = []

    def query(self, images):
        # images: tensor (B, C, H, W)
        if self.size <= 0:
            return images
        return_list = []
        for img in images:
            img = img.detach()
            if len(self.pool) < self.size:
                self.pool.append(img.cpu())
                return_list.append(img)
            else:
                if np.random.rand() > 0.5:
                    idx = np.random.randint(0, self.size)
                    tmp = self.pool[idx].clone()
                    self.pool[idx] = img.cpu()
                    return_list.append(tmp.to(img.device))
                else:
                    return_list.append(img)
        return torch.stack(return_list, dim=0)

# ------------------------------
# Hyperparameters
# ------------------------------
INPUT_NC = 1
OUTPUT_NC = 1
NDF = 64
NGF = 64
USE_BIAS = False
DISCRIMINATOR_LAYERS = 3
N_RESNET_LAYERS = 6
PADDING_TYPE = "reflect"
USE_DROPOUT = True
GAN_MODE = "lsgan"
LEARNING_RATE = 0.0002
BETA_1 = 0.5
BETA_2 = 0.999
EPOCHS = 2
BATCH_SIZE = 16
PICKLE_FILE_PATH_A = "./data/final_era5.pkl"
PICKLE_FILE_PATH_B = "./data/final_chirps.pkl"
LAMBDA_A = 10  # cycle A->B->A
LAMBDA_B = 10  # cycle B->A->B
IDT_MULT = 0.5 # as in CycleGAN: lambda_idt = 0.5 * lambda_cyc

# LR schedule
START_DECAY_EPOCH = EPOCHS // 2

# ------------------------------
# Build Models
# ------------------------------
G_A = ResnetGenerator(INPUT_NC, OUTPUT_NC, NGF, N_RESNET_LAYERS, PADDING_TYPE, USE_DROPOUT, USE_BIAS)
G_B = ResnetGenerator(OUTPUT_NC, INPUT_NC, NGF, N_RESNET_LAYERS, PADDING_TYPE, USE_DROPOUT, USE_BIAS)
D_A = NLayerDiscriminator(OUTPUT_NC, NDF, DISCRIMINATOR_LAYERS, USE_BIAS)  # D_A judges real_B vs fake_B
D_B = NLayerDiscriminator(INPUT_NC, NDF, DISCRIMINATOR_LAYERS, USE_BIAS)   # D_B judges real_A vs fake_A

G_A.to(device); G_B.to(device); D_A.to(device); D_B.to(device)

# ------------------------------
# Data
# ------------------------------
dataset = RainfallDataset(PICKLE_FILE_PATH_A, PICKLE_FILE_PATH_B, INPUT_NC, OUTPUT_NC)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# ------------------------------
# Losses & Optims
# ------------------------------
criterionGAN = GANLoss(GAN_MODE).to(device)
criterionCycle = nn.L1Loss()
criterionIdt = nn.L1Loss()

optimizer_G = torch.optim.Adam(itertools.chain(G_A.parameters(), G_B.parameters()), lr=LEARNING_RATE, betas=(BETA_1, BETA_2))
optimizer_D = torch.optim.Adam(itertools.chain(D_A.parameters(), D_B.parameters()), lr=LEARNING_RATE, betas=(BETA_1, BETA_2))

# Linear LR decay
def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g["lr"] = lr

# ------------------------------
# Fake Pools
# ------------------------------
fake_A_pool = ImagePool(50)
fake_B_pool = ImagePool(50)

# ------------------------------
# Training
# ------------------------------
print("Using device:", device)
initial_lr = LEARNING_RATE

for epoch in range(1, EPOCHS + 1):
    # Linear LR decay after START_DECAY_EPOCH
    if epoch > START_DECAY_EPOCH:
        lr_scale = 1 - float(epoch - START_DECAY_EPOCH) / float(EPOCHS - START_DECAY_EPOCH + 1e-8)
        set_lr(optimizer_G, initial_lr * lr_scale)
        set_lr(optimizer_D, initial_lr * lr_scale)

    epoch_loss_G = 0.0
    epoch_loss_D_A = 0.0
    epoch_loss_D_B = 0.0

    for real_A, real_B in tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}", colour="GREEN"):
        real_A = real_A.to(device)
        real_B = real_B.to(device)

        # -------------------------
        # 1) Update Discriminators first
        # -------------------------
        for p in D_A.parameters(): p.requires_grad = True
        for p in D_B.parameters(): p.requires_grad = True
        optimizer_D.zero_grad()

        # Generate fakes (no_grad not used; gradients won't flow into G since we detach for D)
        fake_B = G_A(real_A)
        fake_A = G_B(real_B)

        # Use image pools for fakes to stabilize D
        fake_B_for_D = fake_B_pool.query(fake_B)
        fake_A_for_D = fake_A_pool.query(fake_A)

        # D_A: real_B vs fake_B
        loss_D_A_real = criterionGAN(D_A(real_B), True)
        loss_D_A_fake = criterionGAN(D_A(fake_B_for_D.detach()), False)
        loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5
        loss_D_A.backward()

        # D_B: real_A vs fake_A
        loss_D_B_real = criterionGAN(D_B(real_A), True)
        loss_D_B_fake = criterionGAN(D_B(fake_A_for_D.detach()), False)
        loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5
        loss_D_B.backward()

        optimizer_D.step()

        # -------------------------
        # 2) Update Generators
        # -------------------------
        for p in D_A.parameters(): p.requires_grad = False
        for p in D_B.parameters(): p.requires_grad = False
        optimizer_G.zero_grad()

        # Recompute fakes for G step (fresh graph)
        fake_B = G_A(real_A)
        fake_A = G_B(real_B)

        # GAN losses (want D to classify fakes as real)
        loss_G_A = criterionGAN(D_A(fake_B), True)
        loss_G_B = criterionGAN(D_B(fake_A), True)

        # Cycle consistency
        rec_A = G_B(fake_B)
        rec_B = G_A(fake_A)
        loss_cycle_A = criterionCycle(rec_A, real_A) * LAMBDA_A
        loss_cycle_B = criterionCycle(rec_B, real_B) * LAMBDA_B

        # Identity losses
        lambda_idt_A = IDT_MULT * LAMBDA_A
        lambda_idt_B = IDT_MULT * LAMBDA_B
        idt_A = criterionIdt(G_A(real_B), real_B) * lambda_idt_A
        idt_B = criterionIdt(G_B(real_A), real_A) * lambda_idt_B

        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + idt_A + idt_B
        loss_G.backward()
        optimizer_G.step()

        epoch_loss_G += loss_G.item()
        epoch_loss_D_A += loss_D_A.item()
        epoch_loss_D_B += loss_D_B.item()

    n_batches = len(dataloader)
    print(f"[Epoch {epoch:03d}] D_A: {epoch_loss_D_A/n_batches:.4f} | D_B: {epoch_loss_D_B/n_batches:.4f} | G: {epoch_loss_G/n_batches:.4f} | lr: {optimizer_G.param_groups[0]['lr']:.6f}")

# ------------------------------
# Quick sanity check visualization on one sample
# ------------------------------
G_A.eval()
with torch.inference_mode():
    A, B = next(iter(dataloader))
    A = A.to(device)
    B = B.to(device)
    fake_B = G_A(A)
    rec_A = G_B(fake_B)
    img_in = A[0].squeeze().detach().cpu()
    img_fake = fake_B[0].squeeze().detach().cpu()
    img_rec = rec_A[0].squeeze().detach().cpu()

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(img_in, cmap="turbo"); axs[0].set_title("Input A")
axs[1].imshow((img_fake + 1)/2*255, cmap="turbo"); axs[1].set_title("G_A(A)")
axs[2].imshow(img_rec, cmap="turbo"); axs[2].set_title("G_B(G_A(A))")
for ax in axs: ax.axis("off")
plt.tight_layout()
plt.show()
