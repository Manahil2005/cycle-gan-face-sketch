import numpy as np, torch, torch.nn as nn, torchvision.transforms as T

class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(c, c, 3), nn.InstanceNorm2d(c), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(c, c, 3), nn.InstanceNorm2d(c))
    def forward(self, x): return x + self.b(x)

class Generator(nn.Module):
    """c7s1-64, d128, d256, R256 x n, u128, u64, c7s1-3"""
    def __init__(self, n_res=6, nf=64):
        super().__init__()
        L = [nn.ReflectionPad2d(3), nn.Conv2d(3, nf, 7), nn.InstanceNorm2d(nf), nn.ReLU(True)]
        c = nf
        for _ in range(2):
            L += [nn.Conv2d(c, c*2, 3, 2, 1), nn.InstanceNorm2d(c*2), nn.ReLU(True)]; c *= 2
        L += [ResBlock(c) for _ in range(n_res)]
        for _ in range(2):
            L += [nn.ConvTranspose2d(c, c//2, 3, 2, 1, output_padding=1), nn.InstanceNorm2d(c//2), nn.ReLU(True)]; c //= 2
        L += [nn.ReflectionPad2d(3), nn.Conv2d(c, 3, 7), nn.Tanh()]
        self.m = nn.Sequential(*L)
    def forward(self, x): return self.m(x)

class Discriminator(nn.Module):
    """70x70 PatchGAN: C64-C128-C256-C512 -> 1"""
    def __init__(self, nf=64):
        super().__init__()
        def blk(i, o, s, norm=True):
            l = [nn.Conv2d(i, o, 4, s, 1)]
            if norm: l.append(nn.InstanceNorm2d(o))
            l.append(nn.LeakyReLU(0.2, True)); return l
        self.m = nn.Sequential(*blk(3, nf, 2, False), *blk(nf, nf*2, 2), *blk(nf*2, nf*4, 2),
                               *blk(nf*4, nf*8, 1), nn.Conv2d(nf*8, 1, 4, 1, 1))
    def forward(self, x): return self.m(x)

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None: nn.init.zeros_(m.bias)

# ---------- inference helpers ----------
def load_generators(path, device):
    ck = torch.load(path, map_location=device)
    cfg = ck['config']
    G_AB, G_BA = Generator(cfg['n_res']).to(device), Generator(cfg['n_res']).to(device)
    G_AB.load_state_dict(ck['G_AB']); G_BA.load_state_dict(ck['G_BA'])
    return G_AB.eval(), G_BA.eval(), cfg['img']

@torch.no_grad()
def translate(G, img, size):
    tf = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize((0.5,)*3, (0.5,)*3)])
    dev = next(G.parameters()).device
    y = G(tf(img.convert('RGB')).unsqueeze(0).to(dev))[0].cpu() * 0.5 + 0.5
    return T.ToPILImage()(y.clamp(0, 1))

def is_sketch(img):
    """Heuristic: sketches are almost colourless with a mostly-white page."""
    a = np.asarray(img.convert('RGB').resize((128, 128))).astype(np.float32) / 255.
    mx, mn = a.max(2), a.min(2)
    sat = ((mx - mn) / (mx + 1e-6)).mean()
    white = (a.mean(2) > 0.85).mean()
    return bool(sat < 0.10 and white > 0.25)
