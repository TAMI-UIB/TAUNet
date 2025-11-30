import torch
from torch.nn.functional import fold, unfold


class CreatePatches:
    def __init__(self, data, patch_size, overlapping=None):
        """
        data: tensor [N, C, H, W]
        patch_size: tamaño del patch (int)
        overlapping: solapamiento (int o None)
        """
        self.patch_size = patch_size
        self.overlapping = overlapping
        self.N, self.C, self.H, self.W = data.size()

    def do_patches(self, data):
        """
        Divide la imagen en patches con o sin solapamiento.
        Devuelve tensor [num_patches_totales, C, patch_size, patch_size]
        """
        if self.overlapping is None or self.overlapping == 0:
            if self.H % self.patch_size != 0 or self.W % self.patch_size != 0:
                raise ValueError('La imagen debe ser divisible por el tamaño de patch.')

            patches = unfold(data, kernel_size=self.patch_size, stride=self.patch_size)
            patches = patches.permute(0, 2, 1)
            patches = patches.view(-1, self.C, self.patch_size, self.patch_size)
            return patches

        else:
            stride = self.patch_size - self.overlapping
            patches = []
            for b in range(data.size(0)):
                for i in range(0, self.H - self.patch_size + 1, stride):
                    for j in range(0, self.W - self.patch_size + 1, stride):
                        patch = data[b:b+1, :, i:i+self.patch_size, j:j+self.patch_size]
                        patches.append(patch)
            patches = torch.cat(patches, dim=0)
            return patches

    def undo_patches(self, patches):
        """
        Reconstruye la imagen a partir de los patches.
        Promedia correctamente las zonas solapadas.
        """
        C_out = patches.size(1)
        stride = self.patch_size - (self.overlapping or 0)

        # Inicializa tensores de acumulación
        reconstructed = torch.zeros((self.N, C_out, self.H, self.W), device=patches.device)
        weight = torch.zeros_like(reconstructed)

        patch_idx = 0
        for b in range(self.N):
            for i in range(0, self.H - self.patch_size + 1, stride):
                for j in range(0, self.W - self.patch_size + 1, stride):
                    reconstructed[b, :, i:i+self.patch_size, j:j+self.patch_size] += patches[patch_idx]
                    weight[b, :, i:i+self.patch_size, j:j+self.patch_size] += 1
                    patch_idx += 1

        reconstructed /= torch.clamp(weight, min=1e-8)
        return reconstructed
