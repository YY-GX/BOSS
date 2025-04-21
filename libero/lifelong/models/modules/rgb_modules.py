"""
This file contains all neural modules related to encoding the spatial
information of obs_t, i.e., the abstracted knowledge of the current visual
input conditioned on the language.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


###############################################################################
#
# Modules related to encoding visual information (can conditioned on language)
#
###############################################################################

# directly modify this one
class PatchEncoder(nn.Module):
    """
    A patch encoder that does a linear projection of patches in a RGB image.
    """

    def __init__(
        self, input_shape, patch_size=[16, 16], embed_size=64, no_patch_embed_bias=False
    ):
        super().__init__()
        C, H, W = input_shape
        num_patches = (H // patch_size[0] // 2) * (W // patch_size[1] // 2)
        self.img_size = (H, W)
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.h, self.w = H // patch_size[0] // 2, W // patch_size[1] // 2

        self.conv = nn.Sequential(
            nn.Conv2d(
                C, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            ),
            nn.BatchNorm2d(
                64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(
            64,
            embed_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False if no_patch_embed_bias else True,
        )
        self.bn = nn.BatchNorm2d(embed_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv(x)
        x = self.proj(x)
        x = self.bn(x)
        return x


class SpatialSoftmax(nn.Module):
    """
    The spatial softmax layer (https://rll.berkeley.edu/dsae/dsae.pdf)
    """

    def __init__(self, in_c, in_h, in_w, num_kp=None):
        super().__init__()
        self._spatial_conv = nn.Conv2d(in_c, num_kp, kernel_size=1)

        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1, 1, in_w).float(),
            torch.linspace(-1, 1, in_h).float(),
        )

        pos_x = pos_x.reshape(1, in_w * in_h)
        pos_y = pos_y.reshape(1, in_w * in_h)
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)

        if num_kp is None:
            self._num_kp = in_c
        else:
            self._num_kp = num_kp

        self._in_c = in_c
        self._in_w = in_w
        self._in_h = in_h

    def forward(self, x):
        assert x.shape[1] == self._in_c
        assert x.shape[2] == self._in_h
        assert x.shape[3] == self._in_w

        h = x
        if self._num_kp != self._in_c:
            h = self._spatial_conv(h)
        h = h.contiguous().view(-1, self._in_h * self._in_w)

        attention = F.softmax(h, dim=-1)
        keypoint_x = (
            (self.pos_x * attention).sum(1, keepdims=True).view(-1, self._num_kp)
        )
        keypoint_y = (
            (self.pos_y * attention).sum(1, keepdims=True).view(-1, self._num_kp)
        )
        keypoints = torch.cat([keypoint_x, keypoint_y], dim=1)
        return keypoints


class SpatialProjection(nn.Module):
    def __init__(self, input_shape, out_dim):
        super().__init__()

        assert (
            len(input_shape) == 3
        ), "[error] spatial projection: input shape is not a 3-tuple"
        in_c, in_h, in_w = input_shape
        num_kp = out_dim // 2
        self.out_dim = out_dim
        self.spatial_softmax = SpatialSoftmax(in_c, in_h, in_w, num_kp=num_kp)
        self.projection = nn.Linear(num_kp * 2, out_dim)

    def forward(self, x):
        out = self.spatial_softmax(x)
        out = self.projection(out)
        return out

    def output_shape(self, input_shape):
        return input_shape[:-3] + (self.out_dim,)


class ResnetEncoder(nn.Module):
    """
    A Resnet-18-based encoder for mapping an image to a latent vector

    Encode (f) an image into a latent vector.

    y = f(x), where
        x: (B, C, H, W)
        y: (B, H_out)

    Args:
        input_shape:      (C, H, W), the shape of the image
        output_size:      H_out, the latent vector size
        pretrained:       whether use pretrained resnet
        freeze: whether   freeze the pretrained resnet
        remove_layer_num: remove the top # layers
        no_stride:        do not use striding
    """

    def __init__(
        self,
        input_shape,
        output_size,
        pretrained=False,
        freeze=False,
        remove_layer_num=2,
        no_stride=False,
        language_dim=768,
        language_fusion="film",
    ):

        super().__init__()

        ### 1. encode input (images) using convolutional layers
        assert remove_layer_num <= 5, "[error] please only remove <=5 layers"
        layers = list(torchvision.models.resnet18(pretrained=pretrained).children())[
            :-remove_layer_num
        ]
        self.remove_layer_num = remove_layer_num

        assert (
            len(input_shape) == 3
        ), "[error] input shape of resnet should be (C, H, W)"

        in_channels = input_shape[0]
        if in_channels != 3:  # has eye_in_hand, increase channel size
            conv0 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )
            layers[0] = conv0

        self.no_stride = no_stride
        if self.no_stride:
            layers[0].stride = (1, 1)
            layers[3].stride = 1

        self.resnet18_base = nn.Sequential(*layers[:4])
        self.block_1 = layers[4][0]
        self.block_2 = layers[4][1]
        self.block_3 = layers[5][0]
        self.block_4 = layers[5][1]

        self.language_fusion = language_fusion
        if language_fusion != "none":
            self.lang_proj1 = nn.Linear(language_dim, 64 * 2)
            self.lang_proj2 = nn.Linear(language_dim, 64 * 2)
            self.lang_proj3 = nn.Linear(language_dim, 128 * 2)
            self.lang_proj4 = nn.Linear(language_dim, 128 * 2)

        if freeze:
            if in_channels != 3:
                raise Exception(
                    "[error] cannot freeze pretrained "
                    + "resnet with the extra eye_in_hand input"
                )
            for param in self.resnet18_embeddings.parameters():
                param.requires_grad = False

        ### 2. project the encoded input to a latent space
        x = torch.zeros(1, *input_shape)
        y = self.block_4(
            self.block_3(self.block_2(self.block_1(self.resnet18_base(x))))
        )
        output_shape = y.shape  # compute the out dim
        self.projection_layer = SpatialProjection(output_shape[1:], output_size)
        self.output_shape = self.projection_layer(y).shape

    def forward(self, x, langs=None):
        h = self.resnet18_base(x)

        h = self.block_1(h)
        if langs is not None and self.language_fusion != "none":  # FiLM layer
            B, C, H, W = h.shape
            beta, gamma = torch.split(
                self.lang_proj1(langs).reshape(B, C * 2, 1, 1), [C, C], 1
            )
            h = (1 + gamma) * h + beta

        h = self.block_2(h)
        if langs is not None and self.language_fusion != "none":  # FiLM layer
            B, C, H, W = h.shape
            beta, gamma = torch.split(
                self.lang_proj2(langs).reshape(B, C * 2, 1, 1), [C, C], 1
            )
            h = (1 + gamma) * h + beta

        h = self.block_3(h)
        if langs is not None and self.language_fusion != "none":  # FiLM layer
            B, C, H, W = h.shape
            beta, gamma = torch.split(
                self.lang_proj3(langs).reshape(B, C * 2, 1, 1), [C, C], 1
            )
            h = (1 + gamma) * h + beta

        h = self.block_4(h)
        if langs is not None and self.language_fusion != "none":  # FiLM layer
            B, C, H, W = h.shape
            beta, gamma = torch.split(
                self.lang_proj4(langs).reshape(B, C * 2, 1, 1), [C, C], 1
            )
            h = (1 + gamma) * h + beta

        h = self.projection_layer(h)
        return h

    def output_shape(self, input_shape, shape_meta):
        return self.output_shape






###############################################################################
#
# R3M
# MVP
# LIV
#
###############################################################################





import torch
import torch.nn as nn
from r3m import load_r3m


class R3MEncoder(nn.Module):
    """
    A wrapper around the R3M visual encoder that uses a pretrained ResNet-based
    vision model trained on Ego4D with language supervision.

    This module freezes the R3M weights and provides a fixed visual representation.

    Args:
        input_shape: (C, H, W) tuple representing the input image size.
        output_size: desired feature output dimensionality (will apply projection).
        model_name: backbone to use for R3M ('resnet18', 'resnet34', 'resnet50').
    """

    def __init__(self, input_shape, output_size, model_name='resnet50', device="cuda:0"):
        super().__init__()
        self.r3m = load_r3m(model_name)
        self.r3m.eval()  # set to eval mode
        # self.r3m.to(device)

        # Freeze R3M
        for param in self.r3m.parameters():
            param.requires_grad = False

        # # Dummy forward pass to determine R3M feature dim
        # with torch.no_grad():
        #     dummy_input = torch.zeros(1, *input_shape)
        #     r3m_feat = self.r3m(dummy_input)
        # self.r3m_output_dim = r3m_feat.shape[-1]
        self.r3m_output_dim = 2048


        # Optional projection to match expected output_size
        self.projection = nn.Linear(self.r3m_output_dim, output_size)
        self.output_size = output_size

    def forward(self, x):  # langs ignored; R3M is image-only
        with torch.no_grad():
            r3m_feat = self.r3m(x)  # (B, D)
        # print(f"R3M feature shape: {r3m_feat.shape}")
        return self.projection(r3m_feat)

    def output_shape(self, input_shape, shape_meta=None):
        return (self.output_size,)



import torch
import torch.nn as nn
import mvp  # make sure `pip install git+https://github.com/ir413/mvp` is run

class MVPEncoder(nn.Module):
    """
    A wrapper around the MVP visual encoder that uses a pretrained ViT-based
    model trained with Masked Autoencoding for robotic visual tasks.

    This module freezes the MVP weights and provides a fixed visual representation.

    Args:
        input_shape: (C, H, W) tuple representing the input image size.
        output_size: desired feature output dimensionality (will apply projection).
        model_name: pre-trained model identifier ('vits-mae-egosoup', 'vitb-mae-egosoup', etc.).
    """

    def __init__(self, input_shape, output_size, model_name="vitb-mae-egosoup", device="cuda:0"):
        super().__init__()
        self.mvp = mvp.load(model_name).to(device)
        self.mvp.eval()
        model.freeze()

        # # Freeze MVP
        # for param in self.mvp.parameters():
        #     param.requires_grad = False

        # MVP ViT-B outputs 768-dimensional embeddings (for vitb); adjust if needed
        if "vits" in model_name:
            self.mvp_output_dim = 384
        elif "vitb" in model_name:
            self.mvp_output_dim = 768
        elif "vitl" in model_name:
            self.mvp_output_dim = 1024
        else:
            raise ValueError(f"Unknown model type in model_name: {model_name}")

        # Optional projection to match expected output_size
        self.projection = nn.Linear(self.mvp_output_dim, output_size)
        self.output_size = output_size

    def forward(self, x):  # x should be normalized (B, 3, H, W)
        with torch.no_grad():
            mvp_feat = self.mvp(x)  # (B, D)
        return self.projection(mvp_feat)

    def output_shape(self, input_shape, shape_meta=None):
        return (self.output_size,)


import torch
import torch.nn as nn
from liv import load_liv
import clip


class LIVEncoder(nn.Module):
    """
    A wrapper around the LIV multi-modal encoder that computes embeddings for both
    vision and language inputs. It uses a frozen CLIP-based encoder trained with
    language and vision alignment for robotic tasks.

    Args:
        input_shape: (C, H, W) tuple representing the input image size.
        output_size: dimensionality of the projected embedding.
        device: device for running the model.
    """

    def __init__(self, input_shape, output_size, device="cuda:0"):
        super().__init__()
        self.device = device
        self.liv = load_liv().to(device)
        self.liv.eval()

        for param in self.liv.parameters():
            param.requires_grad = False

        # LIV uses a 512-d output embedding (CLIP ViT-B/32 typically)
        self.liv_output_dim = 512
        self.output_size = output_size

        # Optional projection
        self.projection = nn.Linear(self.liv_output_dim, output_size)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Image tensor (B, 3, H, W)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: projected image features, projected text features
        """

        with torch.no_grad():
            image_embedding = self.liv(input=x.to(self.device), modality="vision")   # (B, D)

        return self.projection(image_embedding)

    def output_shape(self, input_shape, shape_meta=None):
        return (self.output_size,)
