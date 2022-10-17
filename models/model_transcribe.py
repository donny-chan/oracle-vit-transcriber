from functools import partial
from typing import List

import torch
from torch import Tensor
import torch.nn as nn

from models.vit import VisionTransformer, Block


class VitTranscriber(nn.Module):
    """
    只做图像复原任务的单模态模型
    """

    def __init__(
        self,
        config: dict = None,
        tokenizer=None,
        init_deit: bool = True,
        distributed: bool = False,
    ):
        super().__init__()
        self.distributed = distributed

        self.channel_num = len(config["img_mode"])
        input_number_classes = (
            config["image_res"] * config["image_res"] * self.channel_num
        )
        self.visual_encoder = VisionTransformer(
            img_size=config["image_res"],
            patch_size=16,
            embed_dim=config['embed_dim'],
            depth=config["encoder_layer"],
            num_heads=config["num_att_heads"],
            in_chans=self.channel_num,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

        self.reconstruct_decoder = nn.ModuleList(
            [
                Block(
                    dim=config['embed_dim'],
                    num_heads=config['num_att_heads'],
                    mlp_ratio=4,
                    qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                )
                for _ in range(config["decoder_layer"])
            ]
        )
        self.reconstruct_norm = nn.LayerNorm(config['embed_dim'])
        self.reconstruct_head = nn.Linear(
            config['embed_dim'], input_number_classes // 64)
        self.reconstruct_loss = nn.MSELoss()
        # self.reconstruct_loss = nn.L1Loss()

        self.config = config
        self.tokenizer = tokenizer
        # self.rec_idx = 0

        self.image_classification_factor = config[
            "image_classification_factor"
        ]
        if sum(self.image_classification_factor) > 0:
            self.classification_head = nn.Linear(
                768, self.tokenizer.vocab_size
            )
            self.classification_loss = nn.CrossEntropyLoss()

        print("Initializing weights in Transcriber")
        self.apply(self._init_weights)
        # nn.init.eye_(self.reconstruct_head.weight)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # This will make all Att and MLP modules return the input?
            # nn.init.constant_(m.weight, 0)
            nn.init.eye_(m.weight)
            # nn.init.uniform_(m.weight, a=-0.00001, b=0.00001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, images: Tensor):
        """
        images: (B, C, H, W)
        """
        # enc_embeds: (BL, # patch = 16, d)
        # all_hidden: list of (BL, # patch = 16, d)
        enc_embeds, all_layer_inputs = self.visual_encoder(images)
        # NOTE: hidden[-1] == enc_embeds
        cls_embed = enc_embeds[:, 0, :]  # (B L )
        # cls_embed = cls_embed.view(batch_size, 768)
        return cls_embed, enc_embeds, all_layer_inputs

    def forward_decoder(
        self,
        embeds: Tensor,
        all_layer_inputs: List[Tensor],
        targets: Tensor,
        images=None,
    ):
        """
        embeds: (B, P, d)
        targets: (B, C*H*W)
        all_hidden: list of (B, P, d), length is number of layers (ie. 2)
        """
        for i, blk in enumerate(self.reconstruct_decoder):
            embeds = blk(embeds)  # (B, ..., d)
            embeds += all_layer_inputs[-1 - i]
        gen_img = embeds[:, 1:, :]  # (B, 64, 768)

        # gen_img = self.reconstruct_head(self.reconstruct_norm(embeds))
        gen_img = self.reconstruct_head(gen_img)  # (B, 64, 768)

        # Reverse the patch embedding.
        b, c, h, w = targets.shape
        gen_img = gen_img.view(b, 8, 8, c, 16, 16)
        gen_img = gen_img.permute(
            0, 3, 1, 4, 2, 5
        ).contiguous()  # (B, c, 16, 8, 16, 8)
        gen_img = gen_img.view(b, c, h, w)

        # print('Mean of gen:', gen_img.mean())
        # print('Mean of target:', targets.mean())
        # img = F.to_pil_image(gen_img[0].cpu(), self.config['img_mode'])
        # img.save('gen.png')
        # img = F.to_pil_image(targets[0].cpu(), self.config['img_mode'])
        # img.save('target.png')
        # print('saved to gen.png and target.png')
        # exit()

        loss = self.reconstruct_loss(gen_img, targets)
        return gen_img, embeds, loss

    def forward_classification(self, embeds, texts):
        batch_sz, seq_len = texts.shape
        label_mask = texts != -100
        instance_num = torch.sum(label_mask).item()
        if sum(self.image_classification_factor) > 0:
            txt_embeds = self.classification_head(embeds)
            loss_cls = self.classification_loss(
                txt_embeds.view(batch_sz * seq_len, -1),
                texts.view(-1),
            )
            with torch.no_grad():
                predict_result_ids = torch.max(txt_embeds, dim=2)[1]
                correct_num = torch.sum(
                    torch.logical_and(
                        label_mask,
                        predict_result_ids == texts,
                    ),
                ).item()
        else:
            loss_cls, correct_num = torch.tensor(0.0).to(embeds), 0
        assert correct_num <= instance_num
        return loss_cls, correct_num, instance_num

    def forward(self, source, target, mode):
        """
        source: (B, C*H*W)
        target: (B, C*H*W)
        """
        assert mode in ["train", "valid", "test"]

        cls_embed, enc_embeds, all_layer_inputs = self.forward_encoder(source)
        gen_img, dec_embeds, gen_loss = self.forward_decoder(
            enc_embeds,
            # cls_embed.view(source.shape[0], 1, -1),
            all_layer_inputs,
            target,
            source,
        )
        total_loss = gen_loss
        return total_loss, gen_loss, gen_img
