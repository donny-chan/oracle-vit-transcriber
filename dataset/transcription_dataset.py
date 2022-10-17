from pathlib import Path
import os
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from dataset.data_utils import resize_pad_image


def process_image_reconstruct(book: dict, mids: list, config):
    img_res, img_mode, pad_color = (
        config["image_res"],
        config["img_mode"],
        config["pad_color"],
    )
    res_images, res_captions = [], []
    for cid, (char, image) in enumerate(book["characters"]):
        if cid not in mids:
            continue
        assert os.path.exists(os.path.join(config["data_prefix"], image))
        cur_img = Image.open(
            os.path.join(config["data_prefix"], image)
        ).convert(img_mode)
        pad_img = resize_pad_image(
            cur_img, (img_res, img_res), do_trans=False, pad_color=pad_color
        )
        pad_mask_img = resize_pad_image(
            cur_img,
            (img_res, img_res),
            do_trans=config["img_random_transform"],
            pad_color=pad_color,
            mask_ratio=config["img_mask_ratio"],
            noise_ratio=config["img_noise_ratio"],
            do_rotate=config["img_do_rotate"],
        )
        res_images.append(
            (
                Image.fromarray(pad_img, mode=img_mode),
                Image.fromarray(pad_mask_img, mode=img_mode),
            )
        )
        res_captions.append(char)
    assert len(res_images) > 0
    return res_images, res_captions


def is_valid_image(book, cid):
    book_name, row_order = book["book_name"], book["row_order"]
    return os.path.basename(book["characters"][cid][1]).startswith(
        f"{book_name}-{row_order}"
    )


class TranscriptionDataset(Dataset):
    def __init__(self, config, mode):
        self.mode = config["dataset_mode"]
        self.config = config

        img_dir = Path(config["data_prefix"], mode)
        img_files = sorted(img_dir.glob("*.png"))
        self.source_files = [f for f in img_files if f.name.endswith("_r.png")]
        self.target_files = [f for f in img_files if f.name.endswith("_t.png")]

        self.model_mode = mode
        if config["img_mode"] == "RGB":
            # This normalization makes the white strokes black
            # with ragged white pixels around it, so just remove it.
            mean, std = {
                128: ([0.5601, 0.5598, 0.5596], [0.4064, 0.4065, 0.4066]),
                64: ([0.5601, 0.5598, 0.5596], [0.4064, 0.4065, 0.4066]),
            }[config["image_res"]]
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=mean, std=std),
                ]
            )
        else:
            mean, std = {
                128: ([0.5599], [0.4065]),
                64: ([0.5599], [0.4065]),
            }[config["image_res"]]
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=mean, std=std),
                ]
            )

    def __len__(self):
        # return len(self.data)
        return len(self.source_files)

    def random_crop_characters(self, book):
        limit, chars = self.config["max_length"], book["characters"]
        if limit < 0:
            return book
        if len(chars) > limit:
            begin = random.randint(0, len(chars) - limit)
            book["characters"] = chars[begin: (begin + limit)]
        return book

    def process_img(
        self, file: Path, data_augmentation: bool = False
    ) -> Image.Image:
        """
        Load image and resize, pad, and do data augmentation (random
        transform, mask, noise, etc), then transform and flatten.
        """
        img_res = self.config["image_res"]
        img_mode = self.config["img_mode"]
        pad_color = self.config["pad_color"]
        img = Image.open(file).convert(img_mode)
        img.save("test_orig.png")
        if data_augmentation:
            img = resize_pad_image(
                img,
                (img_res, img_res),
                do_trans=self.config["img_random_transform"],
                pad_color=pad_color,
                mask_ratio=self.config["img_mask_ratio"],
                noise_ratio=self.config["img_noise_ratio"],
                do_rotate=self.config["img_do_rotate"],
            )
        else:
            img = resize_pad_image(
                img,
                (img_res, img_res),
                do_trans=False,
                pad_color=pad_color,
            )
        img = Image.fromarray(img, mode=img_mode)
        img = self.transform(img)  # Transform
        return img

    def __getitem__(self, index):
        # book, mids = self.data[index]
        source_file = self.source_files[index]
        target_file = self.target_files[index]
        source_img = self.process_img(source_file, data_augmentation=True)
        target_img = self.process_img(target_file, data_augmentation=False)
        # exit()

        # x = target_img.cpu().detach().numpy() # (3, 128, 128)
        # x = x.transpose(1, 2, 0)
        # x.save('x.png')
        # img = F.to_pil_image(target_img, 'RGB')
        # img = Image.fromarray(target_img, mode='RGB')
        # img.save('x.png')
        # exit()
        return source_img, target_img
