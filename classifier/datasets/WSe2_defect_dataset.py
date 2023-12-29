import os
import glob
import math
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from pathlib import Path
from PIL import Image
from scipy import stats
from scipy.signal import convolve2d

from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import transforms as T

from atomai.utils import extract_patches, create_multiclass_lattice_mask

from nanoscopy import spm
from nanoscopy.utilities import progbar
from ..imgen import create_lattice_mask, create_multiclass_lattice_mask


def norm(data):
    # Normalize to (0, 1)
    data = (data - data.min()) / (data.max() - data.min())
    return data


def get_transform():
    transforms = [
        T.ToTensor(),
        #        T.Resize(size=(512, 512))
    ]

    return T.Compose(transforms)


class WSe2DefectDataset_VIA(torch.utils.data.Dataset):
    def __init__(self, root, d_dims_path=None, transforms=None):
        self.root = root
        self.transforms = get_transform() if transforms is None else transforms
        self.target_size = 512
        self.defect_types = {
            "SBP": {
                "name": "Small Bright Peak",
                "id": 1,
                "color": "#002060",
                "count": 0,
            },
            "SDP": {"name": "Small Dim Peak", "id": 2, "color": "#00B0F0", "count": 0},
            "SBT": {
                "name": "Small Bright Trough",
                "id": 3,
                "color": "#7030A0",
                "count": 0,
            },
            "SDT": {
                "name": "Small Dim Trough",
                "id": 4,
                "color": "#0070C0",
                "count": 0,
            },
            "LBP": {
                "name": "Large Bright Peak",
                "id": 5,
                "color": "#00B050",
                "count": 0,
            },
            "LDP": {"name": "Large Dim Peak", "id": 6, "color": "#FFC000", "count": 0},
            "LBT": {
                "name": "Large Bright Trough",
                "id": 7,
                "color": "#92D050",
                "count": 0,
            },
            "LDT": {
                "name": "Large Dim Trough",
                "id": 8,
                "color": "#C00000",
                "count": 0,
            },
        }

        self.default_dim_path = os.path.join(root, "Mean Defect Diminsions.xlsx")
        self.default_dim_path = d_dims_path or self.default_dim_path
        self.ann_path = os.path.join(root, "annotations.json")

        self.d_dims = pd.read_excel(self.default_dim_path)
        self.anns = pd.read_json(self.ann_path)
        self.clean_anns()
        self.all_anns = self.anns
        print(f"{len(self.anns)} annotations read.")

        # codes, uniques = pd.factorize(anns['label'])
        # self.unique_labels = uniques
        # anns['label_id'] = codes + 1
        # self.num_classes = len(anns['label'].unique())
        # UCF_mask = anns_VIA['Data_Source'] == 'UCF'
        # columbia_mask = anns_VIA['Data_Source'] == 'COL'
        # crystal_batch_mask = anns_VIA['Crystal_Batch'] == 'SL445'

        # self.anns = anns_VIA[crystal_batch_mask]

    def __len__(self):
        return len(self.anns)

    def update_filenames(self):
        self.img_filenames = np.array(sorted(self.anns["filename"].unique()))

    def clean_anns(self):
        # clean annotations
        self.anns = self.anns.T
        self.anns = self.anns.reset_index(drop=True)
        file_attributes_df = pd.json_normalize(self.anns["file_attributes"])
        self.anns = self.anns.drop(["file_attributes"], axis=1)
        self.anns = self.anns.join(file_attributes_df)

        # drop all instances with empty regions (i.e. no annotations)
        no_regions_mask = self.anns["regions"].str.len() != 0
        self.anns = self.anns[no_regions_mask]
        # select datetime substring from filename
        # self.anns["datetime"] = self.anns["filename"].str.split("_", expand=True)[1]
        self.anns.loc[self.anns["Data_Source"] == "UCF", "datetime"] = self.anns[
            "filename"
        ].str.split("_", expand=True)[1]
        self.anns.loc[self.anns["Data_Source"] == "COL", "datetime"] = self.anns[
            "filename"
        ].str.split("_", expand=True)[4]

        self.anns["Scan_Width(nm)"] = self.anns["Scan_Width(nm)"].astype(int)

        self.update_filenames()

    def extract_defect_images(self, dataloader):
        cropped_defects = []
        labels = []
        plasma_cmap = plt.cm.get_cmap("plasma")

        for batch_idx, batch_sample in enumerate(dataloader):
            images, targets = batch_sample
            image = images[0].permute(1, 2, 0).numpy()
            image = image[:, :, 0]

            target = targets[0]
            bboxes = target["boxes"].numpy()
            label_ids = target["labels"].numpy()
            image_id = target["image_id"].numpy()[0]
            filename = self.img_filenames[image_id]
            defect_mask = target["masks"].numpy()
            area = targets[0]["area"].numpy()

            for i, bbox in enumerate(bboxes):
                label_id = label_ids[i]
                # color_i = plasma_cmap(norm(label_id))
                label_name = ""

                for key in self.defect_types:
                    defect_type = self.defect_types[key]
                    type_id = defect_type["id"]

                    if type_id == label_id:
                        label_name = defect_type["name"]

                xmin, ymin, xmax, ymax = bbox
                w, h = xmax - xmin, ymax - ymin
                xc, yc = self.get_bbox_center(bbox)

                cropped = image[int(ymin) : int(ymax), int(xmin) : int(xmax)]
                cropped_defects.append(cropped)

                label_acro = "".join(list(filter(lambda c: c.isupper(), label_name)))
                labels.append(label_acro)

        return cropped_defects, labels

    def reset_anns(self):
        self.anns = self.all_anns
        # self.clean_anns()

    def set_crystal_batch(self, batch):
        mask = self.anns["Crystal_Batch"] == batch
        self.anns = self.anns[mask]
        self.update_filenames()
        print(f"{len(self.anns)} annotations of {batch} selected.")

    def set_data_source(self, source):
        mask = self.anns["Data_Source"] == source
        self.anns = self.anns[mask]
        self.update_filenames()
        print(f"{len(self.anns)} annotations of {source} selected.")

    def set_scan_size(
        self,
        size: int = 50,
        mode: str = "eq",
    ) -> None:
        if mode == "eq":
            mask = self.anns["Scan_Width(nm)"].astype(int) == size
            mode_sym = "=="
        elif mode == "ge":
            mask = self.anns["Scan_Width(nm)"].astype(int) >= size
            # print(self.anns["Scan_Width(nm)"].astype(int), size)
            mode_sym = ">="
        else:
            mask = self.anns["Scan_Width(nm)"].astype(int) <= size
            mode_sym = "<="

        self.anns = self.anns[mask]
        self.update_filenames()
        print(f"{len(self.anns)} annotations {mode_sym} {size} nm selected.")

    def get_idx(self, filename):
        ann_mask = self.anns["filename"] == filename

        if not ann_mask.any():
            # print('filename not found', self.img_filenames[idx])
            raise ValueError("filename not found in annotations:", filename)

        ann = self.anns[ann_mask]
        return ann.index[0]

    def get_ann(self, idx):
        ann = self.anns.iloc[idx]
        return ann

    def get_ann_from_filename(self, filename):
        idx = self.get_idx(filename)
        ann = self.get_ann(idx)
        return ann

    def get_image(self, ann):
        img_path = os.path.join(self.root, "STM_images", ann["filename"])
        img = Image.open(img_path).convert("RGB")
        _, x = img.size
        return img

    def get_crystal_batches(self):
        return self.anns["Crystal_Batch"].value_counts().index.tolist()

    def get_rec_index(self, ann):
        if ann["Data_Source"] == "UCF":
            rec_idx = ann["filename"].split("_")[-1].split(".")[0]
        else:
            rec_idx = (
                ann["filename"].split("_")[-3] + "-" + ann["filename"].split("_")[-2]
            )

        return rec_idx

    def get_size_nm(self, ann):
        w, h = int(ann["Scan_Width(nm)"]), int(ann["Scan_Height(nm)"])
        return w, h

    def get_sxm_path(self, idx):
        ann = self.get_ann(idx)

        try:
            datetime_obj = datetime.strptime(ann["datetime"], "%y%m%d")
            datetime_str = datetime_obj.strftime("%Y-%m-%d")
            rec_index = self.get_rec_index(ann)
            filename = f"STM_WTip_WSe2-SL445_{rec_index}.sxm"
            sxm_path = os.path.join(self.root, "STM_data", datetime_str, filename)
            return sxm_path
        except Exception as e:
            print(e)
            return ""

    def get_Z(self, sxm_path):
        sxm = spm.read(sxm_path)[0]
        Z_fwd = sxm.fwd()
        Z = spm.correct([Z_fwd], poly=True, equalize=False, rescale=False)
        return Z

    def get_Zs(self, sxm_paths):
        scans = spm.read(sxm_paths)
        Zs_fwd = [spm_data.fwd() for spm_data in scans]
        Zs = spm.correct(Zs_fwd, poly=True, equalize=False, rescale=False)
        return Zs

    def estimate_noise(self, I):
        """
        512 x 512 Image of random gaussian noise produces 86.
        """
        H, W = I.shape

        M = [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]

        sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))
        sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W - 2) * (H - 2))

        return sigma

    def estimate_all_noise(self, images):
        noises = []

        for image in images:
            noise = self.estimate_noise(image)
            noises.append(noise)

        noises = np.array(noises)
        return noises

    def get_T_crtical(self, degrees_free, q=1 - 0.975):
        t_crit = abs(stats.t.ppf(q, degrees_free))
        t_crit[np.isnan(t_crit)] = 0
        return t_crit

    def get_defect_counts(self, ann):
        defect_types = {
            "SBP": 0,
            "SDP": 0,
            "SBT": 0,
            "SDT": 0,
            "LBP": 0,
            "LDP": 0,
            "LBT": 0,
            "LDT": 0,
        }

        for region in ann["regions"]:
            shape_attr = region["shape_attributes"]
            region_attr = region["region_attributes"]
            defect_type = str(region_attr["Defect_Type"])
            defect_types[defect_type] += 1

        return defect_types

    def get_total_area(self):
        areas = []

        for i, ann in self.anns.iterrows():
            w, h = self.get_size_nm(ann)
            areas.append(w * h)

        areas = np.array(areas)
        return np.sum(areas)

    def get_all_defect_counts(self):
        areas = []
        widths = []
        defect_counts = []

        for i, ann in self.anns.iterrows():
            defect_count = self.get_defect_counts(ann)
            w, h = self.get_size_nm(ann)
            areas.append(w * h)
            widths.append(w)
            defect_counts.append(defect_count)

        type_counts_df = pd.DataFrame(defect_counts)
        areas_df = pd.DataFrame(np.array([widths, areas]).T, columns=["width", "area"])
        counts_areas_df = pd.concat([type_counts_df, areas_df], axis=1)
        return type_counts_df, areas_df

    def get_all_defect_densities(self):
        counts_df, areas_df = self.get_all_defect_counts()
        # summed_defect_type_counts_areas = counts_areas_df.sum()
        total_area_nm = areas_df.sum()["area"]
        total_area_cm = total_area_nm * 1e-14
        # summed_defect_type_counts = counts_areas_df.loc['SBP':'LDT']
        densities = counts_df.sum() / total_area_cm
        densities = densities.sort_values(ascending=False)
        print(
            f"total count = {counts_df.sum().sum()}\ntotal area = {total_area_cm}\n=> densitiy = {densities.sum():0.2e}"
        )
        # errors =
        return densities

    def get_all_density_errors(self):
        counts_areas_df = self.get_all_defect_counts()
        summed_defect_type_counts = counts_areas_df.sum()

    def get_defect_counts_from_filename(self, filename):
        ann = self.get_ann_from_filename(filename)
        defect_counts = self.get_defect_counts(ann)
        return defect_counts

    def get_defect_densities_from_filename(self, filename):
        ann = self.get_ann_from_filename(filename)
        defect_counts = self.get_defect_counts(ann)
        counts_df = pd.DataFrame(
            list(defect_counts.values()),
            columns=["count"],
            index=list(defect_counts.keys()),
        )
        w, h = self.get_size_nm(ann)
        area_nm = w * h
        area_cm = area_nm * 1e-14
        densities = counts_df / area_cm
        densities = densities.rename({"count": "density"}, axis="columns")
        densities = densities.sort_values(by="density", ascending=False)
        return densities

    def plot_defect_counts(self):
        counts_areas_df = self.get_all_defect_counts()
        summed_defect_type_counts = counts_areas_df.sum()
        counts_df = summed_defect_type_counts.loc["SBP":"LDT"]
        sorted_counts = counts_df.sort_values(ascending=False)
        labels = [
            self.defect_types[label]["name"] for label in sorted_counts.index.values
        ]
        color = [
            self.defect_types[label]["color"] for label in sorted_counts.index.values
        ]
        batches = self.get_crystal_batches()
        plt.bar(labels, sorted_counts, color=color)
        plt.ylabel("Count")
        plt.xticks(rotation=-70)
        plt.title(
            f'WSe2 Point Defect Class Distribution ({np.sum(sorted_counts)} Total)\n Crystal Batches: {", ".join(np.unique(batches))}'
        )
        plt.show()

    def plot_defect_densities(self):
        densities = self.get_all_defect_densities()
        final_counts = densities.loc["SBP":"LDT"].sort_values(ascending=False)
        labels = [
            self.defect_types[label]["name"] for label in final_counts.index.values
        ]
        color = [
            self.defect_types[label]["color"] for label in final_counts.index.values
        ]
        batches = self.get_crystal_batches()
        plt.bar(labels, final_counts, color=color)
        plt.ylabel("Count")
        plt.xticks(rotation=-70)
        plt.title(
            f'WSe2 Point Defect Class Distribution ({np.sum(final_counts)} Total)\n Crystal Batches: {", ".join(np.unique(batches))}'
        )
        plt.show()

    def plot_defect_counts_from_filename(self, filename):
        counts = self.get_defect_counts_from_filename(filename)
        counts_df = pd.DataFrame(
            list(counts.values()), columns=["count"], index=list(counts.keys())
        )
        sorted_counts = counts_df.sort_values(by="count", ascending=False)
        labels = [
            self.defect_types[label]["name"] for label in sorted_counts.index.values
        ]
        color = [
            self.defect_types[label]["color"] for label in sorted_counts.index.values
        ]

        plt.bar(labels, sorted_counts.values.reshape(len(sorted_counts)), color=color)
        plt.ylabel("Count")
        plt.xticks(rotation=-70)
        plt.title(
            f"WSe2 Point Defect Class Distribution ({np.sum(sorted_counts.values)} Total)\n {filename}"
        )
        plt.show()

        return sorted_counts

    def gen_meshgrid(self, width_nm, height_nm):
        x_nm = np.linspace(0, width_nm, self.target_size)
        y_nm = np.linspace(0, height_nm, self.target_size)
        xx, yy = np.meshgrid(x_nm, y_nm)
        img[:, :, 1] = xx.astype(np.float)
        img[:, :, 2] = yy.astype(np.float)
        img = Image.fromarray(img.astype(np.uint8))
        return img

    def extract_scan_dims_from_filename(self, filename):
        parts = filename.split("_")
        w, h = 0, 0

        for part in parts:
            part = part.lower()
            if part.find("x") != -1:
                w, h = part.split("x")

        return float(w), float(h)

    def extract_params_from_filename(self, filename):
        name, date, bias, area, channel, idx = filename.split("_")
        material, sample_id = name.split("-")
        w, h = area.split("x")
        params = {
            "material": material,
            "sample_id": sample_id,
            "date": date,
            "bias (V)": bias,
            "width (nm)": w,
            "height (nm)": h,
            "channel": channel,
            "scan_index": idx,
        }
        return params

    def get_bbox(self, ann, truncate=False):
        xmin, ymin = int(ann["x"]), int(ann["y"])
        w, h = int(ann["width"]), int(ann["height"])
        xmax, ymax = xmin + w, ymin + h
        area = (xmax - xmin) * (ymax - ymin)

        if area == 0:
            raise ValueError("Bounding box area must be non-zero.")

        bbox = (xmin, ymin, xmax, ymax)

        if truncate:
            bbox = self.truncate_bbox(bbox)

        return bbox

    # crop any bounding box which extends past the image boundaries
    def truncate_bbox(self, bbox):
        xmin, ymin, xmax, ymax = bbox

        if xmin < 0:
            xmin = 0
        if ymin < 0:
            ymin = 0
        if xmax > self.target_size:
            xmax = self.target_size
        if ymax > self.target_size:
            ymax = self.target_size

        return xmin, ymin, xmax, ymax

    def get_bbox_center(self, bbox):
        xmin, ymin, xmax, ymax = bbox
        xc = (xmin + xmax) // 2
        yc = (ymin + ymax) // 2

        if xc > self.target_size or yc > self.target_size:
            print("bbox center=(", xc, yc, ") past image boundry!")

        return xc, yc

    def resize_bbox(self, bbox, x, y):
        # set xy rescale values
        x_scale = self.target_size / x
        y_scale = self.target_size / y

        # resize boxes
        xmin, ymin, xmax, ymax = bbox
        xmin = float(np.round(xmin * x_scale))
        ymin = float(np.round(ymin * y_scale))
        xmax = float(np.round(xmax * x_scale))
        ymax = float(np.round(ymax * y_scale))

        return xmin, ymin, xmax, ymax

    def get_defect_sizes(self, idx, img):
        ann = self.anns.iloc[idx]
        scan_w, scan_h = int(ann["Scan_Width(nm)"]), int(ann["Scan_Height(nm)"])
        size_nm = scan_w

        if size_nm == 0:
            raise ValueError("Scan size must be non-zero. Scan size found: ", size_nm)

        _, image_size_pixels = img.shape
        image_size_A = size_nm * 10
        pixels_per_A = image_size_pixels / image_size_A
        defect_sizes = {}

        for label_name in self.d_dims["Defect Code"]:
            if label_name not in defect_sizes:
                size_mask = self.d_dims["Defect Code"] == label_name
                size_A = self.d_dims[size_mask]["mean length (A)"].iat[0]
                size_pixels = int(size_A * pixels_per_A)
                defect_sizes[label_name] = size_pixels

        return defect_sizes

    def create_defect_masks(self, idx, img, d_coords):
        # convert to numpy grayscale image
        img_gray = np.array(img.convert("L"))

        defect_mask = np.zeros_like(img_gray, dtype=float)
        # get radius for each defect class in pixels for this image
        defect_sizes = self.get_defect_sizes(idx, img_gray)
        d_masks = []

        for defect_label in d_coords:
            xy = np.array(d_coords[defect_label])
            d_size_pizels = defect_sizes[defect_label]
            # print(defect_label, ':radius', d_size_pizels, 'Number of defects:', xy.shape, 'max-min:', xy.max(), xy.min())
            _, d_mask = create_lattice_mask(
                img_gray, xy, scale=d_size_pizels, rmask=d_size_pizels
            )
            d_mask = d_mask.T
            d_masks.append(d_mask)

        defect_mask = np.sum(d_masks, axis=0)
        defect_mask[defect_mask > 1] = 1

        return defect_mask

    def create_defect_masks_multiclass(self, idx, img, d_coords):
        # convert to numpy grayscale image
        img_gray = np.array(img.convert("L"))

        defect_mask = np.zeros_like(img_gray, dtype=float)
        # get radius for each defect class in pixels for this image
        defect_sizes = self.get_defect_sizes(idx, img_gray)
        d_masks = []

        for defect_label in d_coords:
            xy = np.array(d_coords[defect_label])
            d_size_pizels = defect_sizes[defect_label]
            # print(defect_label, ':radius', d_size_pizels, 'Number of defects:', xy.shape, 'max-min:', xy.max(), xy.min())

            # only two peaks or throughs for now
            # defect_id = self.defect_types[defect_label]['id']
            defect_id = 1 if "T" in defect_label else 2
            defect_ids = np.full_like(xy[:, 0], defect_id)

            xyc = np.concatenate((xy.T, [defect_ids]), axis=0).T
            _, d_mask = create_multiclass_lattice_mask(
                img_gray, xyc, scale=d_size_pizels, rmask=d_size_pizels
            )
            d_mask[d_mask >= 1] = defect_id
            d_mask = d_mask.T
            d_masks.append(d_mask)

        defect_mask = np.array(d_masks)
        # defect_mask = np.squeeze(defect_mask)

        if np.ndim(defect_mask) == 2:
            defect_mask = np.expand_dims(defect_mask, axis=0)

        # summed_mask = np.sum(d_masks, axis=0)[:,:,:,0]
        # summed_mask[summed_mask>=1] = 1
        # lattice_mask_backgr = 1 - summed_mask

        summed_mask = np.sum(defect_mask, axis=-1)

        # TODO: account for overlapping defect mask areas with different classes
        summed_mask = np.sum(summed_mask, axis=0)

        # for i in range(1, len(summed_mask)):
        #     mask_prev = summed_mask[i-1]
        #     mask_curr = summed_mask[i]
        #     mask_new = mask_prev + mask_curr

        #     id1 = np.unique(mask_prev)
        #     id2 = np.unique(mask_curr)
        #     ids_new = np.unique(mask_new)

        #     condition = (mask_new == id1) | (mask_new == id2)
        #     mask_new = np.where(condition, mask_new, [id1])

        # defect_mask = np.squeeze(defect_mask)
        # lattice_mask_backgr = 1 - summed_mask
        # lattice_mask_backgr = lattice_mask_backgr[..., None]
        # mask = np.concatenate((summed_mask, lattice_mask_backgr), axis=-1)
        mask = np.squeeze(summed_mask)
        mask[mask < 0] = 0
        condition = (mask == 0) | (mask == 1) | (mask == 2)
        mask = np.where(condition, mask, [1])

        # lattice_mask_b = 1 - np.sum(lattice_mask, axis=-1)
        # lattice_mask_backgr = 1 - np.sum(defect_mask, axis=-1)
        # print(lattice_mask_backgr.shape, defect_mask.shape, np.unique(defect_mask))
        # plt.imshow(lattice_mask_backgr)

        # defect_mask = np.concatenate((defect_mask, lattice_mask_backgr), axis=0)
        # sum_mask = np.sum(mask, axis=0)

        # lattice_mask = np.concatenate(
        #     (lattice_mask_a[..., None],
        #      lattice_mask_b[..., None],
        #      lattice_mask_backgr[..., None]), # we need to add a background class
        #     axis=-1)

        # lattice_mask_b = 1 - np.sum(lattice_mask, axis=-1)
        # lattice_mask = np.concatenate((lattice_mask, lattice_mask_b[..., None]), axis=-1)

        return mask

    def __getitem__(self, idx):
        # get annotations for this index
        ann = self.anns.iloc[idx]
        filename = ann["filename"]

        # get image for this index
        img = self.get_image(ann)
        # Z = self.get_Z(ann)

        resY, resX = img.size
        if resX != self.target_size:
            img = img.resize((self.target_size, self.target_size))

        n_anns = len(ann["regions"])
        boxes = []
        labels = []
        defect_mask = []
        # d_coords = {key: [] for key in self.defect_types.keys()}
        d_coords = {}

        # get bounding box coordinates and create a mask for each object type in the image
        for region in ann["regions"]:
            defect_type = str(region["region_attributes"]["Defect_Type"])
            label_id = self.defect_types[defect_type]["id"]
            labels.append(label_id)

            shape_attributes = region["shape_attributes"]
            bbox = self.get_bbox(shape_attributes)

            if resX != self.target_size:
                # print('image larger than target size. resizing bboxes...')
                bbox = self.resize_bbox(bbox, resX, resY)

            boxes.append(list(bbox))
            xc, yc = self.get_bbox_center(bbox)

            if defect_type not in d_coords:
                # create a new list for the coords
                d_coords[defect_type] = []

            # append to list of coords
            d_coords[defect_type].append([xc, yc])

        try:
            # TODO: check defect types for empty lists
            # create mask for each type of defect using their xy centers
            # defect_mask = self.create_defect_masks(idx, img, d_coords)
            defect_mask = self.create_defect_masks_multiclass(idx, img, d_coords)
        except Exception as e:
            # print(idx, resX, filename, d_coords)
            raise e

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        labels = torch.as_tensor(labels, dtype=torch.int64)
        mask = torch.as_tensor(defect_mask)
        image_ids = torch.tensor([idx])
        crowd = torch.zeros((n_anns,), dtype=torch.uint8)

        # Set the target dict
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = mask
        target["area"] = area
        target["image_id"] = image_ids
        target["iscrowd"] = crowd
        target["index"] = idx

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def load_all(self, dataloader):
        imgs = []
        masks = []
        paths = []

        for batch_idx, (images, targets) in enumerate(dataloader):
            image = images[0].permute(1, 2, 0).numpy()
            # image = images[0].numpy()
            image = image[:, :, 0]

            idx = targets[0]["index"]
            path = self.get_sxm_path(idx)
            paths.append(path)
            # target = targets[0]
            # # bboxes = target['boxes'].numpy()
            # # label_ids = target['labels'].numpy()
            # # image_id = target['image_id'].numpy()[0]
            # # filename = self.img_filenames[image_id]
            defect_mask = targets[0]["masks"].numpy()
            # # g_lattice = targets[0]['lattice'].numpy()
            # # defect_mask = defect_mask[1]

            imgs.append(image)
            masks.append(defect_mask)

        Zs = self.get_Zs(paths)
        Zs = np.array(Zs)

        imgs = np.array(imgs)

        shapes = [m.shape for m in masks]
        if len(set(shapes)) <= 1:
            masks = np.array(masks)

        return imgs, masks, Zs

    def load_all_Z(self, dataloader):
        paths = []

        for batch_idx, (images, targets) in enumerate(dataloader):
            idx = targets[0]["index"]
            path = self.get_sxm_path(idx)
            paths.append(path)

        Zs = self.get_Zs(paths)
        Zs = np.array(Zs)

        return Zs
