import gc
import math
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import PercentFormatter
import matplotlib.lines as mlines
from pathlib import Path

from atomai.predictors import EnsemblePredictor
from atomai.predictors import Locator
from sklearn.cluster import MeanShift
from sklearn.cluster import KMeans
from fcmeans import FCM

# Custom modules
from .utils import collate_fn, estimate_noise

from nanoscopy.spm.process import correct_image

plot_out_path = "output/plots"


def COM_coords(nn_output_channel, thresh):
    loc = Locator(thresh, dist_edge=0, refine=False, d=None)
    coordinates = loc.run(nn_output_channel)
    coords = coordinates[0].astype(int)
    return coords


def read_Zfwd(dataset, idx):
    try:
        z_path = dataset.get_sxm_path(idx)
        z_fwd = dataset.get_Z(z_path)[0]

        if z_fwd.shape[0] != 512:
            z_fwd = cv2.resize(z_fwd, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)

        return z_fwd

    except Exception as e:
        print(z_path)
        print(e)
        return None


def get_defect_height(Z, contour, label):
    # print(Z.shape, contour.shape, label)
    # Create a mask image that contains the contour filled in
    cimg = np.zeros_like(Z)
    cv2.drawContours(cimg, [contour], 0, color=255, thickness=-1)

    # Access the image pixels inside the contour
    # then add to a new 1D numpy array
    pts = np.where(cimg == 255)
    bkg_pts = np.where(cimg != 255)

    height_mean = Z[pts[0], pts[1]].mean()
    height_std = Z[pts[0], pts[1]].std()
    background_mean = Z[bkg_pts[0], bkg_pts[1]].mean()
    background_std = Z[bkg_pts[0], bkg_pts[1]].std()

    if label == "peak":
        height = Z[pts[0], pts[1]].max()
        height = height_mean if height_mean > height else height
    else:
        height = Z[pts[0], pts[1]].min()
        height = height_mean if height_mean < height else height

    return height, height_mean, height_std, background_mean, background_std


def calculate_solidity(contour):
    if len(contour) > 5:
        area = float(cv2.contourArea(contour))
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area
    else:
        solidity = 0

    return solidity


def draw_ellipse(image, ellipse):
    (xc, yc), (MA, ma), angle = ellipse
    # draw major axis line in red
    rmajor = max(MA, ma) / 2
    if angle > 90:
        angle = angle - 90
    else:
        angle = angle + 90

    x1 = xc + math.cos(math.radians(angle)) * rmajor
    y1 = yc + math.sin(math.radians(angle)) * rmajor
    x2 = xc + math.cos(math.radians(angle + 180)) * rmajor
    y2 = yc + math.sin(math.radians(angle + 180)) * rmajor
    cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 255), thickness=1)

    # draw minor axis line in blue
    rminor = min(MA, ma) / 2
    if angle > 90:
        angle = angle - 90
    else:
        angle = angle + 90

    x1 = xc + math.cos(math.radians(angle)) * rminor
    y1 = yc + math.sin(math.radians(angle)) * rminor
    x2 = xc + math.cos(math.radians(angle + 180)) * rminor
    y2 = yc + math.sin(math.radians(angle + 180)) * rminor
    cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 255), thickness=1)

    return image


def draw_diameter(image, xc, yc, r_c):
    x1 = xc + r_c
    y1 = yc
    x2 = xc - r_c
    y2 = yc
    cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color=(0), thickness=1)
    return image


def ellipse_properties(contour):
    if len(contour) > 5:
        ellipse = cv2.fitEllipse(contour)
    else:
        ellipse = ((0, 0), (0, 0), 0)

    (xc, yc), (ma, MA), angle = ellipse

    if (MA == 0) | (ma == 0):
        eccentricity = 2
        aspect_ratio = 0
    else:
        eccentricity = round(np.sqrt(pow(MA, 2) - pow(ma, 2)) / MA, 3)
        aspect_ratio = MA / ma

    return ellipse, eccentricity, aspect_ratio


def crop_defect(bbox, image, pad=0):
    img_w = image.shape[0]
    # print("\n", image.shape, "pad=", pad)

    # Ensure the crop bounds are within the image bounds by padding the image
    # with enough zeros so that any bbox of size (w,h) is in bounds
    x, y, w, h = bbox
    total_pad = int(max(w, h) + pad)

    pad_arr = (
        [(total_pad,), (total_pad,), (0,)]
        if len(image.shape) == 3
        else [(total_pad,), (total_pad,)]
    )
    image = np.pad(image, pad_arr, mode="constant", constant_values=0)
    x = int(x + total_pad)
    y = int(y + total_pad)
    w = int(w)
    h = int(h)

    # print(
    #     "\nBBOX = ",
    #     bbox,
    #     "\nNew BBox = ",
    #     (x, y, w, h),
    # )

    # bbox_half = w // 2
    # bbox_x = 0 if x - pad < 0 else x - pad
    # bbox_y = 0 if y - pad < 0 else y - pad
    # bbox_bot_right_x = img_w if bbox_half + bbox_x + pad > img_w else bbox_half + bbox_x + pad
    # bbox_bot_right_y = img_w if bbox_half + bbox_y + pad > img_w else bbox_half + bbox_y + pad

    # bbox_w = bbox_bot_right_x - bbox_x
    # bbox_h = bbox_bot_right_y - bbox_y

    # defect_crop = image[bbox_y: bbox_y + bbox_h, bbox_x : bbox_x + bbox_w]

    defect_crop = image[y - pad : y + h + pad, x - pad : x + w + pad]

    return defect_crop


def rpdf(coords):
    # calculate radial probabilitiy distribution
    defect_g_r = []
    defect_radii = []
    # defect_g_r = {'trough': [], 'peak': []}
    # defect_radii = {'trough': [], 'peak': []}

    if len(coords) > 0:
        particles = coords[:, [0, 1]]
        g_r, radii = rdf(particles, dr=0.1)
        radii_nm = radii * nm_per_pixel
        defect_g_r.append(g_r)
        defect_radii.append(radii_nm)

    return defect_g_r, defect_radii


def is_defect_valid(defect_props: dict):
    if defect_props["Height (pm)"] is not None:
        height_mask = (
            (defect_props["Type"] == "peak") & (defect_props["Height (pm)"] < 0)
        ) | ((defect_props["Type"] == "trough") & (defect_props["Height (pm)"] > 0))
    else:
        # height info is not avaiable so assume False
        height_mask = False

    drop_mask = (
        height_mask
        # | ((defect_props["Type"] == "peak") & (defect_props["Solidity"] == 1))
        | (defect_props["Minor Axis (nm)"] > 15)
        | (defect_props["Minor Axis (nm)"] < 0.2)
        | (defect_props["Solidity"] < 0.7)
        | (defect_props["Compactness"] < 0.7)
        | (defect_props["Area (nm)"] < 0.2)
    )

    return False if drop_mask else True


def clean_detections(defect_detections: pd.DataFrame):
    drop_mask = (
        ((defect_detections["Type"] == "peak") & (defect_detections["Height (pm)"] < 0))
        | (
            (defect_detections["Type"] == "trough")
            & (defect_detections["Height (pm)"] > 0)
        )
        | ((defect_detections["Type"] == "peak") & (defect_detections["Solidity"] == 1))
        | (defect_detections["Minor Axis (nm)"] > 15)
        | (defect_detections["Minor Axis (nm)"] < 0.2)
        | (defect_detections["Solidity"] < 0.7)
        | (defect_detections["Compactness"] < 0.7)
        | (defect_detections["Area (nm)"] < 0.2)
    )

    drop_index = defect_detections[drop_mask].index
    defect_detections = defect_detections.drop(drop_index)

    return defect_detections


def predict_ensemble(dataset, smodel, ensemble, nb_classes=3, output_shape=None):
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn
    )

    preds, pred_vars = [], []

    for batch_idx, batch_sample in enumerate(dataloader):
        # if batch_idx > 0:
        #     break

        images, targets = batch_sample
        image = images[0].permute(1, 2, 0).numpy()[:, :, 0]
        w, h = image.shape

        if output_shape is None:
            output_shape = (1, 3, w, h)

        predictor = EnsemblePredictor(
            smodel, ensemble, nb_classes=nb_classes, output_shape=output_shape
        )

        nn_output, nn_out_var = predictor.predict(image)
        pred, pred_var = nn_output.squeeze(), nn_out_var.squeeze()

        preds.append(pred)
        pred_vars.append(pred_vars)

    preds = np.array(preds)
    pred_vars = np.array(pred_vars)

    return preds, pred_vars


def detect_defects(
    image, predictor, Z=None, thresh=0.1, draw=True, metadata=(), crop_tight=False
):
    # opencv uses BGR not RGB
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)

    # Z = read_Zfwd(dataset, image_id)

    # target_ann = dataset.get_ann(image_id)
    # filename = target_ann["filename"]
    # scan_width_nm = int(target_ann["Scan_Width(nm)"])
    # w, h = image.shape
    # nm_per_pixel = float(scan_width_nm) / w
    # temperature = 77 if target_ann["Data_Source"] == 'UCF' else 300
    filename, scan_width_nm, nm_per_pixel, temperature = metadata

    nn_output, nn_out_var = predictor.predict(image)
    pred, pred_var = nn_output.squeeze(), nn_out_var.squeeze()
    # plt.imshow(pred)
    # ch_pred = nn_output[:, :, :, [1,0]]
    # trough_coords = COM_coords(ch_pred, thresh)

    # ch_pred = nn_output[:, :, :, [2,0]]
    # trough_coords = COM_coords(ch_pred, thresh)

    defect_properties_keys = [
        "Type",
        "Contour",
        "Area (nm)",
        "Hull Area (nm)",
        "Perimeter (nm)",
        "Diameter (nm)",
        "Height (pm)",
        "Height Mean (pm)",
        "Height Std (pm)",
        "Background Mean (pm)",
        "Background Std (pm)",
        "bbox",
        "ellipse",
        "xc",
        "yc",
        "Minor Axis (nm)",
        "Major Axis (nm)",
        "Angle",
        "Eccentricity",
        "Compactness",
        "Solidity",
        "Aspect Ratio",
        "Filename",
        "nm_per_pixel",
        "Scan Width (nm)",
        "Temperature (K)",
        # "Defect Crop",
        # "Pred Crop",
        # "Pred Var Crop",
        "Crop Path",
    ]

    defect_properties_list = []

    norm_pred = cv2.normalize(pred, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    swap_pred = np.swapaxes(norm_pred, 0, 2)

    for channel, label in zip(swap_pred[1:], ["trough", "peak"]):
        channel = np.rot90(np.fliplr(channel))

        ret, thresh_pred = cv2.threshold(
            channel, thresh, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        contours = cv2.findContours(
            thresh_pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )[-2]
        # print(len(contours))
        for i, contour in enumerate(contours):
            defect_properties_dict = dict.fromkeys(defect_properties_keys)

            area = cv2.contourArea(contour)
            r_c = np.sqrt(area / np.pi)
            perimeter = cv2.arcLength(contour, True)
            perimeter = round(perimeter, 4)
            compactness = (4 * np.pi * area) / perimeter**2 if perimeter > 0 else 0

            tmp = np.copy(image)
            img_w, _ = tmp.shape
            cv2.drawContours(tmp, contour, -1, (0, 255, 0), thickness=1)

            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            ellipse, eccentricity, aspect_ratio = ellipse_properties(contour)
            (xc, yc), (ma, MA), angle = ellipse
            # iamge = draw_ellipse(image, ellipse)

            # x,y,w,h = bbox
            # (x,y) is the top-left coordinate of the rectangle
            # and (w,h) be its width and height.
            bbox = cv2.boundingRect(contour)

            if crop_tight:
                defect_crop = crop_defect(bbox, tmp)
                pred_crop = crop_defect(bbox, pred)
                pred_var_crop = crop_defect(bbox, pred_var)
            else:
                bbox_s_nm = 7
                bbox_s = bbox_s_nm // nm_per_pixel
                bbox_half = bbox_s // 2

                bbox_top_left_x = xc - bbox_half
                bbox_top_left_y = yc - bbox_half
                bbox_bot_right_x = xc + bbox_half
                bbox_bot_right_y = yc + bbox_half

                bbox_w = bbox_bot_right_x - bbox_top_left_x
                bbox_h = bbox_bot_right_y - bbox_top_left_y
                # print(
                #     "\n bbox=",
                #     bbox,
                #     xc,
                #     yc,
                # )

                # print(
                #     "\n new bbox=",
                #     bbox_top_left_x,
                #     bbox_top_left_y,
                #     bbox_bot_right_x,
                #     bbox_bot_right_y,
                #     bbox_w,
                #     bbox_h,
                # )
                fixed_bbox = (bbox_top_left_x, bbox_top_left_y, bbox_w, bbox_h)
                # defect_crop = crop_defect(fixed_bbox, tmp, pad=0)
                # pred_crop = crop_defect(fixed_bbox, pred, pad=0)
                # pred_var_crop = crop_defect(fixed_bbox, pred_var, pad=0)

                defect_crop = None
                pred_crop = None
                pred_var_crop = None

            # print(
            #     "\nBBOX = ",
            #     bbox,
            #     defect_crop.shape,
            #     pred_crop.shape,
            #     pred_var_crop.shape,
            # )

            height_data = None if Z is None else get_defect_height(Z, contour, label)
            height, height_mean, height_std, background_mean, background_std = (
                None,
                None,
                None,
                None,
                None if height_data is None else np.array(height_data) * 1e12,
            )

            # crop_3ch = np.stack((defect_crop, defect_crop, defect_crop), axis=2)
            # crops = np.array(
            #     [
            #         crop_3ch,
            #         pred_crop,
            #         pred_var_crop,
            #     ]
            # )

            date_str = Path(filename).parts[-2]
            # date_str = filename.split("_")[1]
            name_str = Path(filename).stem
            crop_save_path = f"output/crops/{date_str}_{name_str}_det_{i}"

            defect_properties_dict["Type"] = label
            defect_properties_dict["Contour"] = contour
            defect_properties_dict["Height (pm)"] = height
            defect_properties_dict["Height Mean (pm)"] = height_mean
            defect_properties_dict["Height Std (pm)"] = height_std
            defect_properties_dict["Background Mean (pm)"] = background_mean
            defect_properties_dict["Background Std (pm)"] = background_std
            defect_properties_dict["Area (nm)"] = area * nm_per_pixel**2
            defect_properties_dict["Hull Area (nm)"] = hull_area * nm_per_pixel**2
            defect_properties_dict["Perimeter (nm)"] = perimeter * nm_per_pixel
            defect_properties_dict["Diameter (nm)"] = 2 * r_c * nm_per_pixel
            defect_properties_dict["bbox"] = bbox
            defect_properties_dict["ellipse"] = ellipse
            defect_properties_dict["xc"] = xc
            defect_properties_dict["yc"] = yc
            defect_properties_dict["Minor Axis (nm)"] = ma * nm_per_pixel
            defect_properties_dict["Major Axis (nm)"] = MA * nm_per_pixel
            defect_properties_dict["Angle"] = angle
            defect_properties_dict["Eccentricity"] = eccentricity
            defect_properties_dict["Aspect Ratio"] = aspect_ratio
            defect_properties_dict["Solidity"] = solidity
            defect_properties_dict["Compactness"] = compactness
            # defect_properties_dict["Defect Crop"] = defect_crop
            # defect_properties_dict["Pred Crop"] = pred_crop
            # defect_properties_dict["Pred Var Crop"] = pred_var_crop
            defect_properties_dict["Crop Path"] = crop_save_path
            defect_properties_dict["Filename"] = filename
            defect_properties_dict["nm_per_pixel"] = nm_per_pixel
            defect_properties_dict["Scan Width (nm)"] = scan_width_nm
            defect_properties_dict["Temperature (K)"] = temperature

            if is_defect_valid(defect_properties_dict):
                defect_properties_list.append(defect_properties_dict)

                # defect_crop = correct_image(defect_crop, blur=False, equalize=False)
                # plt.imshow(defect_crop)
                # plt.axis("off")
                # plt.show()

                # mpl.image.imsave(crop_save_path + ".png", defect_crop)
                # np.save(crop_save_path + ".npy", crops)

                # im = (
                #     Image.fromarray(defect_crop)
                #     .convert("RGB")
                #     .save(crop_save_path + ".png")
                # )
                # cv2.imwrite(crop_save_path + ".png", defect_crop)

                if draw:
                    cv2.drawContours(image, contour, -1, (0, 255, 0), thickness=2)

            else:
                pass
                # if draw:
                #     image = draw_ellipse(image, ellipse)
                #     cv2.drawContours(image, contour, -1, RED, thickness=2)

    return defect_properties_list, pred, pred_var


def detect_defects_dataset(image, predictor, dataset, image_id, thresh, draw):
    Z = read_Zfwd(dataset, image_id)

    target_ann = dataset.get_ann(image_id)
    filename = target_ann["filename"]
    scan_width_nm = int(target_ann["Scan_Width(nm)"])
    w, h = image.shape
    nm_per_pixel = float(scan_width_nm) / w
    temperature = 77 if target_ann["Data_Source"] == "UCF" else 300

    metadata = (filename, scan_width_nm, nm_per_pixel, temperature)

    return detect_defects(image, predictor, Z, thresh, draw, metadata)


def detect_defects_VIAdataset(smodel, ensemble, dataset, show_plots=False):
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn
    )

    defect_properties_list = []

    for batch_idx, batch_sample in enumerate(dataloader):
        # if batch_idx > 0:
        # break

        images, targets = batch_sample
        image = images[0].permute(1, 2, 0).numpy()
        image = image[:, :, 0]

        target = targets[0]
        bboxes = target["boxes"].numpy()
        label_ids = target["labels"].numpy()
        label_mask = target["masks"].numpy()
        image_id = target["image_id"].numpy()[0]

        w, h = image.shape

        Z = read_Zfwd(dataset, image_id)

        target_ann = dataset.get_ann(image_id)
        filename = target_ann["filename"]
        scan_width_nm = int(target_ann["Scan_Width(nm)"])
        w, h = image.shape

        # print("\n", w, h)
        nm_per_pixel = float(scan_width_nm) / w
        temperature = 77 if target_ann["Data_Source"] == "UCF" else 300

        metadata = (filename, scan_width_nm, nm_per_pixel, temperature)

        if scan_width_nm >= 50:
            # image = cv2.resize(
            #     image,
            #     dsize=(512, 512),
            #     interpolation=cv2.INTER_CUBIC,
            # )
            # Z = cv2.resize(
            #     Z,
            #     dsize=(512, 512),
            #     interpolation=cv2.INTER_CUBIC,
            # )
            predictor = EnsemblePredictor(
                smodel, ensemble, nb_classes=3, output_shape=(1, 3, w, h)
            )

            defects, pred, pred_var = detect_defects(
                image,
                predictor,
                Z,
                thresh=0.1,
                draw=False,
                metadata=metadata,
                crop_tight=False,
            )

            if show_plots:
                plt.imshow(image)
                date_str = Path(dataset.root).name
                plt.savefig(
                    plot_out_path + f"/{date_str}_det_{batch_idx}.svg",
                    bbox_inches="tight",
                    format="svg",
                    dpi=900,
                )
                plt.close()

            del image
            del Z
            gc.collect()

            defect_properties_list.extend(defects)

    test_defect_properties = pd.DataFrame(defect_properties_list)
    # test_defect_properties = clean_detections(test_defect_properties)
    if len(test_defect_properties) > 0:
        test_defect_properties, test_cluster_centers = fit_defect_clusters(
            test_defect_properties
        )

    return test_defect_properties, test_cluster_centers


def predict_dataset(smodel, ensemble, dataset, show_plots=False):
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn
    )

    defect_properties_list = []
    image_noise = []
    drawn_imgs = []
    test_cluster_centers = []
    scan_areas_nm = []

    for batch_idx, batch_sample in enumerate(loader):
        # if batch_idx > 0:
        #     break

        images, targets = batch_sample
        image = images[0].permute(1, 2, 0).numpy()
        image = image[:, :, 0]
        w, h = image.shape

        noise = estimate_noise(image)
        # image_noise.append(noise)

        if noise < 0.015:
            target = targets[0]
            index = target["index"]
            sxm = target["sxm"]
            Z = target["Z"]

            avg_z = np.mean(Z)
            std_z = np.std(Z)

            temperature = 77
            path = sxm.dataframe.at[0, "path"]
            scan_width_nm = sxm.dataframe.at[0, "width (m)"] * 1e9
            nm_per_pixel = scan_width_nm / w

            metadata = (path, scan_width_nm, nm_per_pixel, temperature)
            # print(path[:-5])

            if scan_width_nm >= 50:
                print(
                    "width =",
                    w,
                    "with_nm =",
                    scan_width_nm,
                    "nm per pixel = ",
                    nm_per_pixel,
                    "temp = ",
                    temperature,
                    "noise = ",
                    noise,
                    "Z avg=",
                    avg_z,
                    "Z std=",
                    std_z,
                )

                predictor = EnsemblePredictor(
                    smodel, ensemble, nb_classes=3, output_shape=(1, 3, w, h)
                )

                defects, pred, pred_var = detect_defects(
                    image,
                    predictor,
                    Z,
                    thresh=0.1,
                    draw=True,
                    metadata=metadata,
                    crop_tight=False,
                )

                # image_512 = cv2.resize(
                #     image,
                #     dsize=(512, 512),
                #     interpolation=cv2.INTER_CUBIC,
                # )
                # Z_512 = cv2.resize(
                #     Z,
                #     dsize=(512, 512),
                #     interpolation=cv2.INTER_CUBIC,
                # )
                # predictor_512 = EnsemblePredictor(
                #     smodel, ensemble, nb_classes=3, output_shape=(1, 3, 512, 512)
                # )

                # defects_512, pred_512, pred_var_512 = detect_defects(
                #     image_512, predictor_512, Z_512,
                #     thresh=0.1, draw=False, metadata=metadata,
                #     crop_tight=False,
                # )

                # # nm_per_pixel ratio used to train ensemble
                # t = 50 / 512
                # if nm_per_pixel != t:
                #     print("nm/pixel ratio does not match target.")
                #     print(nm_per_pixel, t)
                #     d = int(scan_width_nm / t)
                #     print("re-size input image to ", d)
                #     # if d < w:
                #     image_uniform = cv2.resize(
                #         image,
                #         dsize=(d, d),
                #         interpolation=cv2.INTER_CUBIC,
                #     )
                #     Z_uniform = cv2.resize(
                #         Z,
                #         dsize=(d, d),
                #         interpolation=cv2.INTER_CUBIC,
                #     )
                #     predictor_uniform = EnsemblePredictor(
                #         smodel,
                #         ensemble,
                #         nb_classes=3,
                #         output_shape=(1, 3, d, d),
                #     )

                #     defects_uniform, pred_uniform, pred_var_uniform = detect_defects(
                #         image_uniform, predictor_uniform, Z_uniform,
                #         thresh=0.1, draw=False, metadata=metadata,
                #         crop_tight=False,
                #     )

                # drawn_imgs.append(image)

                if show_plots:
                    plt.imshow(image)
                    # plt.savefig(
                    #     plot_out_path + f"/det_{batch_idx}.eps",
                    #     bbox_inches="tight",
                    #     format="eps",
                    #     dpi=900,
                    # )
                    date_str = Path(dataset.root).name
                    plt.savefig(
                        plot_out_path + f"/{date_str}_det_{batch_idx}.svg",
                        bbox_inches="tight",
                        format="svg",
                        dpi=900,
                    )
                    plt.close()
                    # plt.show()

                del image
                del sxm
                del Z
                gc.collect()

                if len(defects) > 0:
                    scan_areas_nm.append(scan_width_nm**2)
                    defect_properties_list.extend(defects)

    test_defect_properties = pd.DataFrame(defect_properties_list)
    # test_defect_properties = clean_detections(test_defect_properties)
    if len(test_defect_properties) > 0:
        test_defect_properties, test_cluster_centers = fit_defect_clusters(
            test_defect_properties
        )

    return test_defect_properties, test_cluster_centers


def fit_defect_clusters(
    defect_properties,
    n_clusters=2,
    cols=["Height (pm)", "Minor Axis (nm)"],
    norm=None,
    bandwidth=None,
):
    d_types = ["trough", "peak"]
    cluster_centers = {
        "fcm": {d_type: [] for d_type in d_types},
        "meanshift": {d_type: [] for d_type in d_types},
    }
    n_clusters_by_type = {"trough": 2, "peak": 2}

    for i, d_type in enumerate(d_types):
        df = defect_properties[defect_properties["Type"] == d_type].loc[:, cols]
        # df = df.sort_values(by=["Height (pm)"], ascending=True)

        # if d_type == "trough":
        #     df = df.sort_values(by=["Height (pm)"], ascending=False)
        # else:
        #     df = df.sort_values(by=["Height (pm)"], ascending=True)

        # df = df.loc[:, cols]
        # df = df.loc[:, ['Height (pm)', 'Width (nm)']]
        # df['Width (nm)'] = df[['MA', 'ma']].max(axis=1)
        # df = df.loc[:, ['Height (pm)', 'Width (nm)']]

        if norm == "min-max":
            df = (df - df.min()) / (df.max() - df.min())
        elif norm == "nm-scale":
            df["Height (pm)"] = df["Height (pm)"] * 1e-3
            df = df.rename({"Height (pm)": "Height (nm)"}, axis=1)

        df_np = df.to_numpy()
        n_clusters = n_clusters_by_type[d_type]
        fcm = FCM(n_clusters=n_clusters, random_state=42)
        fcm.fit(df_np)
        fcm_p = fcm.predict(df_np).astype(int)
        fcm_softp = fcm.soft_predict(df_np)
        highest_p = np.array([ps[c] for c, ps in zip(fcm_p, fcm_softp)])

        clustering = MeanShift(bandwidth=bandwidth).fit(df_np)
        df["meanshift"] = clustering.labels_ + 1

        label_offset = (i * n_clusters_by_type["trough"]) + 1
        defect_properties.loc[df.index, "fcm_s"] = highest_p
        defect_properties.loc[df.index, "fcm"] = fcm_p + label_offset
        defect_properties.loc[df.index, "meanshift"] = clustering.labels_ + label_offset

        cluster_centers["fcm"][d_type] = fcm._centers
        cluster_centers["meanshift"][d_type] = clustering.cluster_centers_

    return defect_properties, cluster_centers


def get_total_area(dataset):
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn
    )

    scan_areas_nm = []
    filenames = []

    for batch_idx, batch_sample in enumerate(loader):
        # if batch_idx > 0:
        #     break

        images, targets = batch_sample
        image = images[0].permute(1, 2, 0).numpy()
        image = image[:, :, 0]
        w, h = image.shape

        noise = estimate_noise(image)
        # image_noise.append(noise)

        if noise < 0.015:
            target = targets[0]
            index = target["index"]
            sxm = target["sxm"]
            Z = target["Z"]

            avg_z = np.mean(Z)
            std_z = np.std(Z)

            temperature = 77
            path = sxm.dataframe.at[0, "path"]
            scan_width_nm = sxm.dataframe.at[0, "width (m)"] * 1e9
            nm_per_pixel = scan_width_nm / w

            scan_areas_nm.append(scan_width_nm**2)
            filenames.append(path)

    return scan_areas_nm, filenames


def plot_detection_hists(defect_detections: pd.DataFrame, hue: str = "Type"):
    font = {"size": 12}
    rc("font", **font)

    fig, axes = plt.subplots(3, 2, figsize=(10, 10), constrained_layout=True)

    xs = [
        "Area (nm)",
        "Perimeter (nm)",
        "Major Axis (nm)",
        "Minor Axis (nm)",
        "Angle",
        "Height (pm)",
    ]
    xs_params = {
        "Area (nm)": {"label": "Area (nm$^2$)"},
        "Angle": {"label": "Angle ($^\circ$)"},
        "Height (pm)": {"bins": 100, "xlim": (-500, 500)},
    }

    for i, (x_label, ax) in enumerate(zip(xs, axes.flat)):
        axis_label = x_label
        bins = 50
        xlim = None

        if x_label in xs_params:
            params = xs_params[x_label]

            axis_label = params["label"] if "label" in params else x_label
            bins = params["bins"] if "bins" in params else 50

            if "xlim" in params:
                xlim = params["xlim"]
                ax.set_xlim(xlim)

        ax.set_xlabel(axis_label)

        g = sns.histplot(
            defect_detections,
            x=x_label,
            hue=hue,
            bins=bins,
            kde=True,
            alpha=0.66,
            fill=True,
            # stat="density",
            # common_norm=False,
            kde_kws={
                # 'fill':True,
                # 'bw_adjust': 0.15,
            },
            line_kws={
                "linewidth": 1.5,
            },
            palette=sns.color_palette("tab10"),
            ax=ax,
        )
        g.legend_.set_title("Defect Clusters")

        # binwidth = 5
        # ax.yaxis.set_major_formatter(
        #     PercentFormatter(1 / binwidth)
        # )  # show axis such that 1/binwidth corresponds to 100%
        # ax.set_ylabel(f"Probability for a bin width of {binwidth}")

        # ax.legend(
        #     # ncol=2,
        #     title="Defect Clusters",
        #     loc="upper right",
        #     title_fontsize=12,
        #     fontsize=12,
        #     labels=[4, 3, 2, 1],
        # )

        # putter capital letter in top left corner to denote subplots
        letter = chr(ord("@") + i + 1)
        ax.text(
            -0.18,
            1.0,
            f"{letter}",
            verticalalignment="top",
            horizontalalignment="left",
            transform=ax.transAxes,
            color="Black",
            fontsize=14,
            weight="bold",
        )

    plt.savefig(
        plot_out_path + "/defect_property_hist-grid_all-data_raw.svg",
        bbox_inches="tight",
        format="svg",
    )
    plt.savefig(
        plot_out_path + "/defect_property_hist-grid_all-data_raw.tiff",
        bbox_inches="tight",
        format="tiff",
        dpi=900,
    )

    plt.show()


def plot_calc_defect_properties(defect_detections: pd.DataFrame, hue: str = "Type"):
    font = {"size": 12}
    rc("font", **font)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.66), constrained_layout=True)

    xs = ["Aspect Ratio", "Eccentricity", "Compactness", "Solidity"]
    xs_params = {
        "Area (nm)": {"label": "Area (nm$^2$)"},
        "Angle": {"label": "Angle ($^\circ$)"},
        "Height (pm)": {"bins": 100, "xlim": (-500, 500)},
    }

    for i, (x_label, ax) in enumerate(zip(xs, axes.flat)):
        axis_label = x_label
        bins = 50
        xlim = None

        if x_label in xs_params:
            params = xs_params[x_label]

            axis_label = params["label"] if "label" in params else x_label
            bins = params["bins"] if "bins" in params else 50

            if "xlim" in params:
                xlim = params["xlim"]
                ax.set_xlim(xlim)

        ax.set_xlabel(axis_label)

        sns.histplot(
            defect_detections,
            x=x_label,
            hue=hue,
            bins=bins,
            kde=True,
            alpha=0.66,
            fill=True,
            # stat="probability",
            # common_norm=False,
            kde_kws={
                # 'fill':True,
                # 'bw_adjust': 0.15,
            },
            line_kws={
                "linewidth": 1.75,
            },
            palette=sns.color_palette("tab10"),
            ax=ax,
        )

        # ax.legend(
        #     # ncol=2,
        #     title="Defect Clusters",
        #     loc="upper right",
        #     title_fontsize=12,
        #     fontsize=12,
        #     labels=[4, 3, 2, 1],
        # )

        # putter capital letter in top left corner to denote subplots
        letter = chr(ord("@") + i + 1)
        ax.text(
            -0.18,
            1.0,
            f"{letter}",
            verticalalignment="top",
            horizontalalignment="left",
            transform=ax.transAxes,
            color="Black",
            fontsize=14,
            weight="bold",
        )

    plt.savefig(
        plot_out_path + "/calc_defect_property_hist-grid_all-data_raw.svg",
        bbox_inches="tight",
        format="svg",
    )
    plt.savefig(
        plot_out_path + "/calc_defect_property_hist-grid_all-data_raw.tiff",
        bbox_inches="tight",
        format="tiff",
        dpi=900,
    )
    plt.show()


def plot_detection_kdes(defect_detections: pd.DataFrame, hue: str = "Type", save=False):
    font = {"size": 12}
    rc("font", **font)

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), constrained_layout=True)

    xs = [
        "Minor Axis (nm)",
        "Height (pm)",
        # "Aspect Ratio",
        # "Angle",
        "Eccentricity",
        "Compactness",
        # "Solidity",
    ]
    xs_params = {
        "Minor Axis (nm)": {"xlim": (0, 4)},
        "Height (pm)": {"bins": 100, "xlim": (-500, 500)},
        "Eccentricity": {"xlim": (0, 1), "legend": "upper left"},
        "Compactness": {"xlim": (0.7, 1)},
        "Solidity": {"xlim": (0.85, 1), "legend": "upper left"},
        "Angle": {"xlim": (0, 180)},
    }

    for i, (x_label, ax) in enumerate(zip(xs, axes.flat)):
        legend_flag = True if i == 1 else False
        axis_label = x_label
        xlim = None
        loc = "best"

        ls = ["-", "--", "-.", ":"]

        # g.legend_.set_title("Cluster")
        # print(vars(g.legend_))
        for j in range(len(ls)):
            sel = defect_detections[defect_detections[hue] == j + 1]
            sns.kdeplot(
                sel,
                x=x_label,
                # hue=hue,
                color=sns.color_palette("tab10")[j],
                linestyle=ls[j],
                alpha=0.8,
                linewidth=1.5,
                fill=False,
                legend=False,
                # multiple="fill",
                # stat="density",
                common_norm=False,
                # palette=sns.color_palette("tab10"),
                ax=ax,
            )

        g = sns.kdeplot(
            defect_detections,
            x=x_label,
            hue=hue,
            alpha=0.15,
            linewidth=0.1,
            fill=True,
            legend=legend_flag,
            # multiple="fill",
            # stat="density",
            common_norm=False,
            palette=sns.color_palette("tab10"),
            ax=ax,
        )

        if x_label in xs_params:
            params = xs_params[x_label]

            axis_label = params["label"] if "label" in params else x_label
            loc = params["legend"] if "legend" in params else loc

            if "xlim" in params:
                xlim = params["xlim"]
                ax.set_xlim(xlim)

        ax.set_xlabel(axis_label, fontsize=12)
        ax.set_ylabel("Probability Density", fontsize=12)

        if legend_flag:
            # g.legend_.set_alpha(1)

            plt_lines = []
            for lh, j in zip(g.legend_.legendHandles, range(len(ls))):
                # lh.set_alpha(0.5)
                l = ls[j]
                plt_line = mlines.Line2D(
                    [],
                    [],
                    color=sns.color_palette("tab10")[j],
                    linestyle=l,
                    markersize=15,
                    label=j + 1,
                    alpha=0.8,
                )
                plt_lines.append(plt_line)

            ax.legend(handles=plt_lines)
            sns.move_legend(g, loc, title="Defect Cluster")

        # putter capital letter in top left corner to denote subplots
        letter = chr(ord("@") + i + 1)
        ax.text(
            -0.18,
            1.0,
            f"{letter}",
            verticalalignment="top",
            horizontalalignment="left",
            transform=ax.transAxes,
            color="Black",
            fontsize=14,
            weight="bold",
        )

    if save:
        plt.savefig(
            plot_out_path + "/defect_property_kde-grid_all-data_raw.svg",
            bbox_inches="tight",
            format="svg",
        )
        plt.savefig(
            plot_out_path + "/defect_property_kde-grid_all-data_raw.tiff",
            bbox_inches="tight",
            format="tiff",
            dpi=900,
        )
    plt.show()
