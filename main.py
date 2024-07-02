from pathlib import Path
from configs.config import MainConfig
from confz import BaseConfig, FileSource
import os
import torch
import numpy as np
from tqdm import tqdm
from utils.utils import load_detector, load_classificator, open_mapping, extract_crops
import pandas as pd
from itertools import repeat


def main():
    # Load main config
    main_config = MainConfig(config_sources=FileSource(file=os.path.join("configs", "config.yml")))
    device = main_config.device

    # Load imgs from source dir
    pathes_to_imgs = [i for i in Path(main_config.src_dir).glob("*")
                      if i.suffix.lower() in [".jpeg", ".jpg", ".png"]]

    # Load mapping for classification task
    mapping = open_mapping(path_mapping=main_config.mapping)

    # Separate main config
    detector_config = main_config.detector
    classificator_config = main_config.classificator

    # Load models
    detector = load_detector(detector_config).to(device)
    classificator = load_classificator(classificator_config).to(device)

    # Inference
    if len(pathes_to_imgs):

        list_predictions = []

        num_packages_det = np.ceil(len(pathes_to_imgs) / detector_config.batch_size).astype(np.int32)
        with torch.no_grad():
            for i in tqdm(range(num_packages_det), colour="green"):
                # Inference detector
                batch_images_det = pathes_to_imgs[detector_config.batch_size * i:
                                                  detector_config.batch_size * (1 + i)]
                results_det = detector(batch_images_det,
                                       iou=detector_config.iou,
                                       conf=detector_config.conf,
                                       imgsz=detector_config.imgsz,
                                       verbose=False,
                                       device=device)

                if len(results_det) > 0:
                    # Extract crop by bboxes
                    dict_crops = extract_crops(results_det, config=classificator_config)

                    # Inference classificator
                    for img_name, batch_images_cls in dict_crops.items():
                        # if len(batch_images_cls) > classificator_config.batch_size:
                        num_packages_cls = np.ceil(len(batch_images_cls) / classificator_config.batch_size).astype(
                            np.int32)
                        for j in range(num_packages_cls):
                            batch_images_cls = batch_images_cls[classificator_config.batch_size * j:
                                                                classificator_config.batch_size * (1 + j)]
                            logits = classificator(batch_images_cls.to(device))
                            probabilities = torch.nn.functional.softmax(logits, dim=1)
                            top_p, top_class_idx = probabilities.topk(1, dim=1)

                            # Locate torch Tensors to cpu and convert to numpy
                            top_p = top_p.cpu().numpy().ravel()
                            top_class_idx = top_class_idx.cpu().numpy().ravel()

                            class_names = [mapping[top_class_idx[idx]] for idx, _ in enumerate(batch_images_cls)]

                            list_predictions.extend([[name, cls, prob] for name, cls, prob in
                                                     zip(repeat(img_name, len(class_names)), class_names, top_p)])

        # Create Dataframe with predictions
        table = pd.DataFrame(list_predictions, columns=["image_name", "class_name", "confidence"])
        # table.to_csv("table.csv", index=False) # Раскомментируйте, если хотите увидеть результаты предсказания
        # нейронной сети по каждому найденному объекту

        agg_functions = {
            'class_name': ['count'],
            "confidence": ["mean"]
        }
        groupped = table.groupby(['image_name', "class_name"]).agg(agg_functions)
        img_names = groupped.index.get_level_values("image_name").unique()

        final_res = []

        for img_name in img_names:
            groupped_per_img = groupped.query(f"image_name == '{img_name}'")
            max_num_objects = groupped_per_img["class_name", "count"].max()
            # max_confidence = groupped_per_img["class_name", "confidence"].max()
            statistic_by_max_objects = groupped_per_img[groupped_per_img["class_name", "count"] == max_num_objects]

            if len(statistic_by_max_objects) > 1:
                # statistic_by_max_mean_conf = statistic_by_max_objects.reset_index().max().values
                statistic_by_max_mean_conf = statistic_by_max_objects.loc[[statistic_by_max_objects["confidence", "mean"].idxmax()]]
                final_res.extend(statistic_by_max_mean_conf.reset_index().values)
            else:
                final_res.extend(statistic_by_max_objects.reset_index().values)
        # groupped.to_csv("table_agg.csv", index=True) # Раскомментируйте, если хотите увидеть результаты аггрегации

        final_table = pd.DataFrame(final_res, columns=["image_name", "class_name", "count", "confidence"])
        final_table.to_csv("table_final.csv", index=False)


if __name__ == '__main__':
    main()
