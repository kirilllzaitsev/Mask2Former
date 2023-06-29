# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import copy
import functools
import glob
import json
import multiprocessing as mp
import os

# fmt: off
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import time
from pathlib import Path

import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from mask2former import add_maskformer2_config
from predictor import VisualizationDemo

# constants
WINDOW_NAME = "MaskFormer demo"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--path-to-data-config",
        help="KITTI-format file with paths to files",
    )

    parser.add_argument(
        "--categories-to-extract",
        nargs="+",
        help="thing/stuff to extract the mask for",
        default=[""],
    )

    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def gen_masks(paths, run_config_initial, demo, logger, config_name, args):
    run_config = copy.deepcopy(run_config_initial)
    for path in paths:
        # for path in tqdm.tqdm(paths, disable=not args.output):
        # use PIL, to be consistent with evaluation
        img = read_image(path, format="BGR")
        start_time = time.time()
        predictions, visualized_output = demo.run_on_image(img)
        # contiguous id is unclear
        # what is used is stuff_dataset_id
        predicted_class_idxs = predictions["sem_seg"].argmax(dim=0).unique().tolist()
        # predicted_class_names = [
        #     demo.metadata.stuff_classes[i] for i in predicted_class_idxs
        # ]

        logger.info(
            "{}: {} in {:.2f}s".format(
                path,
                "detected {} instances".format(len(predictions["instances"]))
                if "instances" in predictions
                else "finished",
                time.time() - start_time,
            )
        )

        for predicted_class_idx in predicted_class_idxs:
            class_name = demo.metadata.stuff_classes[predicted_class_idx]
            input_img_name = path
            if 1 or class_name in args.categories_to_extract:
                save_dir = Path(args.output) / config_name / os.path.basename(input_img_name)

                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                mask = predictions["sem_seg"].argmax(dim=0) == predicted_class_idx
                mask = mask.cpu().numpy()
                mask = mask.astype(np.uint8) * 255
                mask_path = os.path.join(
                    save_dir,
                    f"{class_name}.png",
                )
                cv2.imwrite(mask_path, mask)
                run_config["detected_categories"]["class2img"][class_name][
                    "num_occurences"
                ] += 1
                run_config["detected_categories"]["class2img"][class_name][
                    "input_img_paths"
                ].append(path)
                run_config["detected_categories"]["class2img"][class_name][
                    "mask_paths"
                ].append(mask_path)
                run_config["detected_categories"]["img2class"][input_img_name][
                    "mask_paths"
                ].append(mask_path)
                run_config["detected_categories"]["img2class"][input_img_name][
                    "num_objects"
                ] += 1
                run_config["detected_categories"]["img2class"][input_img_name][
                    "object_classes"
                ].append(class_name)

    return run_config


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    setup_logger(name="fvcore")
    logger = setup_logger()

    args = get_parser().parse_args()
    path_to_data_config = Path(args.path_to_data_config)

    with open(path_to_data_config, "r") as f:
        input_img_paths = f.readlines()

    base_dir = Path("/media/master/wext/cv_data/kitti-full")
    input_img_paths = [str(base_dir / p.strip()) for p in input_img_paths]
    print(len(input_img_paths))
    # args.input = [input_img_paths[0]]
    args.input = input_img_paths

    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)
    config_name = path_to_data_config.name.split(".")[0]

    demo = VisualizationDemo(cfg)

    if len(args.input) == 1:
        args.input = glob.glob(os.path.expanduser(args.input[0]))
        assert len(args.input) >= 1, "The input path(s) was not found"

    paths = args.input
    # paths = [args.input[0]]
    run_config = {
        "detected_categories": {
            "class2img": {
                x: {"num_occurences": 0, "input_img_paths": [], "mask_paths": []}
                for x in demo.metadata.stuff_classes
            },
            "img2class": {
                x: {
                    "num_objects": 0,
                    "object_classes": [],
                    "mask_paths": [],
                }
                for x in paths
            },
        },
    }

    num_proc = 2
    # num_proc = 1
    with mp.Pool(num_proc) as pool:
        results = pool.map(
            functools.partial(
                gen_masks,
                run_config_initial=run_config,
                demo=demo,
                logger=logger,
                config_name=config_name,
                args=args,
            ),
            np.array_split(paths, num_proc),
        )
        for run_config_res in results:
            for mapping in ["img2class", "class2img"]:
                for k, v in run_config_res["detected_categories"][mapping].items():
                    run_config["detected_categories"][mapping][k]["mask_paths"].extend(
                        v["mask_paths"]
                    )
            for k, v in run_config_res["detected_categories"]["class2img"].items():
                run_config["detected_categories"]["class2img"][k][
                    "input_img_paths"
                ].extend(v["input_img_paths"])
                run_config["detected_categories"]["class2img"][k][
                    "num_occurences"
                ] += v["num_occurences"]
            for k, v in run_config_res["detected_categories"]["img2class"].items():
                run_config["detected_categories"]["img2class"][k]["num_objects"] += v[
                    "num_objects"
                ]
                run_config["detected_categories"]["img2class"][k]["object_classes"].extend(v["object_classes"])

    run_config["detected_categories"]["class2img"] = {
        k: v
        for k, v in run_config["detected_categories"]["class2img"].items()
        if v["num_occurences"] > 0
    }
    save_dir = Path(args.output)
    # with open(os.path.join(save_dir, "run_config.json"), "w") as f:
    #     json.dump(run_config, f)
    for mapping in ["img2class", "class2img"]:
        with open(os.path.join(save_dir, f"{mapping}.json"), "w") as f:
            json.dump(
                {"file_paths_config_name": config_name, mapping: run_config["detected_categories"][mapping]}, f
            )
