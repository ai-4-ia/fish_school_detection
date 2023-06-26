import copy
import os

import skimage.io as io
import torch
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger

setup_logger()
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.engine import DefaultTrainer
import detectron2.data.transforms as T
from detectron2.data.datasets import register_coco_instances


def custom_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    transform_list = [T.Resize((450, 450))
                      # T.RandomBrightness(0.9, 1.1),
                      # T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
                      # T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                      # T.RandomCrop("absolute", (240, 240))
                      ]
    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict


class AugTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)


def setup(cfg, config_name, num_workers, ims_per_batch, base_lr, warmup_iters, max_iter, checkpoint_period,
          batch_size_per_image, num_classes, dataset_dir, train_dir, test_dir, weight_dir, is_retrain, Trainer,
          train_annotations, test_annotations, custom_mapper):
    register_coco_instances("train", {}, os.path.join(dataset_dir, train_dir, train_annotations),
                            os.path.join(dataset_dir, train_dir))
    register_coco_instances("test", {}, os.path.join(dataset_dir, test_dir, test_annotations),
                            os.path.join(dataset_dir, test_dir))
    dataset_dicts = DatasetCatalog.get("train")
    metadata_dicts = MetadataCatalog.get("train")
    test_dataset_dicts = DatasetCatalog.get("test")
    test_metadata_dicts = MetadataCatalog.get("test")

    cfg.merge_from_file(model_zoo.get_config_file(config_name))
    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.TEST = ("test",)
    cfg.DATALOADER.NUM_WORKERS = num_workers
    if weight_dir == None:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_name)
    else:
        cfg.MODEL.WEIGHTS = weight_dir
    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
    cfg.SOLVER.BASE_LR = base_lr
    cfg.SOLVER.WARMUP_ITERS = warmup_iters
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.CHECKPOINT_PERIOD = checkpoint_period
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size_per_image
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = 'cpu'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=is_retrain)
    return cfg, trainer, test_dataset_dicts, test_metadata_dicts


class Fish_Detection():

    def __init__(self, cfg, trainer, custom_mapper):
        self.cfg = cfg
        self.trainer = trainer
        self.custom_mapper = custom_mapper

    def train(self):
        trainer = self.trainer.train()
        cfg = self.cfg
        return cfg, trainer

    def evaluate(self):
        evaluator = COCOEvaluator("test", self.cfg, False, output_dir="./output/")
        val_loader = build_detection_test_loader(self.cfg, "test", mapper=self.custom_mapper)
        inference_on_dataset(self.trainer.model, val_loader, evaluator)

    def predict(self, threshold):
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # threshold = 0.5
        self.cfg.DATASETS.TEST = ("test",)
        predictor = DefaultPredictor(self.cfg)

        return predictor


class Fish_Inference():

    def __init__(self, cfg, model_path):
        self.cfg = cfg
        self.model_path = model_path

    def predict(self, threshold=0.5):
        self.cfg.MODEL.WEIGHTS = os.path.join(self.model_path, "model_final.pth")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # threshold = 0.5
        self.cfg.DATASETS.TEST = ("test",)
        predictor = DefaultPredictor(self.cfg)

        return predictor

    def visualize(self, image, predictor, test_metadata_dicts):
        im = io.imread(image["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=test_metadata_dicts,
                       scale=0.4
                       # instance_mode=ColorMode.IMAGE_BW
                       )
        output_plot = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        return output_plot
