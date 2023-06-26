import argparse
import os

# Setup detectron2 logger
from detectron2.utils.logger import setup_logger

setup_logger()
# import some common detectron2 utilities
from detectron2.config import get_cfg
from furuno_wrapper import setup, custom_mapper, AugTrainer, Fish_Detection


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def weight_path(string):
    if os.path.isdir(string):
        return string
    else:
        return None


parser = argparse.ArgumentParser(description='Train detectron for Furuno data')
parser.add_argument('--dataset-dir', dest='dataset_dir', type=dir_path, required=True,
                    help='The parent dir of data')
parser.add_argument('--train-dir', dest='train_dir', type=str, required=True,
                    help='The dir of COCO training annotations')
parser.add_argument('--test-dir', dest='test_dir', type=str, required=True,
                    help='The dir of COCO test annotations')
parser.add_argument('--is-retrain', dest='is_retrain', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help='Train at beginning or not')
parser.add_argument('--train-annotations', dest='train_annotations', type=str, required=True,
                    help="Training json file")
parser.add_argument('--test-annotations', dest='test_annotations', type=str, required=True,
                    help="Testing json file")
parser.add_argument('--weight-dir', dest='weight_dir', type=weight_path,
                    help='The path to the pre-trained weight, if None, using default weight in config_name')
parser.add_argument('--config-name', dest='config_name', type=str, required=True,
                    help="dir name of the considered model in model zoo")
parser.add_argument('--num-workers', dest='num_workers', type=int,
                    help=' number of workers nodes used to train model')
parser.add_argument('--ims-per-batch', dest='ims_per_batch', type=int,
                    help=' number of images per batch')
parser.add_argument('--base-lr', dest='base_lr', type=float,
                    help=' learning rate')
parser.add_argument('--warmup-iters', dest='warmup_iters', type=int,
                    help=' number of warmup iteration')
parser.add_argument('--max-iter', dest='max_iter', type=int,
                    help='maximum number of iterations')
parser.add_argument('--checkpoint-period', dest='checkpoint_period', type=int,
                    help='Period for the checkpoint')
parser.add_argument('--batch-size-per-image', dest='batch_size_per_image', type=int,
                    help='Batch size per image')
parser.add_argument('--num-classes', dest='num_classes', type=int,
                    help='number of classes')
parser.add_argument('--threshold', dest='threshold', type=float,
                    help='prediction threshold')
args = parser.parse_args()


def main(args):
    cfg = get_cfg()
    config_name = args.config_name
    num_workers = args.num_workers
    ims_per_batch = args.ims_per_batch
    base_lr = args.base_lr
    warmup_iters = args.warmup_iters
    max_iter = args.max_iter
    checkpoint_period = args.checkpoint_period
    batch_size_per_image = args.batch_size_per_image
    num_classes = args.num_classes
    dataset_dir = args.dataset_dir
    train_dir = args.train_dir
    test_dir = args.test_dir
    train_annotations = args.train_annotations
    test_annotations = args.test_annotations
    weight_dir = args.weight_dir
    is_retrain = args.is_retrain
    Trainer = AugTrainer
    threshold = args.threshold

    cfg, trainer, test_dataset_dicts, test_metadata_dicts = setup(
        cfg, config_name, num_workers, ims_per_batch,
        base_lr, warmup_iters, max_iter, checkpoint_period,
        batch_size_per_image, num_classes, dataset_dir,
        train_dir, test_dir, weight_dir, is_retrain, Trainer,
        train_annotations, test_annotations, custom_mapper)
    fish_detector = Fish_Detection(cfg, trainer, custom_mapper)
    fish_detector.train()
    fish_detector.evaluate()


if __name__ == "__main__":
    main(args)
