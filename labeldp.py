
# JAX/Flax imports
import jax
from flax.training.checkpoints import restore_checkpoint
from label_dp.label_dp.train import multi_stage_train, create_train_state
from label_dp.label_dp.models import CifarResNet18V1, CifarResNet50V1, LogisticRegression, CifarViTSmall16, CifarViTBase16
import label_dp.label_dp.utils as utils
import ml_collections
import tensorflow as tf
from flax.training.common_utils import shard
import tensorflow_datasets as tfds
import jax.numpy as jnp
import logging as native_logging
from absl import logging
from clu import platform

# Parsing and helper imports
import argparse
import yaml
import copy
import os
from typing import Any, Optional, Tuple
import time 
import random
import numpy as np
import datetime


tf.config.experimental.set_visible_devices([], 'GPU')  # Ensure TF doesn't hog GPU if JAX is using it

# mapping from your shorthand to TFDS names
_DS_MAP = {
    "mnist":        "mnist",
    "fmnist":       "fashion_mnist",
    "kmnist":       "kmnist",
    "svhn_cropped": "svhn_cropped",
    "cifar10":      "cifar10",
    "cifar100":     "cifar100",
}

def preprocess_example(example, resize_to_224=False):
    # normalize image to [0, 1]
    img = tf.cast(example["image"], tf.float32) / 255.0
    if resize_to_224:
        img = tf.image.resize(img, [224, 224])
    lbl = example["label"]
    return {"image": img, "label": lbl}

def load_and_split_dataset(dataset_name: str,
                           batch_size: int = 128,
                           forget_size: int = 100,
                           seed: int = 0, use_vit: bool = False):
 
    tfds_name = _DS_MAP.get(dataset_name.lower())
    if tfds_name is None:
        raise ValueError(f"Unsupported dataset: {dataset_name!r}")

    # load raw splits
    train_ds = tfds.load(tfds_name, split="train", as_supervised=False)
    test_ds  = tfds.load(tfds_name, split="test",  as_supervised=False)

    # preprocess, shuffle train, then split
    # train_ds = train_ds.map(preprocess_example, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(lambda x: preprocess_example(x, resize_to_224=use_vit), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(10_000, seed=seed)

    forget_ds = train_ds.take(forget_size)
    retain_ds = train_ds.skip(forget_size)

    # batch & prefetch all
    def batchify(ds):
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return batchify(retain_ds), batchify(forget_ds), batchify(
        #test_ds.map(preprocess_example, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds.map(lambda x: preprocess_example(x, resize_to_224=use_vit), num_parallel_calls=tf.data.AUTOTUNE)
    )


def compute_forget_confidence_distance(flax_state, model, forget_ds, args, num_classes: int = 10):
    total_conf_dist = 0.0
    total_count = 0
    uniform_prob = 1.0 / num_classes

    for batch in tfds.as_numpy(forget_ds):
        x = jnp.array(batch['image'])  
        if jnp.any(jnp.isnan(x)):
            print("WARNING: NaN detected in input images")
            continue
        #forward
        if args.model_type in ["CifarResNet18V1", "CifarResNet50V1"]:
            variables = {
                'params': flax_state.params,
                'batch_stats': flax_state.model_states['batch_stats']  
            }
            logits = model.apply(variables, x, train=False)
        elif args.model_type in ["CifarViTSmall16", "CifarViTBase16"]:
            variables = {
                'params': flax_state.params
            }
            logits = model.apply(variables, x, train=False)
        else:  
            variables = {
                'params': flax_state.params
            }
            logits = model.apply(variables, x)
            
        probs = jax.nn.softmax(logits, axis=-1) 
        if jnp.any(jnp.isnan(probs)):
            print("WARNING: NaN detected in probabilities")
            print(f"Logits range: {jnp.min(logits)} to {jnp.max(logits)}")
            continue                               
        p_max = jnp.max(probs, axis=-1)                                         
        conf_dist = jnp.maximum(p_max - uniform_prob, 0)  
        if jnp.any(jnp.isnan(conf_dist)):
            print("WARNING: NaN detected in confidence distance")
            continue                       

        total_conf_dist += jnp.sum(conf_dist)
        total_count += x.shape[0]

    if total_count == 0:
        return 0.0
    return float(total_conf_dist) / total_count

def evaluate_model(flax_state, model, test_ds, args):
    total_correct = 0
    total_seen = 0

    for batch in tfds.as_numpy(test_ds):
        x = jnp.array(batch['image'])  
        y = jnp.array(batch['label']) 

        # if args.model_type == "CifarResNet18V1" or args.model_type == "CifarResNet50V1":
        #     variables = {
        #         'params': flax_state.params,
        #         'batch_stats': flax_state.model_states['batch_stats']  
        #     }
        #     logits = model.apply(variables, x, train=False)
        # else: 
        #     variables = {
        #         'params': flax_state.params
        #     }
        #     logits = model.apply(variables, x)
        if args.model_type in ["CifarResNet18V1", "CifarResNet50V1"]:
            variables = {
                'params': flax_state.params,
                'batch_stats': flax_state.model_states['batch_stats']  
            }
            logits = model.apply(variables, x, train=False)
        elif args.model_type in ["CifarViTSmall16", "CifarViTBase16"]:
            # ViT models don't use batch normalization, only params
            variables = {
                'params': flax_state.params
            }
            logits = model.apply(variables, x, train=False)
        else:  # LogisticRegression and others
            variables = {
                'params': flax_state.params
            }
            logits = model.apply(variables, x)

        preds = jnp.argmax(logits, axis=-1)
        total_correct += jnp.sum(preds == y)
        total_seen += x.shape[0]

    accuracy = float(total_correct) / float(total_seen)
    return accuracy


def load_config(path: str) -> ml_collections.ConfigDict:
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)
    return ml_collections.ConfigDict(raw)



def config_to_args(config: dict) -> argparse.Namespace:
    # replicate your old function to map YAML to args
    from copy import deepcopy
    namespace = argparse.Namespace()
    # most stuff
    namespace.dataset = config['data']['name']
    namespace.seed = config['train']['run_seed']
    namespace.model_type = config['model']['arch']
    namespace.num_classes = config['model']['kwargs']['num_classes']
    namespace.batch_size = config['train']['batch_size']
    namespace.epochs = config['train']['num_epochs']
    namespace.lr = config['optimizer']['learning_rate']

    # paths
    paths = config['paths']
    namespace.trained_model_save_path = paths['trained_model_save_path']
    namespace.trained_model_load_path = paths['trained_model_load_path']
    namespace.forget_images_path = paths['forget_images_path']
    namespace.results_file_path = paths['results_file_path']

    # rest omitted; load_data only needs dataset, batch_size, num_classes, seed
    return namespace


def main(config_path: str, workdir: str):
    """Jax part: multistep training"""
    # Load YAML config and args
    config = yaml.safe_load(open(config_path, 'r'))
    configs = load_config(config_path)
    args = config_to_args(config)

    # Ensure workdir
    os.makedirs(workdir, exist_ok=True)

    # Hide TF GPUs to leave memory for JAX
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.config.experimental.set_visible_devices([], 'GPU')

    # Run multi-stage LabelDP training
    # logging
    logdir = os.path.join(workdir, 'logs')
    tf.io.gfile.makedirs(logdir)
    log_file = os.path.join(
        logdir, datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.txt')
    log_format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    formatter = native_logging.Formatter(log_format)
    file_stream = tf.io.gfile.GFile(log_file, 'w')
    handler = native_logging.StreamHandler(file_stream)
    handler.setLevel(native_logging.INFO)
    handler.setFormatter(formatter)
    logging.get_absl_logger().addHandler(handler)

    if jax.process_index() == 0:
        work_unit = platform.work_unit()
        work_unit.create_artifact(
            artifact_type=platform.ArtifactType.DIRECTORY,
            artifact=workdir, description='Working directory')
        work_unit.create_artifact(
            artifact_type=platform.ArtifactType.FILE,
            artifact=log_file, description='Log file')


    train_start = time.time()
    multi_stage_train(configs, workdir)
    logging.flush()
    train_dur = time.time() - train_start 

    print(f"got here: {train_dur}")

    # Restore final Flax state
    if args.model_type in ["CifarViTSmall16", "CifarViTBase16"]:
        data_shape = [1, 224, 224, 3]  # ViT models use 224x224
    elif args.dataset in ["mnist", "fmnist", "kmnist"]: 
        data_shape = [1, 28, 28, 3]
    elif args.dataset in ["cifar100", "cifar10", "svhn_cropped"]:
        data_shape = [1, 32, 32, 3]

    last_stage = len(configs.stage_specs) - 1
    ckpt_dir = os.path.abspath(os.path.join(workdir, f'stage{last_stage}'))
    print(ckpt_dir)

    if args.model_type == "CifarResNet18V1":
        model = CifarResNet18V1(num_classes=args.num_classes)
    elif args.model_type == "CifarResNet50V1":
        model = CifarResNet50V1(num_classes=args.num_classes)
    elif args.model_type == "LogisticRegression":
        model = LogisticRegression(num_classes=args.num_classes)
    elif args.model_type == "CifarViTSmall16":
        model = CifarViTSmall16(num_classes=args.num_classes)
    elif args.model_type == "CifarViTBase16":
        model = CifarViTBase16(num_classes=args.num_classes)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    rng = jax.random.PRNGKey(args.seed + last_stage)

    lr_fn = utils.build_lr_fn(
        configs.lr_fn.name, configs.optimizer.learning_rate, configs.train.num_epochs, configs.lr_fn.kwargs)
    optimizer_cfgs = dict(configs.optimizer, learning_rate=lr_fn)

    dummy_state = create_train_state(
        rng,
        data_shape,
        configs.half_precision,
        model,
        optimizer_cfgs=optimizer_cfgs,  
    )

    print("Looking in", ckpt_dir, "contains:", os.listdir(ckpt_dir))
    flax_state = restore_checkpoint(
        ckpt_dir,    
        target=dummy_state
    )
    use_vit = args.model_type in ["CifarViTSmall16", "CifarViTBase16"]

    # Evaluate
    retain_ds, forget_ds, test_ds = load_and_split_dataset(
        dataset_name=args.dataset,
        batch_size=128,
        forget_size=100,
        seed=args.seed,
        use_vit=use_vit
    )
    retain_acc = evaluate_model(flax_state, model, retain_ds, args)
    test_acc = evaluate_model(flax_state, model, test_ds, args)
    
    print(f"Retain Accuracy: {100 * retain_acc:.2f}%")
    print(f"Test Accuracy: {100 * test_acc:.2f}%")

    forget_dist = compute_forget_confidence_distance(flax_state, model, forget_ds, args, args.num_classes)
    retain_dist = compute_forget_confidence_distance(flax_state, model, retain_ds, args, args.num_classes)
    test_dist = compute_forget_confidence_distance(flax_state, model, test_ds, args, args.num_classes)
    print(f"Avg. Forget Conf. Dist, Forget set: {forget_dist:.3f}")
    print(f"Avg. Forget Conf. Dist, Retain set: {retain_dist:.3f}")
    print(f"Avg. Forget Conf. Dist, Test set: {test_dist:.3f}")
    

    results = {
        'model_type': args.model_type,
        'dataset': args.dataset,
        'retain_accuracy': float(retain_acc),
        'test_accuracy': float(test_acc),
        'forget_conf_dist': float(forget_dist) if not np.isnan(forget_dist) else None,
        'retain_conf_dist': float(retain_dist) if not np.isnan(retain_dist) else None,
        'test_conf_dist': float(test_dist) if not np.isnan(test_dist) else None,
        'training_time': train_dur,
        'num_classes': args.num_classes,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'timestamp': datetime.datetime.now().isoformat()
    }

    os.makedirs(args.results_file_path, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{args.model_type}_{args.dataset}_{timestamp}.json"
    full_path = os.path.join(args.results_file_path, filename)

    import json
    with open(full_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {full_path}")

if __name__ == '__main__':
    print("PID:", os.getpid())

    # Parse config and workdir
    parser = argparse.ArgumentParser(description='LabelDP frontend: train, convert, load data')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--workdir', type=str, required=True)
    args = parser.parse_args()

    main(args.config, args.workdir)
    print('Pipeline complete.')


