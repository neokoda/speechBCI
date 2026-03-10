import os

import hydra
import wandb
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig

from neuralDecoder.neuralSequenceDecoder import NeuralSequenceDecoder

@hydra.main(config_path='configs', config_name='config')
def app(config):
    #print(OmegaConf.to_yaml(config))

    #set the visible device to the gpu specified in 'args' (otherwise tensorflow will steal all the GPUs)
    if 'gpuNumber' in config:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        print(f"Setting CUDA_VISIBLE_DEVICES to {config['gpuNumber']}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpuNumber'])

    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Enabled GPU memory growth.")
        except RuntimeError as e:
            print(f"Memory growth exception: {e}")

    if 'Slurm' in HydraConfig.get().launcher._target_:
        # TF train saver doesn't support file name with '[' or ']'. So we'll use relative path here.
        config.outputDir = './'
    print(f'Output dir {config.outputDir}')
    os.makedirs(config.outputDir, exist_ok=True)

    if 'wandb' in config and config.wandb.enabled:
        run = wandb.init(**config.wandb.setup,
                         config=OmegaConf.to_container(config, resolve=True),
                         sync_tensorboard=True,
                         resume=True)

    #instantiate the RNN model
    nsd = NeuralSequenceDecoder(args=config)

    #train or infer
    if config['mode'] == 'train':
        cer = nsd.train()
        return cer
    elif config['mode'] == 'inference':
        nsd.inference()

if __name__ == "__main__":
    app()
