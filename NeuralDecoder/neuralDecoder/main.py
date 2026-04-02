import os

import hydra
import wandb
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig

from neuralDecoder.neuralSequenceDecoder import NeuralSequenceDecoder

@hydra.main(config_path='configs', config_name='config')
def app(config):
                                     

                                                                                                         
    if 'gpuNumber' in config:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        print(f"Setting CUDA_VISIBLE_DEVICES to {config['gpuNumber']}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpuNumber'])

    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
                                                                            
                                                                                    
                                                                                   
                                                                                   
            for gpu in gpus:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=20000)]
                )
            print("Set fixed GPU memory limit to 20000 MB.")
        except RuntimeError as e:
            print(f"GPU memory config exception: {e}")

    if config.get('mixedPrecision', False):
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision enabled (float16 activations, float32 weights).")

    if 'Slurm' in HydraConfig.get().launcher._target_:
                                                                                                    
        config.outputDir = './'
    print(f'Output dir {config.outputDir}')
    os.makedirs(config.outputDir, exist_ok=True)

    if 'wandb' in config and config.wandb.enabled:
        run = wandb.init(**config.wandb.setup,
                         config=OmegaConf.to_container(config, resolve=True),
                         sync_tensorboard=True,
                         resume=True)

                              
    nsd = NeuralSequenceDecoder(args=config)

                   
    if config['mode'] == 'train':
        cer = nsd.train()
        return cer
    elif config['mode'] == 'inference':
        nsd.inference()

if __name__ == "__main__":
    app()
