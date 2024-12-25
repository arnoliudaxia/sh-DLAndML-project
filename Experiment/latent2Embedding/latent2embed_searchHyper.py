import uuid
import wandb
from My.Experiment.latent2Embedding.latent2embed import train as trainer
import argparse

sweep_config = {
    "name": "sweep_theMLPLatent2Embedding",
    "metric": {
        "name": "test_euclidean_distance",
        "goal": "minimize"
    },
    "method": "grid",
    "parameters": {
        "learning_rate": {
            "values": [1e-4, 1e-5]
        },
        'fc_layer_size': {
            'values': [1024, 2048, 3096]
        },
        'depthOfMLP': {
            'values': [1,2, 3]
        },
        'batch-size': {
            'values': [2048, 4096]
        },
        "lossType":{
            'values': ['MSE', 'ContrastiveLoss']
        }
    },
}
sweep_id = wandb.sweep(sweep_config, project = "EEG-project")
def train(): #Initialize a new wandb run
    unique_id = str(uuid.uuid4())  # 创建一个唯一的ID
    config={
        "pipeline-dataprocess": "ChanelWiseWhiteAndCharacterSplit",
        "pipeline-EEGEncoder": 'latent2Embedding',
        "loss": "ContrastiveLoss",
        "MyID": unique_id,
        }
    with wandb.init(config = config):
        wandb.config.update(config)
        args_string = f'--lr {wandb.config.learning_rate} --SaveModelPath My/Model/latent2Embedding/MLPAndContrastiveloss/{unique_id} --batchSize {wandb.config["batch-size"]} --mask 8 --UseWandb\
           -- LossType {wandb.config.lossType} --depthOfMLP {wandb.config.depthOfMLP} --fc_layer_size {wandb.config.fc_layer_size}'
        args_list = args_string.split()
        parser = argparse.ArgumentParser()
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--SaveModelPath', type=str, default=None) 
        parser.add_argument('--LossType', type=str, default="MSE", help="loss的类型, [MSE, ContrastiveLoss]") 
        parser.add_argument('--UseWandb', action='store_true') # 默认False
        parser.add_argument('--batchSize', type=int, default=2048)
        parser.add_argument('--depthOfMLP', type=int, default=2, help="MLP的隐藏层深度")
        parser.add_argument('--fc_layer_size', type=int, default=2048, help="MLP的隐藏层宽度")
        parser.add_argument('--mask', type=int, nargs='+', help="Mask掉的subject（不用做训练）")
    
        args = parser.parse_args(args_list)
        trainer(args)

wandb.agent(sweep_id, train)