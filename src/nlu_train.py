import yaml
import json
import argparse
from multiprocessing import freeze_support
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, LearningRateMonitor

from pathlib import Path
from nlu_models import NLUModel
from nlu_utils import NLUDataModule

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='settings_file_name')
    parser.add_argument('-f', '--file', metavar='N', type=str, default='train_settings.yml',
                        help='settings_file name in the setting_files folder')
    args = parser.parse_args()

    # freeze_support()
    main_path = Path().absolute().parent
    data_path = main_path / 'data'
    setting_path = main_path / 'setting_files'

    with (setting_path / args.file).open('r') as file:
        settings = yaml.load(file, Loader=yaml.FullLoader)

    data_module_settings = settings['data_module']
    model_settings = settings['model']
    trainer_settings = settings['trainer']

    data_module = NLUDataModule(
        train_path=data_path / data_module_settings['train_file'], 
        valid_path=data_path / data_module_settings['valid_file'],
        test_path=data_path / data_module_settings['test_file'],
        labels_path=data_path / data_module_settings['labels_file'],
        batch_size=data_module_settings['batch_size'], 
        max_len=data_module_settings['max_len'],
        num_workers=data_module_settings['num_workers'],
        seed=settings['seed']
    )

    # if model_settings.get('weight_file') is not None:
    #     with (data_path / model_settings['weight_file']).open('r', encoding='utf-8') as file:
    #         weight_dict = json.load(file)
    # else:
    #     weight_dict = None

    hparams = {
        'stage': model_settings['stage'],
        'model_path': model_settings['model_path'], 
        'intent_size': len(data_module.intents2id), 
        'tags_size': len(data_module.tags2id), 
        'lr': model_settings['lr'],
        'weight_decay_rate': model_settings['weight_decay_rate'],
        'loss_type': model_settings['loss_type'],
        'multigpu': True if trainer_settings['n_gpus'] > 1 else False,
        # 'weight_dict': weight_dict
    }
    for k, v in model_settings['schedular'].items():
        hparams[f'schedular_{k}'] = v
    if model_settings['loss_type'] == 'focal':
        for k, v in model_settings['focal'].items():
            hparams[f'focal_{k}'] = v

    model = NLUModel(**hparams)

    log_path = main_path / 'logs'
    
    # checkpoint_path = main_path / 'checkpoints' / settings['exp_name']
    
    logger = TensorBoardLogger(
        save_dir=str(log_path), 
        name=settings['exp_name'],
        version=args.file.rstrip('.yml').split('_', 1)[-1]
    )
    checkpoint_dir = Path(logger.experiment.log_dir) / "checkpoints"
    # filepath = checkpoint_dir
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir), 
        save_top_k=trainer_settings['save_top_k'],
        monitor='val_loss',
        filename='{epoch}-{step}-{val_loss:.2f}'
    )
    progress_callback = TQDMProgressBar(refresh_rate=trainer_settings['refresh_rate'])
    lr_callback = LearningRateMonitor('step')

    if trainer_settings['deterministic']:
        seed_everything(seed=settings['seed'], workers=True)
    deterministic = trainer_settings['deterministic']
    trainer = pl.Trainer(
        gpus=trainer_settings['n_gpus'], 
        max_epochs=trainer_settings['n_epochs'], 
        logger=logger, 
        num_sanity_val_steps=trainer_settings['num_sanity_val_steps'],
        callbacks=[checkpoint_callback, progress_callback, lr_callback],
        log_every_n_steps=trainer_settings['log_every_n_steps'],
        strategy=trainer_settings['strategy'],  #  dp | ddp | ddp2 | ddp_spawn
        deterministic=deterministic,
        gradient_clip_val=trainer_settings['gradient_clip_val']
    )
    trainer.fit(
        model, datamodule=data_module
    )
    test_results = trainer.test(ckpt_path='best', datamodule=data_module)
    with Path(logger.experiment.log_dir).open('w', encoding='utf-8') as file:
        json.dump(test_results, file)
