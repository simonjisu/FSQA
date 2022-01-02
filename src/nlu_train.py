import yaml
import argparse
from multiprocessing import freeze_support
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

from pathlib import Path
from nlu_models import NLUModel
from nlu_utils import NLUDataModule

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='settings_file_name')
    parser.add_argument('-f', '--file', metavar='N', type=str, default='train_settings.yml',
                        help='settings_file name in the setting_files folder')
    args = parser.parse_args()

    freeze_support()
    main_path = Path().absolute().parent
    data_path = main_path / 'data'
    setting_path = main_path / 'setting_files'

    with (setting_path / args.file).open('r') as file:
        settings = yaml.load(file, Loader=yaml.FullLoader)

    data_module = NLUDataModule(
        train_path=data_path / settings['train_file'], 
        valid_path=data_path / settings['valid_file'],
        test_path=data_path / settings['test_file'],
        labels_path=data_path / settings['labels_file'],
        batch_size=settings['batch_size'], 
        max_len=settings['max_len'],
        num_workers=settings['num_workers'],
        seed=settings['seed']
    )

    hparams = {
        'stage': settings['stage'],
        'model_path': settings['model_path'], 
        'intent_size': len(data_module.intents2id), 
        'tags_size': len(data_module.tags2id), 
        'lr': settings['lr'],
        'multigpu': True if settings['n_gpus'] > 1 else False
    }

    model = NLUModel(**hparams)

    log_path = main_path / 'logs'
    checkpoint_path = main_path / 'checkpoints' / settings['exp_name']

    logger = TensorBoardLogger(save_dir=str(log_path), name=settings['exp_name'])

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_path), 
        save_top_k=settings['save_top_k'],
        monitor='val_loss'
    )
    progress_callback = TQDMProgressBar(refresh_rate=settings['refresh_rate'])

    seed_everything(seed=settings['seed'])
    trainer = pl.Trainer(
        gpus=settings['n_gpus'], 
        max_epochs=settings['n_epochs'], 
        logger=logger, 
        num_sanity_val_steps=settings['num_sanity_val_steps'],
        callbacks=[checkpoint_callback, progress_callback],
        deterministic=True,
    )
    trainer.fit(
        model, datamodule=data_module
    )
    trainer.test(ckpt_path='best', datamodule=data_module)

