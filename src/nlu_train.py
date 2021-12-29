import yaml
import pickle
from multiprocessing import freeze_support
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

from pathlib import Path
from nlu_models import NLUModel
from nlu_utils import NLUDataModule


if __name__ == '__main__':
    freeze_support()
    main_path = Path().absolute().parent
    data_path = main_path / 'data'
    setting_path = main_path / 'setting_files'

    with (setting_path / 'train_settings.yml').open('r') as file:
        settings = yaml.load(file, Loader=yaml.FullLoader)

    data_module = NLUDataModule(
        data_path=data_path / settings['data_file'],
        ids_path=data_path / settings['ids_file'],
        batch_size=settings['batch_size'], 
        max_len=settings['max_len'],
        test_size=settings['test_size'],
        num_workers=settings['num_workers'],
        seed=settings['seed']
    )

    with (data_path / settings['ids_file']).open('rb') as file:
        ids = pickle.load(file)

    tags2id = ids['tags2id']
    intents2id = ids['intents2id']

    hparams = {
        'stage': 'train',
        'model_path': 'bert-base-uncased', 
        'intent_size': len(intents2id), 
        'tags_size': len(tags2id), 
        'max_len': settings['max_len'],
        'lr': settings['lr'],
    }

    model = NLUModel(**hparams)

    log_path = main_path / 'logs'
    checkpoint_path = main_path / 'checkpoints'

    logger = TensorBoardLogger(save_dir=str(log_path), name=settings['exp_name'])

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_path), 
        save_top_k=settings['save_top_k'],
        monitor='val_loss'
    )
    progress_callback = TQDMProgressBar(refresh_rate=settings['refresh_rate'])

    trainer = pl.Trainer(
        gpus=settings['n_gpus'], 
        max_epochs=settings['n_epochs'], 
        logger=logger, 
        num_sanity_val_steps=settings['num_sanity_val_steps'],
        callbacks=[checkpoint_callback, progress_callback]
    )
    trainer.fit(
        model, datamodule=data_module
    )


