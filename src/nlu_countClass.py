import yaml
import json
import argparse
from multiprocessing import freeze_support

from tqdm import tqdm
from pathlib import Path
from nlu_utils import NLUDataModule
from collections import Counter

def count(data_module, tags_counter, intent_counter, prefix='train'):
        
        if prefix == 'train':
            dataset = data_module.create_dataset(data_module.train_data)
        elif prefix == 'valid':
            dataset = data_module.create_dataset(data_module.valid_data)
        else:
            dataset = data_module.create_dataset(data_module.test_data)
        for i in tqdm(range(len(dataset)), total=len(dataset), desc=f'Counting {prefix}'):
            x = dataset[i]
            intent_counter.update(x['intent'])
            tags_counter.update([x['tags']])
        
        
        return tags_counter, intent_counter


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='settings_file_name')
    parser.add_argument('-f', '--file', metavar='N', type=str, default='train_settings.yml',
                        help='settings_file name in the setting_files folder')
    parser.add_argument('-c', '--complex_knowledge_tag', action='store_true',
                        help='complex_knowledge_tag')
    args = parser.parse_args()

    freeze_support()
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
    data_module.prepare_data()
    intent_counter = Counter()
    tags_counter = Counter()
    tags_counter, intent_counter = count(data_module, tags_counter, intent_counter, prefix='train')
    tags_counter, intent_counter = count(data_module, tags_counter, intent_counter, prefix='valid')
    # tags_counter, intent_counter = count(data_module, tags_counter, intent_counter, prefix='test')
    print(intent_counter)
    print(tags_counter)
    with (data_path / 'all_data_count.json').open('w', encoding='utf-8') as file:
        json.dump({
            'tags': {int(k): v for k, v in tags_counter.items()},
            'intent': {int(k): v for k, v in intent_counter.items()}
        }, file)