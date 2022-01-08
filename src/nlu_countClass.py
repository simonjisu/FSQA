import yaml
import json
import argparse
import pandas as pd
from multiprocessing import freeze_support

from tqdm import tqdm
from pathlib import Path
from nlu_utils import NLUDataModule
from collections import Counter, defaultdict
from nlu_utils import NLUTokenizer

def count(data_module, all_counters, prefix='train'):
        if prefix == 'train':
            dataset = data_module.create_dataset(data_module.train_data)
        elif prefix == 'valid':
            dataset = data_module.create_dataset(data_module.valid_data)
        else:
            dataset = data_module.create_dataset(data_module.test_data)
        for i in tqdm(range(len(dataset)), total=len(dataset), desc=f'Counting {prefix}'):
            x = dataset[i]
            ents = list(filter(lambda x: x[2] in ['IS', 'BS'], dataset.data[i]['entities']))
            for si, ei, _ in ents:
                acc = dataset.data[i]['text'][si:ei]
                all_counters[prefix]['account'].update([acc])
            all_counters[prefix]['intent'].update([x['intent']])
            all_counters[prefix]['tags'].update(x['tags'])

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

    all_counters = defaultdict(dict)
    for prefix in ['train', 'valid', 'test']:
        all_counters[prefix]['tags'] = Counter()
        all_counters[prefix]['intent'] = Counter()
        all_counters[prefix]['account'] = Counter()
        count(data_module, all_counters, prefix=prefix)
    print(all_counters)
    name = args.file.rstrip('.yml').split('_', 1)[1]
    with (data_path / f'all_data_count_{name}.json').open('w', encoding='utf-8') as file:
        json.dump(all_counters, file)