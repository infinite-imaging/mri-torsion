import os
import json
from collections import Counter
from datetime import date
from tqdm import tqdm
import numpy as np

def calc_age(content):
    if 'PatientAge' in content:
        return int(content['PatientAge'])
    birth_date = date(*[int(x) for x in content.get('PatientBirthDate').split('-')])
    creation_date = date(*[int(x) for x in content.get('InstanceCreationDate', content.get('AcquisitionDate')).split('-')])

    if (creation_date - birth_date).days // 365 == 0:
        breakpoint()
    return (creation_date - birth_date).days // 365

def find_subdirs(root):
    return [os.path.join(root, x) for x in os.listdir(root) if os.path.isdir(os.path.join(root, x))]

split_root_path = ''

general_root_paths = []

splits = ['train', 'val', 'test']

ages = {k: [] for k in splits}
sexes = {k: [] for k in splits}
patients = {k: [] for k in splits}

test_pats = list(os.listdir(''))

for split in tqdm(splits):
    for pat in tqdm([x for x in os.listdir(os.path.join(split_root_path, split)) if os.path.isdir(os.path.join(split_root_path, split, x))]):
        if split == 'test' and (pat.rsplit('_',1)[1] not in test_pats or pat.rsplit('_', 1)[1] == '35'): 
            continue
        curr_age, curr_sex = None, None
        for root_path in general_root_paths:
            pat_dir = os.path.join(root_path, pat)

            if not os.path.isdir(pat_dir):
                continue

            for study in find_subdirs(pat_dir):
                for series in find_subdirs(study):
                    for inst in find_subdirs(series):

                        with open(os.path.join(inst, 'meta_data.json'), 'r') as f:
                            content = json.load(f)

                        if curr_age is None:
                            curr_age = calc_age(content)

                        if curr_age == 0:
                            breakpoint()
                        
                        if curr_sex is None:
                            curr_sex = content['PatientSex']

                        break

                    if curr_age is not None and curr_sex is not None:
                        break
                if curr_age is not None and curr_sex is not None:
                        break
            if curr_age is not None and curr_sex is not None:
                        break

        if (curr_sex is None or curr_age is None):
            raise RuntimeError(pat)


        ages[split].append(curr_age)
        sexes[split].append(curr_sex)
        patients[split].append(pat)

ages['total'] = ages['train'] + ages['val'] + ages['test']
sexes['total'] = sexes['train'] + sexes['val'] + sexes['test']
for split in splits + ['total']:
    print(split, Counter(sexes[split]))
    print(split, f'Mean Age: {np.mean(ages[split])}, Std Age: {np.std(ages[split])}, Min Age: {np.min(ages[split])}, Max Age: {np.max(ages[split])}')

print('')

