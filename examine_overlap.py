### NEW file

import numpy as np
import pickle
import os

fname = lambda dataset, type, pattern_id: (
    '/n/shieber_lab/Lab/users/ssakenis/pet/'
    f'output_few_shot_{dataset}_10_sorted_balanced_512_{type}_2/'
    f'p{pattern_id}-i0/agreement.pickle'
)

agreements = {}
for type in ['beg', 'mid', 'end']:
    # print(f'\n******************* Type: {type} *******************')
    agreements[type] = {}
    for dataset in ['yelp', 'agnews', 'yahoo', 'mnli']:
        # print(f'\nDataset: {dataset}')
        agreements[type][dataset] = {}
        pattern_ids = ([0, 1, 2, 3] if dataset in ['yelp', 'mnli'] else
                       [0, 1, 2, 3, 4, 5])
        for pattern_id in pattern_ids:
            agreements[type][dataset][pattern_id] = None
            acc = '[not found]'
            if os.path.isfile(fname(dataset, type, pattern_id)):
                with open(fname(dataset, type, pattern_id), 'rb') as f:
                    agreement = pickle.load(f)
                    agreements[type][dataset][pattern_id] = agreement
                    acc = agreement.mean()
            # print(f'pattern id: {pattern_id}, accuracy: {acc}')

# print('\n*************************************\n')

overlap = lambda agrs: [set(a[i] for a in agrs if a is not None) == {True}
                        for i in range(len(agrs[0]))]

# print('Overlap across patterns:')
# for dataset in ['yelp', 'agnews', 'yahoo', 'mnli']:
#     list_of_agreements = list(agreements['end'][dataset].values())
#     print(f'{dataset}: {np.mean(overlap(list_of_agreements))}')

print('Overlap across placements:')
for dataset in ['yelp', 'agnews', 'yahoo', 'mnli']:
    list_of_agreements = [
        agreements['beg'][dataset][3],
        agreements['mid'][dataset][3],
        agreements['end'][dataset][3],
    ]
    print(f'{dataset}: {np.mean(overlap(list_of_agreements))}')
