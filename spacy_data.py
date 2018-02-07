import numpy as np
import pandas as pd

from spacy_features import anygram_kernel

def get_embeddings(vocab):
    return vocab.vectors.data, vocab.vectors.find


def get_train_dev_test(
        filename='/data/claims/export_new.xlsx',
        test=None,
        balanced=False,
        split=0.8,
        content='sentence',
        target='label'
    ):

    df_original = pd.read_excel(filename)
    df_original = df_original[pd.notnull(df_original[content])]
    #df_original = df_original.reindex(np.random.permutation(df_original.index))

    # rename datasets
    df_original.loc[df_original['dataset_name'].isin(['SSGA from becky', 'SSGA_set1.2017-11-21']), 'dataset_name'] = 'SSGA'

    print 'Datasets:', df_original['dataset_name'].unique()

    test_df = df_original.loc[df_original['dataset_name'].isin(test)]
    print "test set", test_df.shape

    df = df_original.loc[~df_original['dataset_name'].isin(test)]
    print "train set", df.shape

    true_class = df[df[target] == 'HV']
    true_class = true_class.reindex(np.random.permutation(true_class.index))
    print "true class", true_class.shape

    false_class = df[df[target] == 'NOT']
    false_class = false_class.reindex(np.random.permutation(false_class.index))
    print "false class", false_class.shape

    if balanced:
        if len(true_class) > len(false_class):
            true_class = true_class.head(n=len(false_class))
            print "balanced true class", true_class.shape
        else:
            false_class = false_class.head(n=len(true_class))
            print "balanced false class", false_class.shape

    nf = pd.concat([true_class, false_class])
    nf = nf.reindex(np.random.permutation(nf.index))
    print 'total shape', nf.shape

    texts = []
    labels = []

    for index, row in nf.iterrows():
        texts.append(unicode(row[content]))
        labels.append(row[target])

    cats = [y == 'HV' for y in labels]

    if test_df.shape[0] > 0:
        test_texts = []
        test_labels = []

        for index, row in test_df.iterrows():
            test_texts.append(unicode(row[content]))
            test_labels.append(row[target])

        test_cats = [y == 'HV' for y in test_labels]

    else:
        test_split = int(len(texts) * 0.2)
        test_texts = texts[:test_split]
        test_cats = cats[:test_split]
        texts = texts[test_split:]
        cats = cats[test_split:]

    split = int(len(texts) * split)

    return texts[:split], cats[:split], texts[split:], cats[split:], test_texts, test_cats


def get_train_dev_for_embedding(
        filename='/data/claims/export_new.xlsx',
        test=None,
        balanced=False,
        split=0.8,
        content='sentence',
        target='label'
    ):

    df_original = pd.read_excel(filename)
    df_original = df_original[pd.notnull(df_original[content])]
    df_original = df_original.reindex(np.random.permutation(df_original.index))

    # rename datasets
    df_original.loc[df_original['dataset_name'].isin(['SSGA from becky', 'SSGA_set1.2017-11-21']), 'dataset_name'] = 'SSGA'

    print 'Datasets:', df_original['dataset_name'].unique()

    test_df = df_original.loc[df_original['dataset_name'].isin(test)]
    print "test set", test_df.shape

    df = df_original.loc[~df_original['dataset_name'].isin(test)]
    print "train set", df.shape

    true_class = df[df[target] == 'HV']
    true_class = true_class.reindex(np.random.permutation(true_class.index))
    print "true class", true_class.shape

    false_class = df[df[target] == 'NOT']
    false_class = false_class.reindex(np.random.permutation(false_class.index))
    print "false class", false_class.shape

    nf = pd.concat([true_class, false_class])
    nf = nf.reindex(np.random.permutation(nf.index))
    print 'total shape', nf.shape

    texts = []
    labels = []

    for index, row in nf.iterrows():
        texts.append(unicode(row[content]))
        labels.append(row[target])

    cats = [y == 'HV' for y in labels]

    if test_df.shape[0] > 0:
        test_texts = []
        test_labels = []

        for index, row in test_df.iterrows():
            test_texts.append(unicode(row[content]))
            test_labels.append(row[target])

        test_cats = [y == 'HV' for y in test_labels]

    else:
        test_split = int(len(texts) * 0.2)
        test_texts = texts[:test_split]
        test_cats = cats[:test_split]
        texts = texts[test_split:]
        cats = cats[test_split:]

    split = int(len(texts) * split)

    return texts[:split], cats[:split], texts[split:], cats[split:], test_texts, test_cats


def  get_train_dev_test_similarity(
        filename='/data/claims/export_plus_twitter.xlsx',
        test=None,
        balanced=False,
        split=0.8,
        content='sentence',
        target='label'
    ):

    df = pd.read_excel(filename)
    df = df[pd.notnull(df[content])]

    # rename datasets
    df.loc[df['dataset_name'].isin(['SSGA from becky', 'SSGA_set1.2017-11-21']), 'dataset_name'] = 'SSGA'

    print 'Datasets:', df['dataset_name'].unique()

    true_class = df[df[target] == 'HV'][:100]
    false_class = df[df[target] == 'NOT'][:100]

    all = list()
    similars = [(x, y) for x in true_class[content] for y in true_class[content] if y != x]
    not_similars = [(x, y) for x in true_class[content] for y in false_class[content][:len(true_class)]]

    all.extend(similars)
    all.extend(not_similars)

    labels = [1] * len(similars) + [0] * len(not_similars)

    # shuffle
    from random import shuffle
    c = list(zip(all, labels))
    shuffle(c)
    all, labels = zip(*c)

    split = int(len(all) * split)

    return all[:split], list(labels)[:split], all[split:], list(labels)[split:], all[split:], list(labels)[split:]






