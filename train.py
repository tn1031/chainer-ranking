import argparse
import json
import numpy as np
import os
import pickle
import scipy.stats as ss

import chainer
from chainer import training
from chainer.training import extensions
from chainer.dataset import concat_examples

from dataset import HipsterWarsDataset, HipsterWarsImageDataset, SimpleIndexImageDataset
from evaluator import SpearmanrEvaluator
from model import MultinomialNormal, MultinomialNormalCNN, DeterministicCNN


def main(args):
    #chainer.backends.cuda.get_device(0).use()

    # dataset
    data = np.load('./data/Bohemian.npz')
    results = open('./data/results_Bohemian.tsv').read().strip().split('\n')
    results = list(map(lambda line: line.split('\t'), results))
    if args.model == 'mn':
        train_data = HipsterWarsDataset(results[:4500])
        valid_data = HipsterWarsDataset(results[4500:])
    else:
        train_data = HipsterWarsImageDataset(results[:4500], data['imgarr'], True)
        valid_data = HipsterWarsImageDataset(results[4500:], data['imgarr'], False)
    all_image_data = SimpleIndexImageDataset(data['imgarr'])

    # iterator
    train_iter = chainer.iterators.SerialIterator(train_data,
            batch_size=args.batchsize)
    valid_iter = chainer.iterators.SerialIterator(valid_data,
            batch_size=args.batchsize, repeat=False, shuffle=False)
    eval_iter = chainer.iterators.SerialIterator(all_image_data,
            batch_size=args.batchsize, repeat=False, shuffle=False)

    # model
    if args.model == 'mn':
        model = MultinomialNormal(len(data['imgarr']), 3.)
    elif args.model == 'mncnn':
        model = MultinomialNormalCNN(len(data['imgarr']), 3.)
    elif args.model == 'dcnn':
        model = DeterministicCNN(len(data['imgarr']))
    else:
        raise ValueError('unknown model.')

    # optimizer
    optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.5)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.00004))
    optimizer.add_hook(chainer.optimizer.GradientClipping(5))

    # updater
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=-1)

    out = os.getenv('SM_OUTPUT_DIR', 'result')
    if not os.path.isdir(out):
        os.mkdir(out)
    with open(os.path.join(out, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    val_interval = (1, 'epoch')
    log_interval = (100, 'iteration')
    snapshot_interval = (1, 'epoch')
    print_metrics = ['main/loss', 'spearmanr/cor', 'spearmanr/p']

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=out)

    trainer.extend(
            extensions.Evaluator(valid_iter, model, device=-1),
            trigger=val_interval)
    trainer.extend(
            SpearmanrEvaluator(eval_iter, model, device=-1,
            eval_func=lambda x: ss.spearmanr(data['mu'], x)),
            trigger=val_interval)

    trainer.extend(extensions.snapshot(), trigger=snapshot_interval)
    trainer.extend(
            extensions.snapshot_object(model, 'model_epoch_{.updater.epoch}'),
            trigger=snapshot_interval)
    trainer.extend(
            extensions.snapshot_object(optimizer, 'opt_epoch_{.updater.epoch}'),
            trigger=snapshot_interval)

    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport(
        ['iteration', 'epoch', 'elapsed_time', 'lr'] + print_metrics),
        trigger=log_interval)

    trainer.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=32)
    parser.add_argument('--epoch', '-e', type=int, default=10)
    parser.add_argument('--lr', '-l', type=float, default=0.05)
    parser.add_argument('--model', '-m', choices=['mn', 'mncnn', 'dcnn'], default='mn')
    args = parser.parse_args()

    main(args)