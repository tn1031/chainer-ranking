import copy
import chainer
from chainer import reporter
from chainer.training import extensions


class SpearmanrEvaluator(extensions.Evaluator):

    def evaluate(self):
        iterator = self._iterators['main']
        compute_mu = self._targets['main'].compute_mu
        eval_func = self.eval_func

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter.DictSummary()

        pred_mu = []
        for batch in it:
            in_arrays = self.converter(batch, self.device)
            with chainer.function.no_backprop_mode():
                if isinstance(in_arrays, tuple):
                    _mu = compute_mu(*in_arrays)
                elif isinstance(in_arrays, dict):
                    _mu = compute_mu(**in_arrays)
                else:
                    _mu = compute_mu(in_arrays)
                pred_mu.append(_mu)

        pred_mu = chainer.functions.vstack(pred_mu)
        pred_mu = chainer.functions.squeeze(pred_mu).data
        cor, p = eval_func(pred_mu)
        summary.add({'spearmanr/cor': cor, 'spearmanr/p': p})
        return summary.compute_mean()
