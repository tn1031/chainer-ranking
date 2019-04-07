import chainer
import chainer.distributions as D
import chainer.functions as F
import chainer.links as L
from chainer import reporter


def _xent_loss(prob, y):
    return F.softmax_cross_entropy(prob, y)

def _nll_loss(prob, y):
    return -F.mean(D.Independent(D.Categorical(prob)).log_prob(y))

class MultinomialNormal(chainer.Chain):
    def __init__(self, n_players, beta=1., draw_prob=.1111, eps=1e-5,
                 loss_type='xent'):
        super(MultinomialNormal, self).__init__()
        self.n_players = n_players
        self.beta = beta
        self.draw_prob = draw_prob
        self.eps = eps
        if loss_type == 'xent':
            self.compute_loss = _xent_loss
        else:
            self.compute_loss = _nll_loss

        with self.init_scope():
            self.lookup = L.EmbedID(n_players, 2)

    def __call__(self, i, j, y):
        xp = self.xp
        mu_i, ln_sigma_i = F.split_axis(self.lookup(i), 2, 1)
        mu_j, ln_sigma_j = F.split_axis(self.lookup(j), 2, 1)
        skill_i = D.Independent(D.Normal(loc=mu_i, log_scale=ln_sigma_i)).sample()
        skill_j = D.Independent(D.Normal(loc=mu_j, log_scale=ln_sigma_j)).sample()
        diff_ij = D.Independent(
            D.Normal(loc=skill_i - skill_j,
                     log_scale=F.log(2 * self.beta * xp.ones(skill_i.shape, dtype=xp.float32)))).sample()
        pos = F.sigmoid(diff_ij)
        ratio = F.concat([self.draw_prob * xp.ones(pos.shape, dtype=xp.float32), pos, 1-pos])
        eps = self.eps * xp.ones(3, dtype=xp.float32)
        ratio += F.broadcast_to(eps, (ratio.shape[0], 3))
        prob = ratio / F.sum(ratio, axis=1, keepdims=True)

        loss = self.compute_loss(prob, y)
        reporter.report({'loss': loss}, self)
        return loss

    def compute_mu(self, i, x):
        return F.split_axis(self.lookup(i), 2, 1)[0]


class DeterministicCNN(chainer.Chain):
    def __init__(self, n_players, draw_prob=.1111, eps=1e-5,
                 loss_type='xent'):
        super(DeterministicCNN, self).__init__()
        self.n_players = n_players
        self.draw_prob = draw_prob
        self.eps = eps
        if loss_type == 'xent':
            self.compute_loss = _xent_loss
        else:
            self.compute_loss = _nll_loss

        with self.init_scope():
            self.cnn = TinyCNN(1)

    def __call__(self, xi, xj, y):
        xp = self.xp
        skill_i = self.cnn(xi)
        skill_j = self.cnn(xj)
        diff_ij = skill_i - skill_j
        pos = F.sigmoid(diff_ij)
        ratio = F.concat([self.draw_prob * xp.ones(pos.shape, dtype=xp.float32), pos, 1-pos])
        eps = self.eps * xp.ones(3, dtype=xp.float32)
        ratio += F.broadcast_to(eps, (ratio.shape[0], 3))
        prob = ratio / F.sum(ratio, axis=1, keepdims=True)

        loss = self.compute_loss(prob, y)
        reporter.report({'loss': loss}, self)
        return loss

    def compute_mu(self, i, x):
        return self.cnn(x)


class MultinomialNormalCNN(chainer.Chain):
    def __init__(self, n_players, beta=1., draw_prob=.1111, eps=1e-5,
                 loss_type='nll'):
        super(MultinomialNormalCNN, self).__init__()
        self.n_players = n_players
        self.beta = beta
        self.draw_prob = draw_prob
        self.eps = eps
        if loss_type == 'xent':
            self.compute_loss = _xent_loss
        else:
            self.compute_loss = _nll_loss

        with self.init_scope():
            self.cnn = TinyCNN(2)

    def __call__(self, xi, xj, y):
        xp = self.xp
        mu_i, ln_sigma_i = F.split_axis(self.cnn(xi), 2, 1)
        mu_j, ln_sigma_j = F.split_axis(self.cnn(xj), 2, 1)
        skill_i = D.Independent(D.Normal(loc=mu_i, log_scale=ln_sigma_i)).sample()
        skill_j = D.Independent(D.Normal(loc=mu_j, log_scale=ln_sigma_j)).sample()
        diff_ij = D.Independent(
            D.Normal(loc=skill_i - skill_j,
                     log_scale=F.log(2 * self.beta * xp.ones(skill_i.shape, dtype=xp.float32)))).sample()
        pos = F.sigmoid(diff_ij)
        ratio = F.concat([self.draw_prob * xp.ones(pos.shape, dtype=xp.float32), pos, 1-pos])
        eps = self.eps * xp.ones(3, dtype=xp.float32)
        ratio += F.broadcast_to(eps, (ratio.shape[0], 3))
        prob = ratio / F.sum(ratio, axis=1, keepdims=True)
        
        loss = self.compute_loss(prob, y)
        reporter.report({'loss': loss}, self)
        return loss

    def compute_mu(self, i, x):
        return F.split_axis(self.cnn(x), 2, 1)[0]


class TinyCNN(chainer.Chain):
    def __init__(self, n_out):
        super(TinyCNN, self).__init__()
        with self.init_scope():
            self.conv1 = BasicConv2d(3, 32, ksize=3, stride=2, pad=1)
            self.conv2 = BasicConv2d(32, 64, ksize=3, stride=2, pad=1)
            self.conv3 = BasicConv2d(64, 128, ksize=3, stride=2, pad=1)
            self.fc = L.Linear(512, n_out)

    def __call__(self, x):
        x = F.max_pooling_2d(self.conv1(x), 2, 2)
        x = F.max_pooling_2d(self.conv2(x), 2, 2)
        x = F.max_pooling_2d(self.conv3(x), 2, 2)
        x = self.fc(x)
        return x


class BasicConv2d(chainer.Chain):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(in_channels, out_channels, nobias=True,
                    **kwargs)
            self.bn = L.BatchNormalization(out_channels, decay=0.9997, eps=0.001)

    def __call__(self, x):
        h = self.conv(x)
        h = self.bn(h)
        return F.relu(h)
