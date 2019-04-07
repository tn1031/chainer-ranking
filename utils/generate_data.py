import argparse
import os
import numpy as np
import scipy.stats as ss
from scipy import io as spio
from scipy.special import expit as sigmoid
import trueskill
from trueskill import Rating, quality_1vs1, rate_1vs1


class HipsterWars:
    def __init__(self, n_players):
        self.n_players = n_players
        self.players = [Rating() for _ in range(n_players)]
        self.played_cnt = np.zeros(n_players)
        trueskill.backends.choose_backend('scipy')

    def get_player(self, i):
        return self.players[i].mu, self.players[i].sigma, self.played_cnt[i]

    def append_newcomers(self, n_newcomers):
        newcomers = [Rating() for _ in range(n_newcomers)]
        self.players.extend(newcomers)
        self.played_cnt = np.append(self.played_cnt, [0] * n_newcomers)
        self.n_players += n_newcomers

    def append_player(self, mu, sigma, cnt):
        player = Rating(mu=mu, sigma=sigma)
        self.players.append(player)
        self.played_cnt = np.append(self.played_cnt, cnt)
        self.n_players += 1

    def insert_player(self, i, mu, sigma, cnt):
        assert i < len(self.players)
        r = Rating(mu=mu, sigma=sigma)
        self.players[i] = r
        self.played_cnt[i] = cnt

    def _compute_draw_prob(self, i):
        mu_var = np.array([[p.mu, p.sigma**2] for p in self.players])
        beta = trueskill.BETA ** 2  # using default env
        z = 1. / (2*beta + mu_var[i,1] + mu_var[:,1])
        diff_mu = (mu_var[i,0] - mu_var[:,0]) ** 2
        _prob = np.sqrt(z) * np.exp(-0.5 * diff_mu * z)
        return _prob

    def update_and_matchmaking(self, winner_id, loser_id, is_draw,
                               is_deterministic=True):
        self.update_rating(winner_id, loser_id, is_draw)
        return self.get_next_pair(is_deterministic)

    def get_next_pair(self, is_deterministic=False):
        least_played_id = np.argmin(self.played_cnt)
        player_i = self.players[least_played_id]

        prob = self._compute_draw_prob(least_played_id)
        prob /= np.sum(prob)

        if is_deterministic:
            # get the second best because the largest is oneself.
            competitor_id = np.argsort(prob)[-2]
            if competitor_id == least_played_id:
                # the largest is equal to the second largest.
                competitor_id = np.argsort(prob)[-1]
        else:
            # get a competitor probabilistically.
            competitor_id = np.random.choice(self.n_players, p=prob)
            while competitor_id == least_played_id:
                competitor_id = np.random.choice(self.n_players, p=prob)

        self.played_cnt[least_played_id] += 1
        self.played_cnt[competitor_id] += 1
        return least_played_id, competitor_id

    def update_rating(self, winner_id, loser_id, is_draw):
        winner = self.players[winner_id]
        loser = self.players[loser_id]
        new_rating_winner, new_rating_loser = rate_1vs1(winner, loser, is_draw)
        self.players[winner_id] = new_rating_winner
        self.players[loser_id] = new_rating_loser


def compare(i, j, mu, sigma):
    if np.random.choice([0, 1], p=[0.9, 0.1]):
        # draw
        return 0
    z_i = np.random.normal(mu[i], sigma[i])
    z_j = np.random.normal(mu[j], sigma[j])
    p = sigmoid(z_i - z_j)
    if np.random.choice([0, 1], p=[1-p, p]):
        return 1
    else:
        return 2


def generate_data(category, n):
    data = np.load('./data/{}.npz'.format(category))

    game = HipsterWars(len(data['mu']))
    for i, (m, s) in enumerate(zip(data['mu'], data['sigma'])):
        game.insert_player(i, m, s, 0)

    results = []
    for t in range(n):
        if t == 0:
            i, j = game.get_next_pair(False)
        res = compare(i, j, data['mu'], data['sigma'])
        results.append((i, j, res))
        if res == 0:
            i, j = game.update_and_matchmaking(i, j, True, False)
        elif res == 1:
            i, j = game.update_and_matchmaking(i, j, False, False)
        elif res == 2:
            i, j = game.update_and_matchmaking(j, i, False, False)
        else:
            raise

    pred_mu = np.array([p.mu for p in game.players])
    print(ss.spearmanr(data['mu'], pred_mu))
    with open('./data/results_{}.tsv'.format(category), 'w') as f:
        for r in results:
            f.write('{}\t{}\t{}\n'.format(*r))



def load_matfile(category):
    matdata = spio.loadmat('./data/hipsterwars_Jan_2014.mat', squeeze_me=True)
    data = [matdata['samples'][i] for i in range(len(matdata['samples'])) if matdata['samples'][i][1] == category]
    mu = np.array([e[3] for e in data])
    sigma = np.array([np.sqrt(e[4]) for e in data])
    imgarr = np.array([e[6] for e in data])
    print(len(mu))
    np.savez('./data/{}.npz'.format(category), mu=mu, sigma=sigma, imgarr=imgarr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('category', choices=['Pinup', 'Goth', 'Hipster', 'Bohemian', 'Preppy'])
    parser.add_argument('--num_transactions', '-n', type=int, default=5000)
    args = parser.parse_args()

    if not os.path.isfile('./data/{}.npz'.format(args.category)):
        load_matfile(args.category)
    generate_data(args.category, args.num_transactions)
