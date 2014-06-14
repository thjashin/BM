#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# @file: BM.py

from __future__ import division

import numpy as np


class BM:

    def __init__(self, n_input, n_hidden, n_output):
        """
        self.nodes: input | output | hidden
        """
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.N = n_input + n_hidden + n_output

        self.nodes = np.zeros((self.N))
        self.w = np.random.uniform(
            -np.sqrt(3.0 / self.N), np.sqrt(3.0 / self.N), (self.N, self.N))
        for i in xrange(self.w.shape[0]):
            self.w[i, i] = 0

    def train(self, X, Y, epoches=1000, alpha=0.1):
        """
        X: np array of shape: (num_of_sample, n_input), elements from {-1, +1}
        Y: np array of shape: (num_of_sample), elements from [0, n_output)
        """
        for ep in xrange(epoches):
            # random sample
            _id = np.random.randint(0, X.shape[0])

            self.rand_nodes()
            # fix input and output nodes
            self.nodes[:self.n_input] = X[_id]
            self.nodes[self.n_input:-self.n_hidden] = np.array(
                [1 if i == Y[_id] else -1 for i in xrange(self.n_output)])
            # anneal
            states1, T = self.anneal((self.n_input + self.n_output, self.N))

            self.rand_nodes()
            # fix input nodes
            self.nodes[:self.n_input] = X[_id]
            # anneal
            states2, T = self.anneal((self.n_input, self.N))

            for i in xrange(self.N):
                for j in xrange(self.N):
                    self.w[i, j] += alpha / T * \
                        (states1[i] * states1[j] - states2[i] * states2[j])

    def anneal(self, free_range):
        K = 1000
        T = 100
        c = 0.9
        for k in xrange(K):
            i = np.random.randint(free_range[0], free_range[1])
            l_i = np.sum(np.dot(self.w[i], self.nodes))
            self.nodes[i] = np.tanh(l_i / T)
            T *= c
        # print self.nodes
        return np.array([1 if i > 0 else -1 for i in self.nodes]), T

    def rand_nodes(self):
        for i in xrange(self.nodes.shape[0]):
            self.nodes[i] = 1 if np.random.rand() > 0.5 else -1

    def predict(self, testX):
        result = np.zeros((testX.shape[0], self.n_output))
        testY = np.zeros((testX.shape[0]))
        for i, sample in enumerate(testX):
            self.rand_nodes()
            # fix input nodes
            self.nodes[:self.n_input] = sample
            # anneal
            ret, T = self.anneal((self.n_input + self.n_output, self.N))
            result[i] = ret[self.n_input:-self.n_hidden]
            testY[i] = np.argmax(result[i])
        # print 'result:'
        # print result
        return testY.astype('int')


if __name__ == "__main__":
    def hw7():
        with open('data.txt', 'r') as f:
            raw = f.read()
            arr = raw.split(',')
            w = [i.strip().split('\n') for i in arr]
            for i in xrange(len(w)):
                for j in xrange(len(w[i])):
                    w[i][j] = [1 if k == '+' else -1 for k in w[i][j]]
        # for i in w:
        #     for j in i:
        #         print j
        #     print

        [w1, w2, w3, testX] = map(np.array, w)

        # question a
        Y1 = np.array([0 for i in w2])
        Y2 = np.array([1 for i in w3])
        data1, data2 = np.vstack((w2.T, Y1)).T, np.vstack((w3.T, Y2)).T
        data = np.vstack((data1, data2))
        # np.random.shuffle(data)
        X = data[:, :-1]
        Y = data[:, -1]
        # print X
        print 'Y:'
        print Y
        bm = BM(X.shape[1], 4, 2)
        bm.train(X, Y)

        # print testX
        print bm.predict(X)

    hw7()
