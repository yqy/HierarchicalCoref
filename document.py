from collections import defaultdict
import numpy as np
import evaluation
from collections import Counter


class Document:
    def __init__(self, did, mentions, gold, mention_to_gold):
        self.did = did
        self.mentions = mentions
        self.gold = gold
        self.mention_to_gold = {m: tuple(g) for m, g in mention_to_gold.iteritems()}
        self.reset()

    def reset(self):
        self.clusters = []
        self.mention_to_cluster = {}
        self.rs = {}
        self.ps = {}
        self.ana_to_ant = {}
        self.ant_to_anas = {}
        for m in self.mentions:
            c = (m,)
            self.mention_to_cluster[m] = c
            self.clusters.append(c)
            self.rs[m] = 0
            self.ps[m] = 0
            self.ana_to_ant[m] = -1
            self.ant_to_anas[m] = []
        self.p_num = self.r_num = self.p_den = 0
        self.r_den = sum(len(g) for g in self.gold)

    def get_f1(self, beta=1):
        return evaluation.f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=beta)

    def update_b3(self, c, hypothetical=False):
        if len(c) == 1:
            self.p_den -= 1
            self.p_num -= self.ps[c[0]]
            self.r_num -= self.rs[c[0]]
            self.ps[c[0]] = 0
            self.rs[c[0]] = 0
        else:
            intersect_counts = Counter()
            for m in c:
                if m in self.mention_to_gold:
                    intersect_counts[self.mention_to_gold[m]] += 1
            for m in c:
                if m in self.mention_to_gold:
                    self.p_num -= self.ps[m]
                    self.r_num -= self.rs[m]

                    g = self.mention_to_gold[m]
                    ic = intersect_counts[g]
                    self.p_num += ic / float(len(c))
                    self.r_num += ic / float(len(g))
                    if not hypothetical:
                        self.ps[m] = ic / float(len(c))
                        self.rs[m] = ic / float(len(g))

    def link(self, m1, m2, hypothetical=False, beta=1):
        if m1 == -1:
            return self.get_f1(beta=beta) if hypothetical else None

        c1, c2 = self.mention_to_cluster[m1], self.mention_to_cluster[m2]
        assert c1 != c2
        new_c = c1 + c2
        p_num, r_num, p_den, r_den = self.p_num, self.r_num, self.p_den, self.r_den

        if len(c1) == 1:
            self.p_den += 1
        if len(c2) == 1:
            self.p_den += 1
        self.update_b3(new_c, hypothetical=hypothetical)

        if hypothetical:
            f1 = evaluation.f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=beta)
            self.p_num, self.r_num, self.p_den, self.r_den = p_num, r_num, p_den, r_den
            return f1
        else:
            self.ana_to_ant[m2] = m1
            self.ant_to_anas[m1].append(m2)
            self.clusters.remove(c1)
            self.clusters.remove(c2)
            self.clusters.append(new_c)
            for m in new_c:
                self.mention_to_cluster[m] = new_c

    def unlink(self, m):
        old_ant = self.ana_to_ant[m]
        if old_ant != -1:
            self.ana_to_ant[m] = -1
            self.ant_to_anas[old_ant].remove(m)

            old_c = self.mention_to_cluster[m]
            c1 = [m]
            frontier = self.ant_to_anas[m][:]
            while len(frontier) > 0:
                m = frontier.pop()
                c1.append(m)
                frontier += self.ant_to_anas[m]
            c1 = tuple(c1)
            c2 = tuple(m for m in old_c if m not in c1)

            self.update_b3(c1)
            self.update_b3(c2)

            self.clusters.remove(old_c)
            self.clusters.append(c1)
            self.clusters.append(c2)
            for m in c1:
                self.mention_to_cluster[m] = c1
            for m in c2:
                self.mention_to_cluster[m] = c2
def update_doc(doc, X, scores):
    s = scores
    starts_ends = zip(X['starts'], X['ends'])
    for (start, end) in starts_ends:
        action_scores = s[start:end]
        link = np.argmax(action_scores)
        m1, m2 = X['ids'][start + link]
        doc.link(m1, m2)
