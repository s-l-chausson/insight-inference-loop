# This software is a modified version of the bisect module found
# https://github.com/choderalab/thresholds/blob/master/thresholds/bisect.py

import os
import errno
import numpy as np
import math
import time
from IPython.display import clear_output
import matplotlib.pyplot as plt
import seaborn as sns
from fuzzywuzzy import fuzz, process

sns.set()

class BisectionAnnotator:

    def __init__(self, dataframe, start=0.0, stop=1.0, step=0.001, p=0.7,
                 early_termination_width=0, early_termination_weight=0.95,
                 verbose=2, graphs_out_path=None, level='claim',
                 timeout=90, filter_similar=True, similarity_threshold=90
                 ):

        if p <= 0.5:
            raise (ValueError('the probability of correct responses must be > 0.5'))

        self.df = dataframe
        self.start = start
        self.stop = stop
        self.step = step
        self.p = p
        self.q = 1 - p
        self.early_termination_width = early_termination_width
        self.early_termination_weight = early_termination_weight
        self.verbose = verbose
        self.graphs_out_path = graphs_out_path
        self.level = level
        self.timeout = timeout
        self.filter_similar = filter_similar
        self.similarity_threshold = similarity_threshold

        if not self.graphs_out_path is None:
            if not os.path.isdir(self.graphs_out_path):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.graphs_out_path)

        # initialize a uniform belief over the search interval
        self.x = np.arange(start, stop, step)
        self.x_labels = [round(e, 1) for e in np.arange(start, stop, 0.10)]
        ratio = int(0.10 / step)
        self.x_labels_loc = range(0, len(self.x), ratio)

    def round_down(self, score):
        multiplier = 1.0 / self.step
        return math.floor(score * multiplier) / multiplier

    def get_probability_masses(self, threshold):
        exp_f = np.exp(self.f)
        left_mass = np.sum(exp_f[self.x < threshold])
        right_mass = np.sum(exp_f[self.x >= threshold])
        return left_mass, right_mass

    def get_belief_interval(self):

        exp_f = np.exp(self.f)
        eps = 0.5 * (1 - self.early_termination_weight)
        eps = exp_f.sum() * eps

        try:
            left = self.x[exp_f.cumsum() < eps][-1]
        except IndexError:
            left = self.x[0]
        try:
            right = self.x[exp_f.cumsum() > (exp_f.sum() - eps)][0]
        except IndexError:
            right = self.x[-1]

        return left, right

    def get_median(self):
        exp_f = np.exp(self.f)
        alpha = exp_f.sum() * 0.5
        try:
            median_low = self.x[exp_f.cumsum() <= alpha][-1]
        except IndexError:
            return self.start
        try:
            median_high = self.x[::-1][exp_f[::-1].cumsum() < alpha][-1]
        except IndexError:
            return self.stop
        median_avg = (median_low + median_high) / 2
        return median_avg

    def find_best_match(self, median):

        texts = self.df['text'].to_list()
        scores = np.array(self.df['scores_for_annot'].to_list())
        scores_diff = np.abs(scores - median)
        sorted_indices = np.argsort(scores_diff)

        for i in sorted_indices:
            if texts[i] in self.done:
                continue
            if self.filter_similar:
                if len(self.done) > 0:
                    closest_match = process.extractOne(texts[i], self.done, scorer=fuzz.token_set_ratio)
                    if closest_match[1] > self.similarity_threshold:
                        if self.verbose >= 3:
                            print(
                                "Following example was picked but will be skipped because too similar to another annotated datapoint")
                            print(texts[i])
                            print()
                            print(closest_match)
                            input()
                        continue
            return scores[i], texts[i]

        print('MUST STOP, all the data has been annotated')
        return None, None

    def get_next_datapoint(self):
        median = self.get_median()
        if median == self.start or median == self.stop:
            return median, None, None, None, None
        best_match_score, best_match_text = self.find_best_match(median)
        if best_match_score is None:
            return median, None, None, None, None
        left_mass, right_mass = self.get_probability_masses(best_match_score)

        return median, best_match_score, best_match_text, left_mass, right_mass

    def get_most_probable_threshold(self):
        max_prob = max(self.f)
        max_indices = np.argwhere(self.f == max_prob)
        indices_base = max_indices - max_indices[0].flatten().tolist()
        max_values = self.x[max_indices].flatten().tolist()
        threshold = round(sum(max_values) / len(max_values), 4)
        return threshold

    def probabilistic_bisection(self):

        if not self.graphs_out_path is None:
            graph_path = os.path.join(self.graphs_out_path, 'bissection_graphs')
            if not os.path.exists(graph_path):
                os.mkdir(graph_path)

        round_nb = 0
        tbc = False
        early_stop = False

        while True:

            clear_output()

            round_nb += 1

            # query the oracle at median of previous belief pdf
            belief_interval = self.get_belief_interval()
            if (belief_interval[1] - belief_interval[0]) <= self.early_termination_width:
                break

            median, actual_median, text, left_mass, right_mass = self.get_next_datapoint()

            if actual_median is None:
                if median == self.start:
                    if self.verbose >= 1:
                        print('NEED TO STOP ANNOTATING! Reached lower bound of the threshold range.')
                        input()
                else:
                    if self.verbose >= 1:
                        print('NEED TO STOP ANNOTATING! Reached higher bound of the threshold range.')
                        input()
                break

            if self.verbose >= 2:
                print('===============================')
                print()
                print('ANNOT NB:', round_nb)
                print('MEDIAN: %.3f' % median)
                print()
                print('INTERVAL: %.3f' % belief_interval[0], '- %.3f' % belief_interval[1],
                      '\t==> WIDTH: %.3f' % (belief_interval[1] - belief_interval[0]))
                print('TOTAL =', sum(np.exp(self.f)))
                print('LEFT MASS = %.3f' % left_mass, '\tRIGHT MASS = %.3f' % right_mass,
                      '\tTOTAL = %.3f' % (left_mass + right_mass))
                print()

            if self.verbose >= 2:
                sns.barplot(x=self.x, y=np.exp(self.f), color="cornflowerblue")
                plt.xticks(self.x_labels_loc, self.x_labels)
                # plt.ylim(0, 1)
                plt.ylabel("Probability")
                plt.xlabel("Score")
                plt.title(self.trait_idx + ': ' + self.trait_text)
                if not self.graphs_out_path is None:
                    fig = plt.gcf()
                    fig.savefig(graph_path + '/round_' + str(round_nb) + '.png')
                plt.show()

            if max(left_mass, right_mass) > self.p:
                if self.verbose >= 1:
                    print('NEED TO STOP ANNOTATING! Data is too sparse')
                    z = input()
                break

            else:
                z = self.annotation_function(median, actual_median, text)

            if z == 'r':
                self.fs.pop()
                self.f = self.fs[-1].copy()
                last_item = self.done.pop()
                del self.cache[last_item]
                continue
            elif z == 's':
                tbc = True
                break
            elif z == 'q':
                tbc = True
                early_stop = True
                break

            self.texts_list.append(text)
            self.medians_list.append(median)
            self.scores_list.append(actual_median)
            self.annot_list.append(z)

            if z == False:
                if self.verbose >= 3:
                    scale_right = np.exp((np.log(self.p) - np.log(right_mass)))
                    scale_left = np.exp((np.log(self.q) - np.log(left_mass)))
                    print('Need to DECREASE on the left and INCREASE on the right')
                    print('\tMultiplying by %.5f on the left' % scale_left)
                    print('\tMultiplying by %.5f on the right' % scale_right)
                    if scale_right < 1.0 or scale_left > 1.0:
                        print('!!! WARNING: direction of scaling has switched')
                        input()
                self.f[self.x >= actual_median] += (np.log(self.p) - np.log(right_mass))
                self.f[self.x < actual_median] += (np.log(self.q) - np.log(left_mass))
            elif z == True:
                if self.verbose >= 3:
                    scale_right = np.exp((np.log(self.q) - np.log(right_mass)))
                    scale_left = np.exp((np.log(self.p) - np.log(left_mass)))
                    print('Need to INCREASE on the left and DECREASE on the right')
                    print('\tMultiplying by %.5f on the left' % scale_left)
                    print('\tMultiplying by %.5f on the right' % scale_right)
                    if scale_right > 1.0 or scale_left < 1.0:
                        print('!!! WARNING: direction of scaling has switched')
                        input()
                self.f[self.x >= actual_median] += (np.log(self.q) - np.log(right_mass))
                self.f[self.x < actual_median] += (np.log(self.p) - np.log(left_mass))

            self.fs.append(self.f.copy())
            self.thresholds.append(self.get_most_probable_threshold())

        if self.verbose >= 1:
            print()
            print('*********************************')
            print('***  THRESHOLD FOUND: %.3f' % median, '  ***')
            print('*********************************')
            print()

        time.sleep(3)

        return tbc, early_stop

    def split_text(self, text, width=100):
        tokens = text.split()
        line = list()
        result = list()
        count = 0
        for t in tokens:
            if count > width:
                result.append(' '.join(line))
                line = list()
                count = 0
            count += len(t)
            line.append(t)

        result.append(' '.join(line))
        return '\n'.join(result)

    def annotation_function(self, median, score, text):

        print()
        print('SCORE:\t %.3f' % score, '(diff. = %.3f)' % abs(median - score))
        print()
        print('TEXT:')
        print()
        print(self.split_text(text))
        print()
        if self.level == 'claim':
            print('\tCLAIM:\t\t', self.class_text)
        print('\tSUB-CLAIM:\t', self.trait_text)
        print()

        if text in self.cache:
            annot = self.cache[text]
            self.done.append(text)
            if annot == 1:
                if self.verbose >= 1:
                    print('TRUE')
                return True
            elif annot == 0:
                if self.verbose >= 1:
                    print('FALSE')
                return False

        print('\t"t" + ENTER for True')
        print('\t"f" + ENTER for False')
        print('\t"r" + ENTER to roll back to the previous annotation')
        print('\t"o" + ENTER to get another example')
        print('\t"s" + ENTER to skip this trait and move on to the next one')
        print('\t"q" + ENTER to exit and save the annotation')
        print()

        while True:

            try:
                annot = input()
            except KeyboardInterrupt:
                annot = 'q'
            if annot == 't':
                self.cache[text] = 1
                self.done.append(text)
                return True
            elif annot == 'f':
                self.cache[text] = 0
                self.done.append(text)
                return False
            elif annot == 'r':
                if self.verbose >= 1:
                    print('ROLLING BACK TO PREVIOUS ANNOT')
                return annot
            elif annot == 'o':
                if self.verbose >= 1:
                    print('GETTING ANOTHER EXAMPLE')
                self.cache[text] = None
                self.done.append(text)
                return None
            elif annot == 's':
                if self.verbose >= 1:
                    print('SKIPPING THIS TRAIT')
                return annot
            elif annot == 'q':
                if self.verbose >= 1:
                    print('PAUSING AND SAVING ANNOTATION')
                return annot
            else:
                print('Sorry, this is not a valid option. Please enter one of "t", "f", "r", "o", "s" or "c".')

    def __call__(self, trait_idx, trait_text, class_idx, class_text, column='ZSL_scores'):

        print('====== ANNOTATING:', trait_text, '(' + trait_idx + ') ======')

        self.trait_idx = trait_idx
        self.trait_text = trait_text
        self.class_text = class_text
        self.class_idx = class_idx

        self.texts_list = list()
        self.medians_list = list()
        self.scores_list = list()
        self.annot_list = list()

        self.df['scores_for_annot'] = self.df[column].apply(lambda x: x[trait_text])

        if self.level == 'claim':
            annot_idx = class_idx
        elif self.level == 'sub-claim':
            annot_idx = trait_idx

        if annot_idx + '_annot' in self.df.columns:
            rel_df = self.df[self.df[annot_idx + '_annot'].apply(lambda x: x == 0 or x == 1)]
            if len(rel_df) == 0:
                self.cache = dict()
            else:
                list_annot = list(rel_df.apply(lambda x: (x['text'], x[annot_idx + '_annot']), axis=1))
                self.cache = dict(list_annot)
        else:
            self.cache = dict()

        self.done = list()

        self.df['scores_for_annot'] = self.df[column].apply(lambda x: x[trait_text])

        f = np.ones(len(self.x))
        f /= np.sum(f)
        self.f = np.log(f)
        self.fs = [self.f.copy()]
        self.thresholds = [self.get_most_probable_threshold()]

        tbc, early_stop = self.probabilistic_bisection()

        self.df[class_idx + '_annot'] = self.df['text'].apply(
            lambda x: '' if not x in self.cache else (1 if self.cache[x] == 1 else (0 if self.cache[x] == 0 else '')))

        new_cache = pd.DataFrame()
        text_list = list(self.cache.keys())
        annot_list = list(self.cache.values())
        new_cache['text'] = text_list
        new_cache[annot_idx + '_annot'] = annot_list

        distr_list = [list(f) for f in self.fs]
        out_dict = {
            'texts': self.texts_list,
            'medians': self.medians_list,
            'scores': self.scores_list,
            'annot': self.annot_list,
            'distributions': distr_list,
            'thresholds': self.thresholds,
            'continue': tbc
        }

        return new_cache, out_dict, early_stop