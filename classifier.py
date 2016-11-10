import re
from os.path import join

import numpy
import numpy as np
from numpy import genfromtxt

from ngramgen import find_n_grams


# noinspection SpellCheckingInspection
class Classifier:
    def __init__(self, path):
        self.data_dir = path
        self.gram_dir = join(self.data_dir, "processed_data")
        self.english_bi_wc = genfromtxt(join(self.gram_dir, "english_bigram_wc.txt"), delimiter=',',
                                        dtype=([('gram', np.str_, 2), ('count', int)]))
        self.english_tri_wc = genfromtxt(join(self.gram_dir, "english_trigram_wc.txt"), delimiter=',',
                                         dtype=([('gram', np.str_, 3), ('count', int)]))
        self.english_quad_wc = genfromtxt(join(self.gram_dir, "english_quadgram_wc.txt"), delimiter=',',
                                          dtype=([('gram', np.str_, 4), ('count', int)]))
        self.english_pent_wc = genfromtxt(join(self.gram_dir, "english_pentagram_wc.txt"), delimiter=',',
                                          dtype=([('gram', np.str_, 5), ('count', int)]))
        self.english_hexa_wc = genfromtxt(join(self.gram_dir, "english_hexagram_wc.txt"), delimiter=',',
                                          dtype=([('gram', np.str_, 6), ('count', int)]))

        self.english_bi_total = sum(count for count in self.english_bi_wc['count'])
        self.english_tri_total = sum(count for count in self.english_tri_wc['count'])
        self.english_quad_total = sum(count for count in self.english_quad_wc['count'])
        self.english_pent_total = sum(count for count in self.english_pent_wc['count'])
        self.english_hexa_total = sum(count for count in self.english_hexa_wc['count'])

        self.hindi_bi_wc = genfromtxt(join(self.gram_dir, "hindi_bigram_wc.txt"), delimiter=',',
                                      dtype=([('gram', np.str_, 2), ('count', int)]))
        self.hindi_tri_wc = genfromtxt(join(self.gram_dir, "hindi_trigram_wc.txt"), delimiter=',',
                                       dtype=([('gram', np.str_, 3), ('count', int)]))
        self.hindi_quad_wc = genfromtxt(join(self.gram_dir, "hindi_quadgram_wc.txt"), delimiter=',',
                                        dtype=([('gram', np.str_, 4), ('count', int)]))
        self.hindi_pent_wc = genfromtxt(join(self.gram_dir, "hindi_pentagram_wc.txt"), delimiter=',',
                                        dtype=([('gram', np.str_, 5), ('count', int)]))
        self.hindi_hexa_wc = genfromtxt(join(self.gram_dir, "hindi_hexagram_wc.txt"), delimiter=',',
                                        dtype=([('gram', np.str_, 6), ('count', int)]))

        self.hindi_bi_total = sum(count for count in self.hindi_bi_wc['count'])
        self.hindi_tri_total = sum(count for count in self.hindi_tri_wc['count'])
        self.hindi_quad_total = sum(count for count in self.hindi_quad_wc['count'])
        self.hindi_pent_total = sum(count for count in self.hindi_pent_wc['count'])
        self.hindi_hexa_total = sum(count for count in self.hindi_hexa_wc['count'])

        self.e_multiplier_norm = self.english_bi_total + self.english_tri_total + self.english_quad_total + \
                                 self.english_pent_total + self.english_hexa_total
        self.h_multiplier_norm = self.hindi_bi_total + self.hindi_tri_total + self.hindi_quad_total + \
                                 self.hindi_pent_total + self.hindi_hexa_total

        self.e_multiplier = [
            self.english_hexa_total + self.english_pent_total + self.english_quad_total + self.english_tri_total,
            self.english_hexa_total + self.english_pent_total + self.english_quad_total + self.english_bi_total,
            self.english_hexa_total + self.english_pent_total + self.english_tri_total + self.english_bi_total,
            self.english_hexa_total + self.english_quad_total + self.english_tri_total + self.english_bi_total,
            self.english_pent_total + self.english_quad_total + self.english_tri_total + self.english_bi_total
        ]

        self.e_multiplier = [normalized_wt for normalized_wt in
                             map(lambda x: x / self.e_multiplier_norm, self.e_multiplier)]
        self.h_multiplier = [
            self.hindi_hexa_total + self.hindi_pent_total + self.hindi_quad_total + self.hindi_tri_total,
            self.hindi_hexa_total + self.hindi_pent_total + self.hindi_quad_total + self.hindi_bi_total,
            self.hindi_hexa_total + self.hindi_pent_total + self.hindi_tri_total + self.hindi_bi_total,
            self.hindi_hexa_total + self.hindi_quad_total + self.hindi_tri_total + self.hindi_bi_total,
            self.hindi_pent_total + self.hindi_quad_total + self.hindi_tri_total + self.hindi_bi_total
        ]

        self.h_multiplier = [normalized_wt for normalized_wt in
                             map(lambda x: x / self.h_multiplier_norm, self.h_multiplier)]

    def classify(self, data):
        classes = []
        allwords = []

        words = re.sub(r'[^a-zA-Z ]', '', data.lower()).split()

        allwords.extend(words)
        for word in words:
            lengthofword = currentlength = len(word)

            if lengthofword > 6:
                currentlength = 6

            if currentlength > 1:
                for gram_length in range(2, currentlength + 1):
                    grams = find_n_grams(word, gram_length)
                    hindicounter = 0
                    englishcounter = 0

                    for gram in grams:
                        if gram_length == 6:
                            if gram in self.english_hexa_wc['gram']:
                                index = numpy.where(self.english_hexa_wc['gram'] == gram)[0][0]
                                englishcounter += 1 * self.e_multiplier[4] * self.english_hexa_wc['count'][
                                    index] / self.english_hexa_total
                            if gram in self.hindi_hexa_wc['gram']:
                                index = numpy.where(self.hindi_hexa_wc['gram'] == gram)[0][0]
                                hindicounter += 1 * self.h_multiplier[4] * self.hindi_hexa_wc['count'][
                                    index] / self.hindi_hexa_total
                        elif gram_length == 5:
                            if gram in self.english_pent_wc['gram']:
                                index = numpy.where(self.english_pent_wc['gram'] == gram)[0][0]
                                englishcounter += 1 * self.e_multiplier[3] * self.english_pent_wc['count'][
                                    index] / self.english_pent_total
                            if gram in self.hindi_pent_wc['gram']:
                                index = numpy.where(self.hindi_pent_wc['gram'] == gram)[0][0]
                                hindicounter += 1 * self.h_multiplier[3] * self.hindi_pent_wc['count'][
                                    index] / self.hindi_pent_total
                        elif gram_length == 4:
                            if gram in self.english_quad_wc['gram']:
                                index = numpy.where(self.english_quad_wc['gram'] == gram)[0][0]
                                englishcounter += 1 * self.e_multiplier[2] * self.english_quad_wc['count'][
                                    index] / self.english_quad_total
                            if gram in self.hindi_quad_wc['gram']:
                                index = numpy.where(self.hindi_quad_wc['gram'] == gram)[0][0]
                                hindicounter += 1 * self.h_multiplier[2] * self.hindi_quad_wc['count'][
                                    index] / self.hindi_quad_total
                        elif gram_length == 3:
                            if gram in self.english_tri_wc['gram']:
                                index = numpy.where(self.english_tri_wc['gram'] == gram)[0][0]
                                englishcounter += 1 * self.e_multiplier[1] * self.english_tri_wc['count'][
                                    index] / self.english_tri_total
                            if gram in self.hindi_tri_wc['gram']:
                                index = numpy.where(self.hindi_tri_wc['gram'] == gram)[0][0]
                                hindicounter += 1 * self.h_multiplier[1] * self.hindi_tri_wc['count'][
                                    index] / self.hindi_tri_total
                        elif gram_length == 2:
                            if gram in self.english_bi_wc['gram']:
                                index = numpy.where(self.english_bi_wc['gram'] == gram)[0][0]
                                englishcounter += 1 * self.e_multiplier[0] * self.english_bi_wc['count'][
                                    index] / self.english_bi_total
                            if gram in self.hindi_bi_wc['gram']:
                                index = numpy.where(self.hindi_bi_wc['gram'] == gram)[0][0]
                                hindicounter += 1 * self.h_multiplier[0] * self.hindi_bi_wc['count'][
                                    index] / self.hindi_bi_total

                # 10% relaxation
                if englishcounter > 1.1*hindicounter:
                    classes.append('E')
                elif hindicounter > 1.1*englishcounter:
                    classes.append('H')
                else:
                    classes.append('H/E')

            if lengthofword == 1:
                classes.append('E')

        return classes
