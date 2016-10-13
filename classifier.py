import sys
from os.path import join

import numpy as np
from numpy import genfromtxt
from segtok import tokenizer

from ngramgen import find_n_grams

testdata = ['Too cultivated use solicitude frequently.'
            'Dashwood likewise up consider continue entrance ladyship oh.'
            'Wrong guest given purse power is no.'
            'Friendship to connection an am considered difficulty.'
            'Country met pursuit lasting moments why calling certain the.'
            'Middletons boisterous our way understood law.'
            'Among state cease how and sight since shall.'
            'Material did pleasure breeding our humanity she contempt had.'
            'So ye really mutual no cousin piqued summer result.']

data = "".join(testdata).split(".")

data_dir = sys.argv[1]
gram_dir = join(data_dir, "processed_data")

langs = ["english", "hindi"]

classes = []

english = genfromtxt(join(data_dir, "english_train.txt"), delimiter=',', dtype=str)
english_bi_wc = genfromtxt(join(gram_dir, "english_bigram_wc.txt"), delimiter=',',
                           dtype=([('gram', np.str_, 2), ('count', int)]))
english_tri_wc = genfromtxt(join(gram_dir, "english_trigram_wc.txt"), delimiter=',',
                            dtype=([('gram', np.str_, 3), ('count', int)]))
english_quad_wc = genfromtxt(join(gram_dir, "english_quadgram_wc.txt"), delimiter=',',
                             dtype=([('gram', np.str_, 4), ('count', int)]))
english_pent_wc = genfromtxt(join(gram_dir, "english_pentagram_wc.txt"), delimiter=',',
                             dtype=([('gram', np.str_, 5), ('count', int)]))
english_hexa_wc = genfromtxt(join(gram_dir, "english_hexagram_wc.txt"), delimiter=',',
                             dtype=([('gram', np.str_, 6), ('count', int)]))

hindi = genfromtxt(join(data_dir, "hindi_train.txt"), delimiter=',', dtype=str)
hindi_bi_wc = genfromtxt(join(gram_dir, "hindi_bigram_wc.txt"), delimiter=',',
                         dtype=([('gram', np.str_, 2), ('count', int)]))
hindi_tri_wc = genfromtxt(join(gram_dir, "hindi_trigram_wc.txt"), delimiter=',',
                          dtype=([('gram', np.str_, 3), ('count', int)]))
hindi_quad_wc = genfromtxt(join(gram_dir, "hindi_quadgram_wc.txt"), delimiter=',',
                           dtype=([('gram', np.str_, 4), ('count', int)]))
hindi_pent_wc = genfromtxt(join(gram_dir, "hindi_pentagram_wc.txt"), delimiter=',',
                           dtype=([('gram', np.str_, 5), ('count', int)]))
hindi_hexa_wc = genfromtxt(join(gram_dir, "hindi_hexagram_wc.txt"), delimiter=',',
                           dtype=([('gram', np.str_, 6), ('count', int)]))

for line in data:
    words = tokenizer.split_contractions(tokenizer.split_possessive_markers(line.lower().split()))
    for word in words:
        answerfound = False
        lengthofword = currentlength = len(word)

        if lengthofword > 6:
            currentlength = 6
        while answerfound is not True and currentlength > 1:
            grams = find_n_grams(word, currentlength)
            grams.sort()
            hindicounter = 0
            englishcounter = 0
            multiplier = 10000000

            # if currentlength == lengthofword:
            #    if word in english:
            #        englishcounter += 1
            #    if word in hindi:
            #        hindicounter += 1
            #    if englishcounter > hindicounter:
            #        classes.append('E')
            #        break
            #    elif hindicounter > englishcounter:
            #        classes.append('H')
            #        break
            #    if englishcounter == hindicounter and answerfound is not True:
            #        classes.append('E/H')
            #        break

            for gram in grams:
                if currentlength == 6:
                    if gram in english_hexa_wc['gram']:
                        englishcounter += 1 * multiplier
                    if gram in hindi_hexa_wc['gram']:
                        hindicounter += 1 * multiplier
                if currentlength == 5:
                    if gram in english_pent_wc['gram']:
                        englishcounter += 1 * multiplier
                    if gram in hindi_pent_wc['gram']:
                        hindicounter += 1 * multiplier
                if currentlength == 4:
                    if gram in english_quad_wc['gram']:
                        englishcounter += 1 * multiplier
                    if gram in hindi_quad_wc['gram']:
                        hindicounter += 1 * multiplier
                if currentlength == 3:
                    if gram in english_tri_wc['gram']:
                        englishcounter += 1 * multiplier
                    if gram in hindi_tri_wc['gram']:
                        hindicounter += 1 * multiplier
                if currentlength == 2:
                    if gram in english_bi_wc['gram']:
                        englishcounter += 1 * multiplier
                    if gram in hindi_bi_wc['gram']:
                        hindicounter += 1 * multiplier

            if englishcounter > hindicounter:
                answerfound = True
                classes.append('E')
            elif hindicounter > englishcounter:
                answerfound = True
                classes.append('H')

            multiplier /= 10
            if englishcounter == hindicounter and answerfound is not True:
                answerfound = True
                classes.append('E/H')
            currentlength -= 1
        if lengthofword == 1:
            classes.append('E')
print(classes)
