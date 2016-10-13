import sys
from os import listdir, makedirs
from os.path import isfile, join
from ngramgen import find_n_grams
try:
    data_dir = sys.argv[1]
    gram_dir = join(data_dir, "gram")
    makedirs(gram_dir, exist_ok=True)
except Exception as e:
    print(e)
else:
    if not isfile(data_dir):
        files = [f for f in listdir(data_dir) if isfile(join(data_dir, f)) and f.endswith("train.txt")]

        for file in files:
            lang = file.strip("train.txt")
            with open(join(data_dir, file), "r") as infile:
                outfiles = [open(join(gram_dir, lang + "bigram.txt"), "w"),
                            open(join(gram_dir, lang + "trigram.txt"), "w"),
                            open(join(gram_dir, lang + "quadgram.txt"), "w"),
                            open(join(gram_dir, lang + "pentagram.txt"), "w"),
                            open(join(gram_dir, lang + "hexagram.txt"), "w")]

                for line in infile:
                    grams = [find_n_grams(line.split()[0], i) for i in range(2, 7)]
                    for ngram, outfile in zip(grams, outfiles):
                        for gram in ngram:
                            outfile.write(gram + "\n")
                    grams.clear()
                for outfile in outfiles:
                    outfile.close()
                infile.close()
