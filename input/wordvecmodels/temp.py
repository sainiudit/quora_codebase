import sys
args = dict([arg.split('=', 1) for arg in sys.argv[1:]])

print (args)


#python word2vecsimilarity.py model=glove.6B.50d.txt outprefix=glove50 outdir=w2vecsimvectors/