#!/usr/bin/env python

# plots all the PR curves for fddb
# As in this case we do not optimize the bbox, we use a separate code
#import matplotlib
# matplotlib.use('Agg')
import glob
import numpy
import pylab
import os
import sys
from getColorLabel import *


def load_fddb(ff):
    print ff
    ftxt = open(ff, "rb")
    if ff == "detections/fddb/OlaworksDiscROC.txt":
        data = numpy.genfromtxt(ftxt, delimiter="\t")
    else:
        data = numpy.genfromtxt(ftxt, delimiter=" ")
    data.sort(axis=0)
    data = data[::-1]
    ln = os.path.splitext(ff)
    ln = os.path.basename(ln[0])
    if ln[-1] == "_":
        ln = ln[:-1]
    label = ln
    pp = label.find("DiscROC")
    label = label[:pp]
    if label[-1] == "_":
        label = label[:-1]
    color, label = getColorLabel(label)
    pos = numpy.where(data[:, 1] < 1000)[0][0]
    pr = data[pos, 0]
    print pr
    label = label + " (%0.3f)" % (pr)
    plotid, = pylab.plot(data[:, 1], data[:, 0], color=color, linewidth=5)
    return plotid, pr, label


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Plots AP curves on fddb faces')
    parser.add_argument('detfile', type=str, nargs="?",
                        default="", help='Detection file in fddb format')
    args = parser.parse_args()
    color_list = pylab.cm.Set1(numpy.linspace(0, 1, 26))
    colorCounter = 11
    pylab.figure(figsize=(8, 7))
    lff = glob.glob("detections/fddb/*.txt")

    method = []
    for idff, ff in enumerate(lff):
        pid, pr, l = load_fddb(ff)
        method.append((pr, idff, pid, l))

    if args.detfile != "":
        pid, pr, l = load_fddb(args.detfile)
        method.append((pr, idff, pid, l))

    method = sorted(method, reverse=True)
    ll = []
    ii = []
    for this_idx, i in enumerate(method):
        ii.append(i[2])
        ll.append(i[3])
        pylab.setp(i[2],  zorder=len(method) - this_idx)
    fontsize=18
    pylab.legend(ii, ll, loc='lower right', ncol=2, fontsize=11)
    pylab.tick_params(axis='both', which='major', labelsize=fontsize)
    pylab.ylabel("True positive rate", fontsize=fontsize)
    pylab.xlabel("False positives", fontsize=fontsize)
    pylab.grid()
    pylab.gca().set_xlim((0, 2000))
    pylab.gca().set_ylim((0, 1))
    pylab.yticks(numpy.linspace(0, 1, 11))
    savename = "FDDB_final.pdf"
    pylab.savefig(savename)
    os.system("pdfcrop %s" % (savename))

    pylab.show()
    pylab.draw()
