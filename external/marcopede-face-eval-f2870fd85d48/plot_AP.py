#!/usr/bin/env python

#import matplotlib
# matplotlib.use('Agg')
import sys
from database import *
import VOCpr
import os
import numpy as np
from loadData import loadDetections
from getColorLabel import *
from VOCpr import evaluate_optim, filterdet

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Plots AP curves on AFW/PASCAL faces')
    parser.add_argument(
        'detfile', type=str, nargs="?", default="", help='Detection file')
    parser.add_argument(
        '--dataset', default="PASCAL", help='Select the dataset (AWF,PASCAL)')
    parser.add_argument(
        '--oldAnn', help='Use old annotations', action="store_true")
    parser.add_argument(
        '--minw', type=int, default=30, help='Minimum wide of a detectiion')
    parser.add_argument(
        '--minh', type=int, default=30, help='Minimum height of a detectiion')
    parser.add_argument('--nit', type=int, default=5,
                        help='Number of iterations for the bounding box refinement')
    args = parser.parse_args()
    minw = args.minw
    minh = args.minh
    nit = args.nit

    minpix = int(np.sqrt(0.5 * minw * minh))

    if args.dataset == "AFW":
        baseFolder = "detections/AFW"
        #testsetpath = "/users/visics/mpederso/databases/afw/testimages/"
        tsImages = getRecord(
            AFW(minw=minw, minh=minh, useOldAnn=args.oldAnn), 10000)
    elif args.dataset == "PASCAL":
        baseFolder = "detections/PASCAL"
        # testsetpath="/esat/kochab/mmathias/faceData/Annotations_Face_PASCALLayout_large/images"
        tsImages = getRecord(
            PASCALfaces(minw=minw, minh=minh, useOldAnn=args.oldAnn), 10000)
    else:
        raise ValueError('Unknown Dataset')

    # BASELINES
    pylab.figure(figsize=(8, 7))
    res = []
    for fn in glob.glob(os.path.join(baseFolder, "*")):
        ovr = 0.5
        is_point = False
        ff = os.path.basename(fn).split(".")
        if ff[0] == "Face++" or ff[0] == "Picasa" or ff[0] == "Face":
            is_point = True
        if ff[0] == "Picasa":  # and ff[1]=="cvs":
            # special case for picasa
            # we evaluated manually, selecting overlap threshold of 0.1 gives
            # in this case the correct result
            ovr = 0.1
        dets = loadDetections(fn)
        dets = filterdet(dets, minpix)
        color, label = getColorLabel(fn)
        r = evaluate_optim(
            tsImages, dets, label, color, point=is_point, iter=nit, ovr=ovr)
        res.append(r)

    # current plot
    if args.detfile != "":
        dets = loadDetections(args.detfile)
        dets = filterdet(dets, minpix)
        r = evaluate_optim(tsImages, dets, args.detfile, 'green', iter=nit)
        res.append(r)

    res.sort(key=lambda tup: tup[0], reverse=True)
    ii = []
    ll = []

    for this_idx, i in enumerate(res):
        idx = i[3]
        plot_id = i[1]
        label = i[2]
        ii.append(plot_id)
        ll.append(label)
        #print (this_idx)
        if pylab.getp(plot_id, 'zorder') < 40:
            pylab.setp(pylab.findobj(plot_id), zorder=len(res) - this_idx)
    fontsize=18
    pylab.legend(ii, ll, loc='lower left',fontsize=fontsize)
    pylab.tick_params(axis='both', which='major', labelsize=fontsize)
    pylab.xlabel("Recall",fontsize=fontsize)
    pylab.ylabel("Precision",fontsize=fontsize)
    pylab.grid()
    pylab.gca().set_xlim((0, 1))
    pylab.gca().set_ylim((0, 1))
    savename = "%s_final.pdf" % args.dataset
    pylab.savefig(savename)
    os.system("pdfcrop %s" % (savename))
    pylab.show()
    pylab.draw()
