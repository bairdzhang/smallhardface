
from __future__ import print_function

#from hog import *
import numpy
import pylab
from database import *
from util import box, overlap, overlapx
import time


def cmpscore(a, b):
    return -cmp(a[1], b[1])


def VOCprRecord(gtImages, detlist, show=False, ovr=0.5, pixels=None):
    """
        calculate the precision recall curve
    """
    dimg = {}
    tot = 0
    for idx in range(len(gtImages)):
        rect = gtImages[idx]["bbox"][:]
        if rect != []:
            dimg[gtImages[idx]["name"].split(
                "/")[-1].split(".")[0]] = {"bbox": rect, "det": [False] * len(rect)}
            for i, recti in enumerate(rect):
                if recti[5] == 0:
                    tot = tot + 1

    imname = []
    cnt = 0
    tp = numpy.zeros(len(detlist))
    fp = numpy.zeros(len(detlist))
    thr = numpy.zeros(len(detlist))
    detlist.sort(cmpscore)
    for idx, detbb in enumerate(detlist):
        found = False
        maxovr = 0
        gt = 0
        if dimg.has_key(detbb[0]):
            rect = dimg[detbb[0]]["bbox"]
            found = False
            for ir, r in enumerate(rect):
                rb = (float(detbb[3]), float(detbb[2]),
                      float(detbb[5]), float(detbb[4]))
                if pixels == None:
                    covr = overlap(rb, r)
                else:
                    covr = overlapx(rb, r, pixels)
                if covr >= maxovr:
                    maxovr = covr
                    gt = ir

        if maxovr > ovr:
            if dimg[detbb[0]]["bbox"][gt][5] == 0:
                if not(dimg[detbb[0]]["det"][gt]):
                    tp[idx] = 1
                    dimg[detbb[0]]["det"][gt] = True
                else:
                    fp[idx] = 1
        else:
            fp[idx] = 1
        thr[idx] = detbb[1]
        if show:
            prec = numpy.sum(tp) / float(numpy.sum(tp) + numpy.sum(fp))
            rec = numpy.sum(tp) / tot
            print("Scr:", detbb[1], "Prec:%.3f" % prec, "Rec:%.3f" % rec)
            ss = raw_input()
            if ss == "s" or not(found):
                pylab.ioff()
                img = gtImages.getImageByName2(detbb[0])
                pylab.figure(1)
                pylab.clf()
                pylab.imshow(img)
                rb = (float(detbb[3]), float(detbb[2]),
                      float(detbb[5]), float(detbb[4]))
                for r in rect:
                    pylab.figure(1)
                    pylab.ioff()
                    box(r[0], r[1], r[2], r[3], 'b', lw=1.5)
                if found:
                    box(rb[0], rb[1], rb[2], rb[3], 'g', lw=1.5)
                else:
                    box(rb[0], rb[1], rb[2], rb[3], 'r', lw=1.5)
                pylab.draw()
                pylab.show()
                rect = []

    return tp, fp, thr, tot


def VOCprRecordOptim(gtImages, detlist, show=False, ovr=0.5, pixels=None):
    """
        calculate the precision recall curve
    """
    tx = []
    ty = []
    sx = []
    sy = []
    dimg = {}
    tot = 0
    for idx in range(len(gtImages)):
        rect = gtImages[idx]["bbox"][:]
        if rect != []:
            dimg[gtImages[idx]["name"].split(
                "/")[-1].split(".")[0]] = {"bbox": rect, "det": [False] * len(rect)}
            for i, recti in enumerate(rect):
                if recti[5] == 0:
                    tot = tot + 1

    imname = []
    cnt = 0
    tp = numpy.zeros(len(detlist))
    fp = numpy.zeros(len(detlist))
    thr = numpy.zeros(len(detlist))
    detlist.sort(cmpscore)
    for idx, detbb in enumerate(detlist):
        found = False
        maxovr = 0
        gt = 0
        if dimg.has_key(detbb[0]):
            rect = dimg[detbb[0]]["bbox"]
            found = False
            for ir, r in enumerate(rect):
                rb = (float(detbb[3]), float(detbb[2]),
                      float(detbb[5]), float(detbb[4]))
                if pixels == None:
                    covr = overlap(rb, r)
                else:
                    covr = overlapx(rb, r, pixels)
                if covr >= maxovr:
                    maxovr = covr
                    gt = ir

        if maxovr > ovr:
            if dimg[detbb[0]]["bbox"][gt][5] == 0:
                if not(dimg[detbb[0]]["det"][gt]):
                    tp[idx] = 1
                    dimg[detbb[0]]["det"][gt] = True
                    gtx = dimg[detbb[0]]["bbox"][gt][
                        3] - dimg[detbb[0]]["bbox"][gt][1]
                    dtx = detbb[4] - detbb[2]
                    gty = dimg[detbb[0]]["bbox"][gt][
                        2] - dimg[detbb[0]]["bbox"][gt][0]
                    dty = detbb[5] - detbb[3]
                    gtcx = (
                        dimg[detbb[0]]["bbox"][gt][3] + dimg[detbb[0]]["bbox"][gt][1]) / 2.
                    dtcx = (detbb[4] + detbb[2]) / 2.
                    gtcy = (
                        dimg[detbb[0]]["bbox"][gt][2] + dimg[detbb[0]]["bbox"][gt][0]) / 2.
                    dtcy = (detbb[5] + detbb[3]) / 2.
                    tx.append((gtcx - dtcx) / float(dtx))
                    ty.append((gtcy - dtcy) / float(dty))
                    sx.append(gtx / float(dtx))
                    sy.append(gty / float(dty))
                else:
                    fp[idx] = 1
        else:
            fp[idx] = 1

        thr[idx] = detbb[1]
        if show:
            prec = numpy.sum(tp) / float(numpy.sum(tp) + numpy.sum(fp))
            rec = numpy.sum(tp) / tot
            print("Scr:", detbb[1], "Prec:%.3f" % prec, "Rec:%.3f" % rec)
            ss = raw_input()
            if ss == "s" or not(found):
                pylab.ioff()
                img = gtImages.getImageByName2(detbb[0])
                pylab.figure(1)
                pylab.clf()
                pylab.imshow(img)
                rb = (float(detbb[3]), float(detbb[2]),
                      float(detbb[5]), float(detbb[4]))
                for r in rect:
                    pylab.figure(1)
                    pylab.ioff()
                    box(r[0], r[1], r[2], r[3], 'b', lw=1.5)
                if found:
                    box(rb[0], rb[1], rb[2], rb[3], 'g', lw=1.5)
                else:
                    box(rb[0], rb[1], rb[2], rb[3], 'r', lw=1.5)
                pylab.draw()
                pylab.show()
                rect = []

    return tp, fp, thr, tot, tx, ty, sx, sy


def VOCap(rec, prec):
    mrec = numpy.concatenate(([0], rec, [1]))
    mpre = numpy.concatenate(([0], prec, [0]))
    for i in range(len(mpre) - 2, 0, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    i = numpy.where(mrec[1:] != mrec[0:-1])[0] + 1
    ap = numpy.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap


def VOColdap(rec, prec):
    rec = numpy.array(rec)
    prec = numpy.array(prec)
    ap = 0.0
    for t in numpy.linspace(0, 1, 11):
        pr = prec[rec >= t]
        if pr.size == 0:
            pr = 0
        p = numpy.max(pr)
        ap = ap + p / 11.0
    return ap


def drawPrfast(tp, fp, tot, show=True, col="g"):
    tp = numpy.cumsum(tp)
    fp = numpy.cumsum(fp)
    rec = tp / tot
    prec = tp / (fp + tp)
    ap = VOColdap(rec, prec)
    ap1 = VOCap(rec, prec)
    if show:
        pylab.plot(rec, prec, '-%s' % col)
        pylab.title("AP=%.1f 11pt(%.1f)" % (ap1 * 100, ap * 100))
        pylab.xlabel("Recall")
        pylab.ylabel("Precision")
        pylab.grid()
        pylab.gca().set_xlim((0, 1))
        pylab.gca().set_ylim((0, 1))
        pylab.show()
        pylab.draw()
    return rec, prec, ap1


def filterdet(det, minpix, scale=1.0):
    ndet = []
    minwidth = 100000000000
    minheight = minwidth
    minsize = minwidth
    for idl, l in enumerate(det):
        minwidth = min(minwidth, l[4] - l[2])
        minheight = min(minheight, l[5] - l[3])
        minsize = min(minwidth, minheight)
        if ((l[4] - l[2]) > minpix) or ((l[5] - l[3]) > minpix):
            ndet.append(
                [l[0], l[1], l[2] * scale, l[3] * scale, l[4] * scale, l[5] * scale])
    print("Minimum detection size in the dataset is: ", minsize)
    return ndet


def transf_dets(dets, tx, ty, sx, sy):
    dets2 = []
    for el in dets:
        w = (el[4] - el[2]) / 2.0
        h = (el[5] - el[3]) / 2.0
        cx = (el[4] + el[2]) / 2. + tx * w * 2
        cy = (el[5] + el[3]) / 2. + ty * h * 2
        dets2.append(
            [el[0], el[1], cx - w * sx, cy - h * sy, cx + w * sx, cy + h * sy])
    return dets2

counter = -1


def evaluate_optim(tsImages, dets, lab, color, iter=3, point=False, ovr=0.5):
    global counter
    counter += 1

    ttx = 0
    tty = 0
    tsx = 1
    tsy = 1
    for l in range(iter):
        tp, fp, scr, tot, tx, ty, sx, sy = VOCprRecordOptim(
            tsImages, dets, show=False, ovr=ovr)
        # for each TP compute the mean translation (tx,ty) and mean scale
        # (sx,sy)
        tx = numpy.mean(tx)
        ty = numpy.mean(ty)
        sx = numpy.mean(sx)
        sy = numpy.mean(sy)
        ttx += tx
        tty += ty
        tsx *= sx
        tsy *= sy
        # print("#%d tx:%f ty:%f sx:%f sy:%f"%(sum(tp),tx,ty,sx,sy))
        dets = transf_dets(dets, tx, ty, sx, sy)
    print("Tranformations " + lab, ttx, tty, tsx, tsy)

    rc, pr, ap1 = drawPrfast(tp, fp, tot, show=False)
    if point:
        lb3 = "%s " % (lab)
        plotid, = pylab.plot(
            rc[-1], pr[-1], color=color, label=lb3, marker="o", markersize=10, zorder=50)
        ap1 = -1
    else:
        lb3 = "%s (AP %.02f)" % (lab, ap1 * 100)
        plotid, = pylab.plot(
            rc, pr, color=color, label=lb3, linewidth=5, zorder=10)
    return (ap1, plotid, lb3, counter)
