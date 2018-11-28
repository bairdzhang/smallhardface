
import cPickle
import numpy
import pylab
from scipy.io.matlab import savemat, loadmat


def myimread(imgname, flip=False, resize=None):
    """
        read an image
    """
    img = None
    if imgname.split(".")[-1] == "png":
        img = pylab.imread(imgname)
    else:
        img = numpy.ascontiguousarray(pylab.imread(imgname)[::-1])
    if flip:
        img = numpy.ascontiguousarray(img[:, ::-1, :])
    if resize != None:
        from scipy.misc import imresize
        img = imresize(img, resize)
    return img


def save(filename, obj, prt=2):
    """
        save any python object
    """
    fd = open(filename, "w")
    cPickle.dump(obj, fd, prt)
    fd.close()

# def savemat(filename,dic):
#    """
#        save an array in matlab format
#    """
#    import scipy.io.matlab
#    fd=open(filename,"w")
#    scipy.io.matlab.savemat(filename,dic)
#    fd.close()


def load(filename):
    """
        load any python object
    """
    fd = open(filename, "r")
    aux = cPickle.load(fd)
    fd.close()
    return aux

# def loadmat(filename):
#    """
#        load an array in matlab format
#    """
#    import scipy.io.matlab
#    aux = scipy.io.matlab.loadmat(filename)
#    fd.close()
#    return aux


def drawModel(mfeat, mode="black", parts=True):
    """
        draw the HOG weight of an object model
    """
    col = ["r", "g", "b"]
    import drawHOG
    lev = len(mfeat)
    if mfeat[0].shape[0] > mfeat[0].shape[1]:
        sy = 1
        sx = lev
    else:
        sy = lev
        sx = 1
    for l in range(lev):
        pylab.subplot(sy, sx, l + 1)
        if mode == "white":
            drawHOG9(mfeat[l])
        elif mode == "black":
            img = drawHOG.drawHOG(mfeat[l])
            pylab.axis("off")
            pylab.imshow(img, cmap=pylab.cm.gray, interpolation="nearest")
        if parts == True:
            for x in range(0, 2 ** l):
                for y in range(0, 2 ** l):
                    boxHOG(mfeat[0].shape[1] * x, mfeat[0].shape[0] * y,
                           mfeat[0].shape[1], mfeat[0].shape[0], col[l], 5 - l)


def drawDeform(dfeat, mindef=0.001):
    """
        draw the deformation weight of an object model
    """
    from matplotlib.patches import Ellipse
    lev = len(dfeat)
    if 1:
        sy = 1
        sx = lev
    else:
        sy = lev
        sx = 1
    pylab.subplot(sy, sx, 1)
    x1 = -0.5
    x2 = 0.5
    y1 = -0.5
    y2 = 0.5
    pylab.fill([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1],
               "b", alpha=0.15, edgecolor="b", lw=1)
    pylab.fill([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1],
               "r", alpha=0.15, edgecolor="r", lw=1)
    wh = numpy.exp(-mindef / dfeat[0][0, 0, 0]) / numpy.exp(1)
    hh = numpy.exp(-mindef / dfeat[0][0, 0, 1]) / numpy.exp(1)
    e = Ellipse(xy=[0, 0], width=wh, height=hh, alpha=0.35)
    col = numpy.array([wh * hh] * 3).clip(0, 1)
    col[0] = 0
    e.set_facecolor(col)
    pylab.axis("off")
    pylab.gca().add_artist(e)
    pylab.gca().set_ylim(-0.5, 0.5)
    pylab.gca().set_xlim(-0.5, 0.5)
    for l in range(1, lev):
        pylab.subplot(sy, sx, l + 1)
        for ry in range(2 ** (l - 1)):
            for rx in range(2 ** (l - 1)):
                drawDef(dfeat[l][ry * 2:(ry + 1) * 2, rx * 2:(rx + 1)
                                 * 2, 2:] * 4 ** l, 4 * ry, 4 * rx, distr="child")
                drawDef(dfeat[l][ry * 2:(ry + 1) * 2, rx * 2:(rx + 1) * 2, :2] *
                        4 ** l, ry * 2 ** (l), rx * 2 ** (l), mindef=mindef, distr="father")
        # pylab.gca().set_ylim(-0.5,(2.6)**l)
        pylab.axis("off")
        pylab.gca().set_ylim((2.6) ** l, -0.5)
        pylab.gca().set_xlim(-0.5, (2.6) ** l)


def drawDef(dfeat, dy, dx, mindef=0.001, distr="father"):
    """
        auxiliary funtion to draw recursive levels of deformation
    """
    from matplotlib.patches import Ellipse
    pylab.ioff()
    if distr == "father":
        py = [0, 0, 2, 2]
        px = [0, 2, 0, 2]
    if distr == "child":
        py = [0, 1, 1, 2]
        px = [1, 2, 0, 1]
    ordy = [0, 0, 1, 1]
    ordx = [0, 1, 0, 1]
    x1 = -0.5 + dx
    x2 = 2.5 + dx
    y1 = -0.5 + dy
    y2 = 2.5 + dy
    if distr == "father":
        pylab.fill([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1],
                   "r", alpha=0.15, edgecolor="b", lw=1)
    for l in range(len(py)):
        aux = dfeat[ordy[l], ordx[l], :].clip(-1, -mindef)
        wh = numpy.exp(-mindef / aux[0]) / numpy.exp(1)
        hh = numpy.exp(-mindef / aux[1]) / numpy.exp(1)
        e = Ellipse(
            xy=[(px[l] + dx), (py[l] + dy)], width=wh, height=hh, alpha=0.35)
        x1 = -0.75 + dx + px[l]
        x2 = 0.75 + dx + px[l]
        y1 = -0.76 + dy + py[l]
        y2 = 0.75 + dy + py[l]
        col = numpy.array([wh * hh] * 3).clip(0, 1)
        if distr == "father":
            col[0] = 0
        e.set_facecolor(col)
        pylab.gca().add_artist(e)
        if distr == "father":
            pylab.fill([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1],
                       "b", alpha=0.15, edgecolor="b", lw=1)


def overlap(rect1, rect2):
    """
        Calculate the overlap between two boxes
    """
    dy1 = abs(rect1[0] - rect1[2]) + 1
    dx1 = abs(rect1[1] - rect1[3]) + 1
    dy2 = abs(rect2[0] - rect2[2]) + 1
    dx2 = abs(rect2[1] - rect2[3]) + 1
    a1 = dx1 * dy1
    a2 = dx2 * dy2
    ia = 0
    if rect1[2] > rect2[0] and rect2[2] > rect1[0] and rect1[3] > rect2[1] and rect2[3] > rect1[1]:
        xx1 = max(rect1[1], rect2[1])
        yy1 = max(rect1[0], rect2[0])
        xx2 = min(rect1[3], rect2[3])
        yy2 = min(rect1[2], rect2[2])
        ia = (xx2 - xx1 + 1) * (yy2 - yy1 + 1)
    return ia / float(a1 + a2 - ia)


def inclusion(rect1, rect2):
    """
        Calculate the intersection percentage between two rectangles
        Note that it is not anymore symmetric
    """
    dy1 = abs(rect1[0] - rect1[2]) + 1
    dx1 = abs(rect1[1] - rect1[3]) + 1
    dy2 = abs(rect2[0] - rect2[2]) + 1
    dx2 = abs(rect2[1] - rect2[3]) + 1
    a1 = dx1 * dy1
    a2 = dx2 * dy2
    ia = 0
    if rect1[2] > rect2[0] and rect2[2] > rect1[0] and rect1[3] > rect2[1] and rect2[3] > rect1[1]:
        xx1 = max(rect1[1], rect2[1])
        yy1 = max(rect1[0], rect2[0])
        xx2 = min(rect1[3], rect2[3])
        yy2 = min(rect1[2], rect2[2])
        ia = (xx2 - xx1 + 1) * (yy2 - yy1 + 1)
    return ia / float(a1)


def myinclusion(rect1, rect2):
    """
        Calculate the intersection percentage between two rectangles
        Note that it is not anymore symmetric
    """
    dy1 = abs(rect1[0] - rect1[2]) + 1
    dx1 = abs(rect1[1] - rect1[3]) + 1
    dy2 = abs(rect2[0] - rect2[2]) + 1
    dx2 = abs(rect2[1] - rect2[3]) + 1
    cy1 = (rect1[0] - rect1[2]) / 2.0
    cx1 = (rect1[1] - rect1[3]) / 2.0
    cy2 = (rect2[0] - rect2[2]) / 2.0
    cx2 = (rect2[1] - rect2[3]) / 2.0
    dc = numpy.sqrt(
        ((cy1 - cy2) / float(dy2)) ** 2 + ((cx1 - cx2) / float(dx2)) ** 2)
    # print dc
    a1 = dx1 * dy1
    a2 = dx2 * dy2
    if dx1 > dy1:  # xgt
        a21 = max(dx2, dx1) * dy1
    else:  # ygt
        a21 = max(dy1, dy2) * dx1
    ia = 0
    if rect1[2] > rect2[0] and rect2[2] > rect1[0] and rect1[3] > rect2[1] and rect2[3] > rect1[1]:
        xx1 = max(rect1[1], rect2[1])
        yy1 = max(rect1[0], rect2[0])
        xx2 = min(rect1[3], rect2[3])
        yy2 = min(rect1[2], rect2[2])
        ia = (xx2 - xx1 + 1) * (yy2 - yy1 + 1)
    # print dy1,dx1,dy2,dx2
    # print ia
    # print a21
    return ia / float(a21) - dc


def overlapx(rect1, rect2, pixels=20):
    """
        Calculate the intersection percentage between two rectangles
        Note that it is not anymore symmetric
    """
    dy1 = abs(rect1[0] - rect1[2]) + 1
    dx1 = abs(rect1[1] - rect1[3]) + 1
    dy2 = abs(rect2[0] - rect2[2]) + 1
    dx2 = abs(rect2[1] - rect2[3]) + 1
    cy1 = (rect1[0] - rect1[2]) / 2.0
    cx1 = (rect1[1] - rect1[3]) / 2.0
    cy2 = (rect2[0] - rect2[2]) / 2.0
    cx2 = (rect2[1] - rect2[3]) / 2.0
    dc = max(abs(cy1 - cy2) / float(pixels * 2),
             abs(cx1 - cx2) / float(pixels * 2))
    return 1 - dc


def boxHOG(px, py, dx, dy, col, lw):
    """
        bbox one the HOG weights
    """
    k = 1
    d = 15
    pylab.plot([px * d + 0 - k, px * d + 0 - k],
               [py * d + 0 - k, py * d + dy * d - k], col, lw=lw)
    pylab.plot([px * d + 0 - k, px * d + dx * d - k],
               [py * d + 0 - k, py * d + 0 - k], col, lw=lw)
    pylab.plot([px * d + dx * 15 - k, px * d + dx * d - k],
               [py * d + 0 - k, py * d + dy * d - k], col, lw=lw)
    pylab.plot([px * d + 0 - k, px * d + dx * d - k],
               [py * d + dy * d - k, py * d + dy * d - k], col, lw=lw)
    pylab.axis("image")


def box(p1y, p1x, p2y, p2x, col='b', lw=1):
    """
        plot a bbox with the given coordinates
    """
    pylab.plot(
        [p1x, p1x, p2x, p2x, p1x], [p1y, p2y, p2y, p1y, p1y], col, lw=lw)
