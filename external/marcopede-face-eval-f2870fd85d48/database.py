import numpy
import pylab
try:
    import scipy.misc.pilutil as pil
except:
    import scipy.misc as pil
import string
import pickle
import os.path
import glob
from util import myimread, loadmat


def getbboxVOC06(filename, cl="person", usetr=False, usedf=False):
    """
    get the ground truth bbox from the PASCAL VOC 2006 database at filename
    """
    fd = open(filename, "r")
    lines = fd.readlines()
    rect = []
    cl = "PAS" + cl
    for idx, item in enumerate(lines):
        p = item.find("Bounding box")  # look for the bounding box
        if p != -1:
            p = item.find(cl)  # check if it is a person
            if p != -1:
                p = item.find("Difficult")  # check that it is not truncated
                if p == -1 or usedf:
                    p = item.find("Trunc")  # check that it is not truncated
                    if p == -1 or usetr:
                        p = item.find(":")
                        item = item[p:]
                        # print item[p:]
                        p = item.find("(")
                        pXmin = int(item[p + 1:].split(" ")[0][:-1])
                        pYmin = int(item[p + 1:].split(" ")[1][:-1])
                        p = item[p:].find("-")
                        item = item[p:]
                        p = item.find("(")
                        pXmax = int(item[p + 1:].split(" ")[0][:-1])
                        pYmax = int(item[p + 1:].split(" ")[1][:-3])
                        rect.append((pYmin, pXmin, pYmax, pXmax, 0, 0))
    return rect

import xml.dom.minidom
from xml.dom.minidom import Node


def getbboxVOC07(filename, cl="person", usetr=False, usedf=False):
    """
    get the ground truth bbox from the PASCAL VOC 2007 database at filename
    """
    rect = []
    doc = xml.dom.minidom.parse(filename)
    for node in doc.getElementsByTagName("object"):
        # print node
        tr = 0
        df = 0
        if node.getElementsByTagName("name")[0].childNodes[0].data == cl:
            pose = node.getElementsByTagName(
                "pose")[0].childNodes[0].data  # last
            if node.getElementsByTagName("difficult")[0].childNodes[0].data == "0" or usedf:
                if node.getElementsByTagName("truncated")[0].childNodes[0].data == "0" or usetr:
                    if node.getElementsByTagName("difficult")[0].childNodes[0].data == "1":
                        df = 1
                    if node.getElementsByTagName("truncated")[0].childNodes[0].data == "1":
                        tr = 1
                    l = node.getElementsByTagName("bndbox")
                    for el in l:
                        if el.parentNode.nodeName == "object":
                            xmin = int(
                                el.getElementsByTagName("xmin")[0].childNodes[0].data)
                            ymin = int(
                                el.getElementsByTagName("ymin")[0].childNodes[0].data)
                            xmax = int(
                                el.getElementsByTagName("xmax")[0].childNodes[0].data)
                            ymax = int(
                                el.getElementsByTagName("ymax")[0].childNodes[0].data)
                            rect.append(
                                (ymin, xmin, ymax, xmax, tr, df, pose))  # last
    return rect


class imageData:

    """
    interface call to handle a database
    """
    def __init__():
        print "Not implemented"

    def getDBname():
        return "Not implemented"

    def getImage(i):
        """
        gives the ith image from the database
        """
        print "Not implemented"

    def getImageName(i):
        """
        gives the ith image name from the database
        """
        print "Not implemented"

    def getBBox(self, i):
        """
        retrun a list of ground truth bboxs from the ith image
        """
        # print "Not implemented"
        return []

    def getTotal():
        """
         return the total number of images in the db
        """
        print "Not implemented"


def getRecord(data, total=-1, pos=True, pose=False, facial=False):
    """return all the gt data in a record"""
    if total == -1:
        total = data.getTotal()
    else:
        total = min(data.getTotal(), total)
    arrPos = numpy.zeros(
        total, dtype=[("id", numpy.int32), ("name", object), ("bbox", list)])
    if facial:
        arrPos = numpy.zeros(total, dtype=[
                             ("id", numpy.int32), ("name", object), ("bbox", list), ("facial", object)])
    if pose:
        arrPos = numpy.zeros(total, dtype=[
                             ("id", numpy.int32), ("name", object), ("bbox", list), ("facial", object), ("pose", object)])
    for i in range(total):
        arrPos[i]["id"] = i
        arrPos[i]["name"] = data.getImageName(i)
        arrPos[i]["bbox"] = data.getBBox(i)
        if pose:
            arrPos[i]["pose"] = data.getPose(i)
        if facial:
            arrPos[i]["facial"] = data.getFacial(i)
    return arrPos


class VOC06Data(imageData):

    """
    VOC06 instance (you can choose positive or negative images with the option select)
    """

    def __init__(self, select="all", cl="person_train.txt",
                 basepath="meadi/DADES-2/",
                 trainfile="VOC2006/VOCdevkit/VOC2006/ImageSets/",
                 imagepath="VOC2006/VOCdevkit/VOC2006/PNGImages/",
                 annpath="VOC2006/VOCdevkit/VOC2006/Annotations/",
                 local="VOC2006/VOCdevkit/local/VOC2006/",
                 usetr=False, usedf=False, precompute=True):
        self.cl = cl
        self.usetr = usetr
        self.usedf = usedf
        self.local = basepath + local
        self.trainfile = basepath + trainfile + cl
        self.imagepath = basepath + imagepath
        self.annpath = basepath + annpath
        self.prec = precompute
        fd = open(self.trainfile, "r")
        self.trlines = fd.readlines()
        fd.close()
        if select == "all":  # All images
            self.str = ""
        if select == "pos":  # Positives images
            self.str = "1\n"
        if select == "neg":  # Negatives images
            self.str = "-1\n"
        self.selines = self.__selected()
        if self.prec:
            self.selbbox = self.__precompute()

    def __selected(self):
        lst = []
        for id, it in enumerate(self.trlines):
            if self.str == "" or it.split(" ")[-1] == self.str:
                lst.append(it)
        return lst

    def __precompute(self):
        lst = []
        tot = len(self.selines)
        cl = self.cl.split("_")[0]
        for id, it in enumerate(self.selines):
            print id, "/", tot
            filename = self.annpath + it.split(" ")[0] + ".txt"
            lst.append(getbboxVOC06(filename, cl, self.usetr, self.usedf))
        return lst

    def getDBname(self):
        return "VOC06"

    def getImage(self, i):
        item = self.selines[i]
        return myimread((self.imagepath + item.split(" ")[0]) + ".png")

    def getImageByName2(self, name):
        return myimread(self.imagepath + name + ".png")

    def getImageByName(self, name):
        return myimread(name)

    def getImageName(self, i):
        item = self.selines[i]
        return (self.imagepath + item.split(" ")[0] + ".png")

    def getImageRaw(self, i):
        item = self.selines[i]
        return im.open((self.imagepath + item.split(" ")[0]) + ".png")

    def getStorageDir(self):
        return self.local

    def getBBox(self, i, cl=None, usetr=None, usedf=None):
        if usetr == None:
            usetr = self.usetr
        if usedf == None:
            usedf = self.usedf
        if cl == None:
            cl = self.cl.split("_")[0]
        bb = []
        if self.prec:
            bb = self.selbbox[i][:]
        else:
            item = self.selines[i]
            filename = self.annpath + item.split(" ")[0] + ".txt"
            bb = getbboxVOC06(filename, cl, usetr, usedf)
        return bb

    def getBBoxByName(self, name, cl=None, usetr=None, usedf=None):
        if usetr == None:
            usetr = self.usetr
        if usedf == None:
            usedf = self.usedf
        if cl == None:
            cl = self.cl.split("_")[0]
        filename = self.annpath + name + ".txt"
        return getbboxVOC06(filename, cl, usetr, usedf)

    def getTotal(self):
        return len(self.selines)


class VOC07Data(VOC06Data):

    """
    VOC07 instance (you can choose positive or negative images with the option select)
    """

    def __init__(self, select="all", cl="person_train.txt",
                 basepath="media/DADES-2/",
                 trainfile="VOC2007/VOCdevkit/VOC2007/ImageSets/Main/",
                 imagepath="VOC2007/VOCdevkit/VOC2007/JPEGImages/",
                 annpath="VOC2007/VOCdevkit/VOC2007/Annotations/",
                 local="VOC2007/VOCdevkit/local/VOC2007/",
                 usetr=False, usedf=False, mina=0):
        self.cl = cl
        self.usetr = usetr
        self.usedf = usedf
        self.local = basepath + local
        self.trainfile = basepath + trainfile + cl
        self.imagepath = basepath + imagepath
        self.annpath = basepath + annpath
        fd = open(self.trainfile, "r")
        self.trlines = fd.readlines()
        fd.close()
        if select == "all":  # All images
            self.str = ""
        if select == "pos":  # Positives images
            self.str = "1\n"
        if select == "neg":  # Negatives images
            self.str = "-1\n"
        self.selines = self.__selected()
        self.mina = mina

    def __selected(self):
        lst = []
        for id, it in enumerate(self.trlines):
            if self.str == "" or it.split(" ")[-1] == self.str:
                lst.append(it)
        return lst

    def getDBname(self):
        return "VOC07"

    def getStorageDir(self):
        return self.local

    def getImage(self, i):
        item = self.selines[i]
        return myimread((self.imagepath + item.split(" ")[0]) + ".jpg")

    def getImageRaw(self, i):
        item = self.selines[i]
        return im.open((self.imagepath + item.split(" ")[0]) + ".jpg")

    def getImageByName(self, name):
        return myimread(name)

    def getImageByName2(self, name):
        return myimread(self.imagepath + name + ".jpg")

    def getImageName(self, i):
        item = self.selines[i]
        return (self.imagepath + item.split(" ")[0] + ".jpg")

    def getTotal(self):
        return len(self.selines)

    def getBBox(self, i, cl=None, usetr=None, usedf=None):
        if usetr == None:
            usetr = self.usetr
        if usedf == None:
            usedf = self.usedf
        if cl == None:  # use the right class
            cl = self.cl.split("_")[0]
        item = self.selines[i]
        filename = self.annpath + item.split(" ")[0] + ".xml"
        bb = getbboxVOC07(filename, cl, usetr, usedf)
        auxb = []
        for b in bb:
            a = abs(b[0] - b[2]) * abs(b[1] - b[3])
            # print a
            if a > self.mina:
                # print "OK!"
                auxb.append(b)
        return auxb


class LFW(VOC06Data):

    """
    LFW
    """

    def __init__(self, select="all", cl="face_train.txt",
                 basepath="media/DADES-2/",
                 trainfile="lfw/lfw_ffd_ann.txt",
                 imagepath="lfw/",
                 annpath="lfw/",
                 local="lfw/",
                 usetr=False, usedf=False, mina=0, fold=0, totalfold=10, fake=False):
        self.fold = fold
        self.totalfold = totalfold
        self.usetr = usetr
        self.usedf = usedf
        self.local = basepath + local
        self.trainfile = basepath + trainfile
        self.imagepath = basepath + imagepath
        self.annpath = basepath + annpath
        fd = open(self.trainfile, "r")
        self.trlines = fd.readlines()
        fd.close()
        self.selines = self.trlines[6:]
        self.total = len(self.selines)  # intial 5 lines of comments
        self.mina = mina
        self.fake = fake

    def __selected(self):
        lst = []
        for id, it in enumerate(self.trlines):
            if self.str == "" or it.split(" ")[-1] == self.str:
                lst.append(it)
        return lst

    def getDBname(self):
        return "VOC07"

    def getStorageDir(self):
        return self.local

    def getImage(self, i):
        i = i + int(self.total / self.totalfold) * self.fold
        item = self.selines[i]
        return myimread((self.imagepath + item.split(" ")[0]))

    def getImageRaw(self, i):
        i = i + int(self.total / self.totalfold) * self.fold
        item = self.selines[i]
        return im.open((self.imagepath + item.split(" ")[0]))

    def getImageByName(self, name):
        return myimread(name)

    def getImageByName2(self, name):
        return myimread(self.imagepath + name + ".jpg")

    def getImageName(self, i):
        i = i + int(self.total / self.totalfold) * self.fold
        item = self.selines[i]
        return (self.imagepath + item.split(" ")[0])

    def getTotal(self):
        return int(self.total / self.totalfold)

    def getBBox(self, i, cl=None, usetr=None, usedf=None):
        if self.fake:
            im = LFW.getImage(self, i)
            dd = 50
            bb = [[dd, dd, im.shape[0] - dd, im.shape[1] - dd, 0, 0]]
            return bb
        i = i + int(self.total / self.totalfold) * self.fold
        item = self.selines[i]
        aux = item.split()
        cx = float(aux[1])
        cy = float(aux[2])
        w = float(aux[3])
        h = float(aux[4])
        bb = [[cy, cx, cy + h, cx + w, 0, 0]]
        auxb = []
        for b in bb:
            a = abs(float(b[0]) - float(b[2])) * abs(float(b[1]) - float(b[3]))
            if a > self.mina:
                auxb.append(b)
        return auxb

    def getPose(self, i):
        i = i + int(self.total / self.totalfold) * self.fold
        item = self.selines[i]
        aux = item.split()
        return int(aux[5])

    def getFacial(self, i):
        i = i + int(self.total / self.totalfold) * self.fold
        item = self.selines[i]
        aux = item.split()
        return (numpy.array(aux[7:7 + int(aux[6]) * 2])).astype(numpy.float32)


class AFW(VOC06Data):

    """
    AFW
    """

    def __init__(self, select="all", cl="face_train.txt",
                 basepath="media/DADES-2/",
                 trainfile="",
                 imagepath="afw/testimages/",
                 annpath="afw/testimages/",
                 local="afw/",
                 usetr=False, usedf=False, minh=0, minw=0, useOldAnn=False):
        self.usetr = usetr
        self.usedf = usedf
        self.local = basepath + local
        self.imagepath = basepath + imagepath
        self.annpath = basepath + annpath
        self.useOldAnn = useOldAnn
        if useOldAnn:
            self.trainfile = "annotations/anno2.mat"
            self.ann = loadmat(self.trainfile)["anno"]
        else:
            self.trainfile = "annotations/new_annotations_AFW.mat"
            self.ann = loadmat(self.trainfile)["Annotations"]
        self.total = len(self.ann)
        self.minh = minh
        self.minw = minw

    def getDBname(self):
        return "AFW"

    def getStorageDir(self):
        return self.local

    def getImage(self, i):
        item = self.ann[i][0][0]
        return myimread((self.imagepath + item))

    def getImageRaw(self, i):
        item = self.ann[i][0][0]
        return im.open((self.imagepath + item))

    def getImageByName(self, name):
        return myimread(name)

    def getImageByName2(self, name):
        return myimread(self.imagepath + name + ".jpg")

    def getImageName(self, i):
        if self.useOldAnn:
            item = self.ann[i][0][0]
        else:
            item = self.ann[i]["imgname"][0][0]
        return (self.imagepath + item)

    def getTotal(self):
        return self.total

    def getBBox(self, i, cl=None, usetr=None, usedf=None):
        if self.useOldAnn:
            item = self.ann[i][1]
            bb = []
            for l in range(item.shape[1]):
                it = item[0, l].flatten()
                bb.append(
                    [float(it[1]), float(it[0]), float(it[3]), float(it[2]), 0, 0])
        else:
            item = self.ann[i]["objects"][0]
            bb = []
            for l in range(len(item)):
                it = item[l]  # .flatten()
                bb.append([it[1], it[0], it[3], it[2], 0, it[5]])
        auxb = []
        for b in bb:
            h = abs(float(b[0]) - float(b[2]))
            w = abs(float(b[1]) - float(b[3]))
            if w < self.minw or h < self.minh:
                b[5] = 1
            auxb.append(b)
        return auxb

    def getPose(self, i):
        return self.ann[i][2][0][0][0]  # int(aux[5])

    def getFacial(self, i):
        return self.ann[i][3][0].flatten()


class PASCALfaces(VOC06Data):

    """
    PASCAL faces
    """

    def __init__(self, select="all", cl="face_train.txt",
                 basepath="media/DADES-2/",
                 trainfile="annotations/Annotations_Face_PASCALLayout_large_fixed.mat",
                 imagepath="/xxx/",
                 annpath="afw/testimages/",
                 local="afw/",
                 usetr=False, usedf=False, minh=0, minw=0, useOldAnn=False):
        self.usetr = usetr
        self.usedf = usedf
        self.local = basepath + local
        if useOldAnn:
            pp = trainfile.find("_fixed")
            self.trainfile = trainfile[:pp] + ".mat"
        else:
            self.trainfile = trainfile
        self.imagepath = basepath + imagepath
        self.annpath = basepath + annpath
        self.minh = minh
        self.minw = minw
        self.ann = loadmat(self.trainfile)["Annotations"]
        self.total = len(self.ann)

    def getDBname(self):
        return "AFW"

    def getStorageDir(self):
        return self.local

    def getImage(self, i):
        item = self.ann[i][0][0]
        return myimread((self.imagepath + item))

    def getImageRaw(self, i):
        item = self.ann[i][0][0]
        return im.open((self.imagepath + item))

    def getImageByName(self, name):
        return myimread(name)

    def getImageByName2(self, name):
        return myimread(self.imagepath + name + ".jpg")

    def getImageName(self, i):
        item = self.ann[i]["imgname"][0][0]
        return (self.imagepath + item)

    def getTotal(self):
        return self.total

    def getBBox(self, i, cl=None, usetr=None, usedf=None):
        item = self.ann[i]["objects"][0]
        bb = []
        for l in range(len(item)):
            it = item[l]
            if len(it) < 6:
                it = [
                    float(it[0]), float(it[1]), float(it[2]), float(it[3]), 0, 0]
            bb.append([it[1], it[0], it[3], it[2], 0, it[5]])
        auxb = []
        for b in bb:
            h = abs(float(b[0]) - float(b[2]))
            w = abs(float(b[1]) - float(b[3]))
            if w < self.minw or h < self.minh:
                b[5] = 1
            auxb.append(b)
        return auxb

    def getPose(self, i):
        return []

    def getFacial(self, i):
        return []


class AFLW(VOC06Data):

    """
    AFLW
    """

    def __init__(self, select="all", cl="face_train.txt",
                 basepath="media/DADES-2/",
                 trainfile="aflw/data/aflw.sqlite",
                 imagepath="aflw/data/flickr/",
                 annpath="aflw/",
                 local="aflw/",
                 usetr=False, usedf=False, mina=0, fold=0):
        self.usetr = usetr
        self.usedf = usedf
        self.local = basepath + local
        self.trainfile = basepath + trainfile
        self.imagepath = basepath + imagepath
        self.annpath = basepath + annpath
        import sqlite3 as lite
        self.mina = mina
        con = lite.connect(self.trainfile)
        self.cur = con.cursor()
        self.cur.execute("SELECT file_id FROM Faces")
        aux = self.cur.fetchall()
        # remove bad image 9437
        del aux[8160]
        for idx, x in enumerate(aux):
            # print idx,x
            if x[0] == "image09437.jpg":
                print "Removing file ", idx
                del aux[idx]
                # raw_input()
        self.items = numpy.unique(aux)
        self.total = len(self.items)

    def getDBname(self):
        return "AFLW"

    def getStorageDir(self):
        return self.local

    def getImage(self, i):
        self.cur.execute(
            "SELECT filepath FROM FaceImages WHERE file_id = '%s'" % self.items[i][0])
        impath = self.cur.fetchall()
        return myimread((impath))

    def getImageRaw(self, i):
        item = self.ann[i][0][0]
        return im.open((self.imagepath + item))

    def getImageByName(self, name):
        return myimread(name)

    def getImageByName2(self, name):
        return myimread(self.imagepath + name + ".jpg")

    def getImageName(self, i):
        self.cur.execute(
            "SELECT filepath FROM FaceImages WHERE file_id = '%s'" % self.items[i][0])
        impath = self.imagepath + self.cur.fetchall()[0][0]
        return (impath)

    def getTotal(self):
        return self.total

    def getBBox(self, i, cl=None, usetr=None, usedf=None):
        self.cur.execute(
            "SELECT face_id FROM Faces WHERE file_id = '%s'" % self.items[i][0])
        faceid = self.cur.fetchall()
        bb = []
        for l in faceid:
            self.cur.execute(
                "SELECT x,y,w,h,annot_type_id FROM FaceRect WHERE face_id = %d" % l)
            bb += self.cur.fetchall()
        auxb = []
        for b in bb:
            a = abs(float(b[0]) - float(b[2])) * abs(float(b[1]) - float(b[3]))
            if a > self.mina:
                auxb.append([b[1], b[0], b[1] + b[3], b[0] + b[2], 0, 0])
        return auxb

    def getPose(self, i):
        self.cur.execute(
            "SELECT face_id FROM Faces WHERE file_id = '%s'" % self.items[i][0])
        faceid = self.cur.fetchall()
        poses = []
        for l in faceid:
            self.cur.execute(
                "SELECT roll,pitch,yaw FROM FacePose WHERE face_id = '%s'" % l)
            poses += self.cur.fetchall()
        return poses

    def getFacial(self, i):
        self.cur.execute(
            "SELECT face_id FROM Faces WHERE file_id = '%s'" % self.items[i][0])
        faceid = self.cur.fetchall()
        facial = []
        for l in faceid:
            self.cur.execute(
                "SELECT descr,FeatureCoords.x,FeatureCoords.y FROM FeatureCoords,FeatureCoordTypes WHERE face_id = '%s'" % l)
            facial += self.cur.fetchall()
        return facial
