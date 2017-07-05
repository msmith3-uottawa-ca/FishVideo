from Tkinter import *
from tkFileDialog import askopenfilename
from tkMessageBox import showerror
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os

class Main_Interface(Frame):
    def nothing(self, x):
        pass

    def onselect(self, verts):
        self.inpaintPoints = verts
        plt.close()
        self.finished = True

    def selectROI(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.roiPts) < 4:
            self.roiPts.append((x, y))
            cv2.circle(self.frame, (x, y), 4, (0, 255, 0), 2)
            cv2.imshow("frame", self.frame)


    # @profile(precision=4)
    def run(self):
        # scaling
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c = cv2.VideoCapture(self.fileName)

            fps = c.get(cv2.CAP_PROP_FPS)
            nFrames = c.get(cv2.CAP_PROP_FRAME_COUNT)

            ret, f = c.read()
            fig, ax = plt.subplots()
            ax.imshow(f)
            print('Select Tank Diagonal (outer)')
            scalePts = plt.ginput(n=2, timeout=0)
            plt.clf()
            plt.close()

        # scaling
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c = cv2.VideoCapture(self.fileName)
            ret, f = c.read()
            fig, ax = plt.subplots()
            ax.imshow(f)
            print('Select First Electrode (left most)')
            first_elec_pt = plt.ginput(n=1, timeout=0)
            plt.clf()
            plt.close()

        # inpainting
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c = cv2.VideoCapture(self.fileName)
            ret, f = c.read()
            fig, ax = plt.subplots()
            ax.imshow(f)
            print('Select Points Around Fish')
            print('Zoom, right click then center click')
            print('')
            inpaintPoints = plt.ginput(n=0, timeout=0)
            plt.clf()
            plt.close()

        """
        Inpainting
        """
        fishMask = np.zeros(f.shape, dtype=np.uint8)
        cv2.fillPoly(fishMask, np.array([inpaintPoints], 'int32'), (255, 255, 255))
        fishMask = cv2.cvtColor(fishMask, cv2.COLOR_BGR2GRAY)
        f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

        dst_NS = cv2.inpaint(f, fishMask, 3, cv2.INPAINT_NS)
        bg = dst_NS;

        """
        Roi selection: fish
        """

        print 'Select 4 Points Around Fish (Continuous Rectangle)'
        print ''
        key = 0
        scale = 0.4
        c = cv2.VideoCapture(self.fileName)
        cv2.namedWindow("frame")
        cv2.setMouseCallback("frame", self.selectROI)
        (grabbed, self.frame) = c.read()
        cv2.imshow('frame', self.frame)
        while len(self.roiPts) < 4:
            cv2.imshow("frame", self.frame)
            cv2.waitKey(1)

        fishMask = np.zeros(self.frame.shape, dtype=np.uint8)
        cv2.fillPoly(fishMask, np.array([self.roiPts]), (255, 255, 255))
        fishMask = cv2.cvtColor(fishMask, cv2.COLOR_BGR2GRAY)
        cv2.destroyAllWindows()

        """
        Bg subtraction
        """

        c = cv2.VideoCapture(self.fileName)
        kernel = np.ones((5, 5), np.uint8)
        ret, f = c.read()
        f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        f = np.multiply(fishMask, f);
        f = cv2.blur(f, (5, 5))
        bg = np.multiply(fishMask, bg)
        bg = cv2.blur(bg, (5, 5))
        diff = f - bg;

        """
        bidirthresh fish
        """
        k = 0
        cv2.namedWindow('Tracking')
        cv2.createTrackbar('Lower', 'Tracking', 25, 255, self.nothing)
        cv2.createTrackbar('Upper', 'Tracking', 150, 255, self.nothing)
        cv2.imshow('Frame', diff)

        while (1):
            try:
                lowerThreshFish = cv2.getTrackbarPos('Lower', 'Tracking')
                upperThreshFish = cv2.getTrackbarPos('Upper', 'Tracking')
                if lowerThreshFish >= upperThreshFish:
                    lowerThreshFish = upperThreshFish - 1
                biDirThreshFish = cv2.inRange(diff, lowerThreshFish, upperThreshFish)
                biDirThreshFish = cv2.morphologyEx(biDirThreshFish, cv2.MORPH_CLOSE, kernel)

                cv2.imshow('Tracking', biDirThreshFish)
                k = cv2.waitKey(1)

                if k == 27:
                    break
            except:
                cv2.imshow('Frame', np.zeros(5))
                k = cv2.waitKey(1)
                print("Failed in Bidirectional Thresholding")

        cv2.destroyAllWindows()

        k = 0

        fishPosX = []
        fishPosY = []
        c = cv2.VideoCapture(self.fileName)
        while (1):
            ret, img = c.read()
            if not ret:
                break

            fish = np.copy(img)
            fish = cv2.cvtColor(fish, cv2.COLOR_BGR2GRAY)
            fish = np.multiply(fish, fishMask)
            fish = cv2.blur(fish, (5, 5))
            diff = fish - bg;

            biDirThreshFish = cv2.inRange(diff, lowerThreshFish, upperThreshFish)
            biDirThreshFish = cv2.morphologyEx(biDirThreshFish, cv2.MORPH_CLOSE, kernel)

            im, contoursFish, hierarchy = cv2.findContours(np.copy(biDirThreshFish), cv2.RETR_TREE,
                                                           cv2.CHAIN_APPROX_SIMPLE)

            cv2.imshow('Tracking', biDirThreshFish)

            max_areaFish = 0
            best_iFish = 0
            i = 0
            for cnt in contoursFish:
                area = cv2.contourArea(cnt)
                if area > max_areaFish:
                    max_areaFish = area
                    best_cntFish = cnt
                    best_iFish = i
                i = i + 1

            cv2.drawContours(img, contoursFish, best_iFish, (0, 255, 0), 3)
            tmpImgFish = np.zeros(np.shape(img)[:2])
            cv2.drawContours(tmpImgFish, contoursFish, best_iFish, 1, -1)
            # plt.imshow(tmpImgFish)
            M = cv2.moments(best_cntFish)
            cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
            fishPosX.append(cx)
            fishPosY.append(cy)
            cv2.circle(img, (cx, cy), 5, 255, -1)
            cv2.imshow('', img)
            k = cv2.waitKey(1)
            if (k == 27):
                break

        cv2.destroyAllWindows()
        plt.clf()
        plt.close('all')

        # cm/px
        scale = self.scale_size / np.sqrt(
            (scalePts[0][0] - scalePts[1][0]) ** 2 + (scalePts[0][1] - scalePts[1][1]) ** 2)
        fishPosX = np.array(fishPosX)
        fishPosY = np.array(fishPosY)

        fishPosX = fishPosX * scale
        fishPosY = fishPosY * scale

        t = np.arange(nFrames) / fps

        track_data = {'fishX': fishPosX, 'fishY': fishPosY, 't': t * 1000}
        track_data = pd.DataFrame(track_data, columns=['t', 'fishX', 'fishY'])

        track_data.to_csv(self.csv_name)
        print "Done Processing " + self.fileName
        exit()


    def __init__(self):
        Frame.__init__(self)
        self.master.title("Example")
        self.master.rowconfigure(5, weight=1)
        self.master.columnconfigure(5, weight=1)
        self.grid(sticky=W + E + N + S)

        self.button = Button(self, text="Browse", command=self.load_file, width=10)
        self.button.grid(row=1, column=0, sticky=W)

        self.fileName = ""
        self.csv_name = ""
        self.frame = None
        self.roiPts = []
        self.inputMode = False
        self.inpaintPoints = None

        self.scale_size = 20.9
        self.first_elec_pt = None

        self.fps = 0
        self.nFrames = 0

    def load_file(self):
        self.fileName = askopenfilename(filetypes=(("Video Files", "*.mp4"),
                                                   ("All files", "*.*")))
        if self.fileName:
            # try:
            print 'Loading:\t' + self.fileName
            self.csv_name = ''.join(self.fileName.split(".")[:-1]) + "_track" + ".csv"
            self.vid_name = self.fileName.split("/")[-1]
            if os.path.isfile(self.csv_name):
                r = raw_input('Video (' + self.vid_name + ')has already been tracked: Do you want to overwrite? [y/n]')
                if r == 'y':
                    self.run()
                else:
                    return
            else:
                self.run()


if __name__ == "__main__":
    Main_Interface().mainloop()
