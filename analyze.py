from collections.abc import Iterable
from tqdm import tqdm
import torch
from matplotlib import pyplot as plt
import cv2
from PIL import Image
import numpy as np

"""
in_distribution: as torch dataset
out_distribution: as a single or list of torch datasets
image: is the data image data
"""

def analyze(in_distribution, out_of_distribution, is_image=False, signal_frequency=None):
    # Names of data distributions
    
    # Check Dimensions
    if not isinstance(out_of_distribution, Iterable):
        out_of_distribution = [out_of_distribution]
    print("Getting sample from datasets...")
    samples = []
    in_distribution_subset = torch.utils.data.Subset(in_distribution, [0])   
    in_distrbution_sample = torch.utils.data.DataLoader(in_distribution_subset, batch_size=1, num_workers=0, shuffle=False)
    for i in out_of_distribution:
        samples.append(torch.utils.data.DataLoader(torch.utils.data.Subset(i, [0]), batch_size=1, num_workers=0, shuffle=False))
    print("Checking Data Demensions...")
    in_dist_shape = list(next(iter(in_distrbution_sample)).size())
    for i in tqdm(samples):
        assert (list(next(iter(i)).size()) != in_dist_shape), ("Dimensions are not consistent, was looking for dimesnions: "+str(in_dist_shape))
    print("Dimensions are consistent - Shape: "+str(in_dist_shape))
    
    # Conducting Approperiate Tests
    plots = []
    image = Image.fromarray(np.asarray(next(iter(in_distrbution_sample)).numpy))

    if is_image:
        print("Conducting the following comparisions for image data: Color distribution, Pixel distribution, and Variance of laplacian operators")
        
        print("Generating Pixel Distribution Histogram...")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow("gray", gray)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        plt.figure()
        plt.title("Grayscale Histogram")
        plt.xlabel("Bins")
        plt.ylabel("# of Pixels")
        plt.plot(hist)
        plt.xlim([0, 256])
        
        print("Generating Color Distribution Histogram...")
        chans = cv2.split(image)
        colors = ("b", "g", "r")
        plt.figure()
        plt.title("'Flattened' Color Histogram")
        plt.xlabel("Bins")
        plt.ylabel("# of Pixels")
        features = []
        # loop over the image channels
        for (chan, color) in zip(chans, colors):
            # create a histogram for the current channel and
            # concatenate the resulting histograms for each
            # channel
            hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
            features.extend(hist)
            # plot the histogram
            plt.plot(hist, color = color)
            plt.xlim([0, 256])
        
        print("Generating Multi-deminsional Color Histograms...")
        fig = plt.figure()
        # plot a 2D color histogram for green and blue
        ax = fig.add_subplot(131)
        hist = cv2.calcHist([chans[1], chans[0]], [0, 1], None,
            [32, 32], [0, 256, 0, 256])
        p = ax.imshow(hist, interpolation = "nearest")
        ax.set_title("2D Color Histogram for Green and Blue")
        plt.colorbar(p)
        # plot a 2D color histogram for green and red
        ax = fig.add_subplot(132)
        hist = cv2.calcHist([chans[1], chans[2]], [0, 1], None,
            [32, 32], [0, 256, 0, 256])
        p = ax.imshow(hist, interpolation = "nearest")
        ax.set_title("2D Color Histogram for Green and Red")
        plt.colorbar(p)
        # plot a 2D color histogram for blue and red
        ax = fig.add_subplot(133)
        hist = cv2.calcHist([chans[0], chans[2]], [0, 1], None,
            [32, 32], [0, 256, 0, 256])
        p = ax.imshow(hist, interpolation = "nearest")
        ax.set_title("2D Color Histogram for Blue and Red")
        plt.colorbar(p)     
        
        # print("Generating Pixel Distribution Histogram...")

        print("Calculating Variance of Laplacian Operators...")
        vlo = cv2.Laplacian(gray, cv2.CV_64F).var()
        
    else:
        print("Conducting the following comparisions for signal data: Min/Max/Mean/StDev, Energy, and Power")
        assert(signal_frequency is not None), "Signal Frequency cannot be None"
        
        print("Calculating Min/Max/Mean/StDev for Channels...")
        chans = cv2.split(image)
        min = []
        max = []
        mean = []
        stdev = []
        for i in chans:
            min.append(cv2.min(i))
            max.append(cv2.max(i))
            mean.append(cv2.mean(i))
            stdev.append(cv2.stdev(i))
        
        fig, axes = plt.subplots(nrows=1, ncols=len(chans), figsize=(7, 7))
        for i in range(len(chans)):
            axes[0,i].title("Energy")
            axes[0,i].magnitude_spectrum(chans[i], Fs=signal_frequency, scale='dB', color='C1')
            fig.tight_layout()
        
        fig, axes = plt.subplots(nrows=1, ncols=len(chans), figsize=(7, 7))
        for i in range(len(chans)):
            axes[0,i].title("Power")
            axes[0,i].psd(chans[i], Fs=signal_frequency, scale='dB', color='C1')
            fig.tight_layout()

    # Generating Result Graphs
    print("Displaying graphs...")