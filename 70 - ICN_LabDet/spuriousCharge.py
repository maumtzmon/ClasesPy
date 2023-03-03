import os
import glob
import sys
import numpy as np
from scipy.stats import norm
from astropy.stats import median_absolute_deviation as mad
from astropy.io import fits
from matplotlib import pyplot as plt


def plotData(argv):
    if len(argv) > 1:
        files = sys.argv[1:]
        #files =argv

        dirname=os.path.dirname(files[0])	# Get dirname of first element in files

        overscan_mask = np.s_[:, 538:]		# 538 <= x
        mask=np.s_[2:, 10:538]			# Area where variable will be computed

        fig_all, axs_all = plt.subplots(1, 4, figsize=(20, 5))		# Create figures
        fig_all.tight_layout()
        if (len(files)>1):
            fig, axs = plt.subplots(len(files), 4, figsize=(15,10))
            fig.tight_layout()

        for image in files:
            hdul=fits.open(image)
            img=os.path.basename(image)				# Get basename of image

            j=files.index(image)
            figctr=0
            for i in range(0, len(hdul)):
                data=hdul[i].data
                header=hdul[i].header
                axs_all[1,figctr].plot(data[1], range(0,700) )

        else:
            print("")
            

            
        # PLOT

        #	plt.legend()
        plt.show()				# Show plot

    else:
        print("To run do: python3 plotdata.py path/img*.fits")

if __name__ == "__main__":
	argv=sys.argv
	plotData(argv)