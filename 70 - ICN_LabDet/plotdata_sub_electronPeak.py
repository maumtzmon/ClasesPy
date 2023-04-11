#!/usr/bin/python3.10

import os
import glob
import sys
import numpy as np
from scipy.stats import norm
from astropy.stats import median_absolute_deviation as mad
from astropy.io import fits
from matplotlib import pyplot as plt

def sortvar(item):
	return float(item.split("_hl")[1].split("_")[0])

def plotData(argv):
	if len(argv) > 1:
		files = sys.argv[1:]
		#files =argv

		dirname=os.path.dirname(files[0])	# Get dirname of first element in files

		overscan_mask = np.s_[:, 538:]		# 538 <= x
		mask=np.s_[2:, 10:538]			# Area where variable will be computed

		fig_all, axs_all = plt.subplots(1, 4, figsize=(20, 5))		# Create figures
	#	fig_all.tight_layout()

		if (len(files)>1):
			fig, axs = plt.subplots(len(files), 4, figsize=(15,10))
			fig.tight_layout()

		for image in files:

			hdul=fits.open(image)
			img=os.path.basename(image)				# Get basename of image

			j=files.index(image)
			figctr=0

			for i in range(0, len(hdul)):


				# Load data and header
				data=hdul[i].data
				header=hdul[i].header

				if data is not None:				# Check if data is not empty

					hlabel = img

					# Plot histogram of data
					hist, bin_edges = np.histogram(data[mask].flatten(), bins=10000)
					offset = bin_edges[np.argmax(hist)]

					# Plot one histogram per extension for ALL images
					axs_all[figctr].hist(data[mask].flatten(), range=[-3000, offset+3000], bins=100, histtype='step', label=hlabel)	# Plot histogram of pixel values

					axs_all[figctr].set_title('ext '+str(figctr+1))

					handles_all, labels_all = axs_all[figctr].get_legend_handles_labels()

					if (len(files)>1):
						# Plot one histogram per extension for EACH image
						axs[j, figctr].hist(data[mask].flatten(), range=[offset-3000, offset+7500], bins=500, histtype='step', label='ext '+str(i), log=True)	# Plot histogram of pixel values

						axs[j, 0].set_title(hlabel, loc='left', pad=-1)

					figctr=figctr+1
		# PLOT
		fig_all.legend(handles_all, labels_all, loc='right')
	#	plt.legend()
		plt.show()				# Show plot

	else:
		print("To run do: python3 plotdata.py path/img*.fits")


if __name__ == "__main__":
	argv=sys.argv
	plotData(argv)
