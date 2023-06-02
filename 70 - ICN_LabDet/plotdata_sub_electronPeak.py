#!/usr/bin/python3.10

import os
import glob
import sys
import numpy as np
from scipy.stats import norm
from astropy.stats import median_absolute_deviation as mad
from astropy.io import fits
from matplotlib import pyplot as plt
from functions_py import voltageDictfromFitsFile, voltageDictfromFile, outputStageTiming

def sortvar(item):
	return float(item.split("_hl")[1].split("_")[0])

def plotData(argv):
	if len(argv) > 1:
		files = sys.argv[1:]
		#files =argv

		dirname=os.path.dirname(files[0])	# Get dirname of first element in files

		overscan_mask = np.s_[:, 538:]		# 538 <= x
		mask=np.s_[2:, 10:538]			# Area where variable will be computed

		
	#	fig_all.tight_layout()
		if (len(files)>1):
			fig_all, axs_all = plt.subplots(1, 4, figsize=(20, 5)) #, sharey=True)		# Create figures
			fig, axs = plt.subplots(len(files), 5, figsize=(15,10)) #, sharey=True)
			plt.rcParams.update({'font.size':'7'})
			fig.tight_layout()

		else:
			fig_all, axs_all = plt.subplots(1, 5, figsize=(20, 5)) #, sharey=True)		# Create figures
			

		for image in files:

			hdul=fits.open(image)
			img=os.path.basename(image)				# Get basename of image

			j=files.index(image)
			figctr=0

			voltageDict,file=voltageDictfromFitsFile(image)

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
					axs_all[figctr].hist(data[mask].flatten(), range=[offset-2000, offset+2000], bins=100, histtype='step', label=hlabel)	# Plot histogram of pixel values

					axs_all[figctr].set_title('ext '+str(figctr+1))

					handles_all, labels_all = axs_all[figctr].get_legend_handles_labels()

					if (len(files)==1 and i==3):
						for key in voltageDict:
							x=0
							if key.startswith('v') or key.startswith('t') or key.startswith('h') or key.startswith('s') or key.startswith('o') or key.startswith('d') or key.startswith('V') or key.startswith('T') or key.startswith('H') or key.startswith('S') or key.startswith('O') or key.startswith('r') or key.startswith('D'):
						
							#   'key' : [high=0, low=1]
								#high states
								x+=1
							if voltageDict[key][0]>0:   
								axs_all[4].annotate(voltageDict[key][0],(key, float(voltageDict[key][0]+0.05)))
								axs_all[4].bar(key,voltageDict[key][0],color='black',label=key)
								if voltageDict[key][1]>0:
									axs_all[4].bar(key,voltageDict[key][1], color='white',label=key)
								else:
									axs_all[4].bar(key,voltageDict[key][1], color='black',label=key)          
							elif voltageDict[key][1]<0:
								axs_all[4].annotate(voltageDict[key][0],(key, float(voltageDict[key][0]+0.05)))
								axs_all[4].bar(key,voltageDict[key][1],color='black',label=key)
								if voltageDict[key][0]<0:
									axs_all[4].bar(key,voltageDict[key][0], color='white',label=key)
								else:
									axs_all[4].bar(key,voltageDict[key][1], color='black',label=key)
						
							if key != 'r':
								axs_all[4].annotate(voltageDict[key][1],(key, float(voltageDict[key][1]-.5)))


					if (len(files)>1):
						# Plot one histogram per extension for EACH image
						axs[j, figctr].hist(data[mask].flatten(), range=[offset-500, offset+2000], bins=500, histtype='step', label='ext '+str(i), log=True)	# Plot histogram of pixel values

						axs[j, 0].set_title(hlabel, loc='left', pad=-1)

						for key in voltageDict:
							x=0
							if key.startswith('v') or key.startswith('t') or key.startswith('h') or key.startswith('s') or key.startswith('o') or key.startswith('d') or key.startswith('V') or key.startswith('T') or key.startswith('H') or key.startswith('S') or key.startswith('O') or key.startswith('r') or key.startswith('D'):
						
							#   'key' : [high=0, low=1]
								#high states
								x+=1
							if voltageDict[key][0]>0:   
								axs[j, 4].annotate(voltageDict[key][0],(key, float(voltageDict[key][0]+0.05)))
								axs[j, 4].bar(key,voltageDict[key][0],color='black',label=key)
								if voltageDict[key][1]>0:
									axs[j, 4].bar(key,voltageDict[key][1], color='white',label=key)
								else:
									axs[j, 4].bar(key,voltageDict[key][1], color='black',label=key)          
							elif voltageDict[key][1]<0:
								axs[j, 4].annotate(voltageDict[key][0],(key, float(voltageDict[key][0]+0.05)))
								axs[j, 4].bar(key,voltageDict[key][1],color='black',label=key)
								if voltageDict[key][0]<0:
									axs[j, 4].bar(key,voltageDict[key][0], color='white',label=key)
								else:
									axs[j, 4].bar(key,voltageDict[key][1], color='black',label=key)
						
							if key != 'r':
								axs[j, 4].annotate(voltageDict[key][1],(key, float(voltageDict[key][1]-.5))) 


					figctr=figctr+1
		# PLOT
		#else:	
					fig_all.legend(handles_all, labels_all, loc='right')
	#	plt.legend()
		plt.show()				# Show plot

	else:
		print("To run do: python3 plotdata.py path/img*.fits")


if __name__ == "__main__":
	argv=sys.argv
	plotData(argv)
