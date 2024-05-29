#!/usr/bin/python3.10

import os
import glob
import sys
import numpy as np
import math
#from ROOT import *
from scipy.stats import norm
from astropy.stats import median_absolute_deviation as mad
from astropy.io import fits
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

def gaussian(x, norm, mean, sigma):
	return norm * np.exp(-((x - mean)**2 / (2 * sigma**2)))

def sqrt(x, sigma0):
	return sigma0 / np.sqrt(x)

def varsort(item):
	return int(item.split("_NSAMP_")[1].split("_")[0])
	

def doublegaus(x, norm, offset, noise, gain, mu):
	return (1.0-mu)*norm*np.exp(-((x-offset)**2/(2*(gain*noise)**2))) + mu*norm*np.exp(-((x-offset-gain)**2/(2*(gain*noise)**2)))

def sumgaus(x, norm, offset, noise, gain, mu, npeaks):
	return sum(norm*((mu**i)/np.math.factorial(i))*np.exp(-((x-offset-(i*gain))**2/(2*(gain*noise)**2))) for i in range(npeaks))

if len(sys.argv) > 1:
	files = sys.argv[1:]
	#files.sort(key=varsort)			# Sort files by key
	files.sort(key=os.path.getmtime)

	dirname=os.path.dirname(files[0])	# Get dirname of first element in files

#	latest_file = max(files, key=os.path.getctime)
#	print(latest_file)
#	image='image.fz'

	# Define active and overscan areas
	active_mask = np.s_[:, 9:538]		#   9 <= x < 538
	overscan_mask = np.s_[:, 538:]		# 538 <= x

	mask=np.s_[:, 538:700] #OverScan			# Area where variable will be computed
	#mask=np.s_[2:, 10:538] 	#active Area
	list_var=[]				# List to store the computed variable

	expgain = [227.7013, 220.4891, 154.6271, 197.7721]#201.8325949210918, 194.70825464645284, 202.97945260519987, 193.2145155088731]	# Expected gain; if only fitting 1 peak, noise is divided by this number
	numpeaks = 2				# Number of peaks to fit

	varsplot = ["Constant (ADU)", "Offset (ADU)", "Noise (e-)", "Gain (ADU/e-)", "SER (e-/pix)"]	# Variables that can be chosen to be plotted
	parplot = 3			# From varsplot, index of parameter to plot; for example if you want to plot Noise (e-), varsplot=3
	string=''				# Variable to be the x axis, as shown in the header of the image

#	fig_all, axs_all = plt.subplots(1, 4, figsize=(20, 5))		# Define figure to stack histograms of all images
#	fig_all.tight_layout()

	# Open datafile and write 
#	datafile=open(dirname+"/noisevspsamp.txt","a+")
#	datafile.write("#PSAMP\tNoise_ext1\tNoise_ext2\tNoise_ext3\tNoise_ext4\n")

	deltaH=[]
	deltaV=[]
	deltaT=[]
	deltaSW=[]
	RunID=[]
	vFile = []
	valuesDict = {}

	for image in files:
		fig_all, axs_all = plt.subplots(1, 4, figsize=(20, 5))	# Define figure to stack histogram of each image
#		fig_all.tight_layout()
		imgNumber=image.split(".")[-1].split('img')[-1]

		hdul=fits.open(image)
#		hdul.info()
		img=os.path.basename(image)				# Get basename of image
		print("\nImage: "+img)

		j=files.index(image)
		figctr=0

		# Extract info from img title
#		string=image.split("img")[1].split(".")[0]		# To obtain imgID

		var=[]; var_fit=[]

		print("# of peaks to fit: "+str(numpeaks)+"\n")

#		for i in range(0,1):
		for i in range(len(hdul)):
			data=hdul[i].data; header=hdul[i].header	# Load data and header
			

			if data is not None:								# Check if data is not empty
#				data = data - np.median(data, axis=0, keepdims=True)			# Subtract median per column
				#data = data - np.median(data[active_mask], axis=1, keepdims=True)	# Subtract OS median per row (use when proc*fits have no baseline substracted)

				# Extract info from header
				####   X axis Variable   ####
				stringval= header['RUNID']  
				#stringval=image.split('delay_')[1].split('_')[0]#   header["RUNID"]#
				#stringval=float(header["H1AH"])-float(header["H1AL"])
				nsamp=float(header['NSAMP'])
				#nsamp=1
#				hlabel = string+" "+stringval			# Define label of histogram
				hlabel = image

				#Plot histogram of data to obtain offset
				hist, bin_edges = np.histogram(data[mask].flatten(), bins=10000)
				offset = bin_edges[np.argmax(hist)]
#				print(offset)
				data = data-offset		# Subtract offset from data (offset artificial)
#				offset = 0			# Offset to plot
				
				bin_heights, bin_borders, _ = axs_all[figctr].hist(data[mask].flatten(), bins=20000, histtype='step', label=hlabel)	# Plot histogram
				offset_fit = bin_borders[np.argmax(bin_heights)]	# Offset to fit
#				offset_fit = 0						# Offset to fit
				axs_all[figctr].set_title('ext '+str(figctr+1))
				handles_all, labels_all = axs_all[figctr].get_legend_handles_labels()

				bin_centers=np.array([(bin_borders[p+1]+bin_borders[p])/2 for p in range(len(bin_heights))])	# Compute centers of each bin

#				if nsamp>100: numpeaks=2
#				else: numpeaks=1
#				print("# of peaks to fit: "+str(numpeaks))

				# Fit gaussians
				aux_arr=np.array([])
				for npeak in range(numpeaks):
					xmin_fit = offset_fit+(npeak*expgain[figctr])-(5*expgain[figctr])/math.sqrt(nsamp) 	# Define fit range
#					xmin_fit = offset_fit+(npeak*expgain[figctr])-(0.25*expgain[figctr])
					xmax_fit = offset_fit+(npeak*expgain[figctr])+(5*expgain[figctr])/math.sqrt(nsamp)
#					xmax_fit = offset_fit+(npeak*expgain[figctr])+(0.25*expgain[figctr])
					print("Ext"+str(figctr)+" trying to fit peak of "+str(npeak)+"e- between: xmin = "+str(xmin_fit)+" & xmax = "+str(xmax_fit))

					bin_heights_peak = bin_heights[(bin_centers>xmin_fit) & (bin_centers<xmax_fit)]		# Constrain histogram to given range
					bin_centers_peak = bin_centers[(bin_centers>xmin_fit) & (bin_centers<xmax_fit)]

					try:	# Try to fit, pass if error
						#popt, pcov = curve_fit(gaussian, bin_centers_peak, bin_heights_peak, p0=[np.max(bin_heights_peak), bin_centers_peak[np.argmax(bin_heights_peak)], 0.5*expgain[figctr]], maxfev=100000, bounds=([0, xmin_fit, 0.01*expgain[figctr]], [1.5*np.max(bin_heights_peak), xmax_fit, 5*expgain[figctr]]))	# Fit histogram with gaussian
						nsampGain=int(stringval)
						popt, pcov = curve_fit(gaussian, bin_centers_peak, bin_heights_peak, p0=[np.max(bin_heights_peak), bin_centers_peak[np.argmax(bin_heights_peak)], 0.5*nsampGain], maxfev=100000, bounds=([0, xmin_fit, 0.01*nsampGain], [1.5*np.max(bin_heights_peak), xmax_fit, 5*nsampGain]))	# Fit histogram with gaussian
#						print(popt)
						axs_all[figctr].plot(bin_centers_peak, gaussian(bin_centers_peak, *popt))			# Plot gaussian fit
						par_fit = np.append(aux_arr, popt)
						aux_arr = par_fit
						print("Successful fit :)")
					except: print("ERROR in fit :(")

				try:	# Enters here if numpeaks>=2
					norm = par_fit[0]; offset = par_fit[1]; gain = par_fit[4]-par_fit[1]; noise = par_fit[2]/gain; mu = par_fit[3]/par_fit[0];
					print("Trying complex fit with initial parameters:")
					print(norm, offset, gain, noise, mu)

					bin_heights_doublegaus = bin_heights[(bin_centers>(offset_fit-0.5*expgain[figctr])) & (bin_centers<(offset_fit+1.5*expgain[figctr]))]
					bin_centers_doublegaus = bin_centers[(bin_centers>(offset_fit-0.5*expgain[figctr])) & (bin_centers<(offset_fit+1.5*expgain[figctr]))]
					popt, pcov = curve_fit(doublegaus, bin_centers_doublegaus, bin_heights_doublegaus, p0=[norm, offset, noise, gain, mu], maxfev=100000, bounds=([0, offset_fit-0.5*expgain[figctr], 0.001, 0, 0], [np.inf, offset_fit+1.5*expgain[figctr], np.inf, 2*expgain[figctr], np.inf]))		# Fit histogram with double gaussian
					axs_all[figctr].plot(bin_centers_doublegaus, doublegaus(bin_centers_doublegaus, *popt))		# Plot gaussian fit
#					popt, pcov = curve_fit(sumgaus, bin_centers_doublegaus, bin_heights_doublegaus, p0=[norm, offset, noise, gain, mu, 2], maxfev=100000)	# Fit histogram with sum of gaussians
#					axs_all[figctr].plot(bin_centers_doublegaus, sumgaus(bin_centers_doublegaus, *popt))							# Plot gaussian fit

#					var_fit.append(abs(popt[2]*popt[3])/float(stringval))
					var_fit.append(popt[parplot-1])
					print("Successful fit :)"); print("Extracting "+varsplot[parplot-1])
				except:
					try:	# Enters here if numpeaks=1, try to extract parameters from gaussian fit to 0e- peak
						norm = par_fit[0]; offset = par_fit[1]; noise = par_fit[2]/expgain[figctr];
						if parplot == 1: var_fit.append(norm); print("Extracting "+varsplot[parplot-1])
						if parplot == 2: var_fit.append(offset); print("Extracting "+varsplot[parplot-1])
						if parplot == 3: var_fit.append(noise); print("Extracting "+varsplot[parplot-1])
						else: var_fit.append(0); print("ERROR extracting "+varsplot[parplot-1])
					except: var_fit.append(0); print("ERROR in fit :(")

#				var.append(np.std(data[overscan_mask])/float(stringval))	# Standard deviation
				var.append(np.std(data[overscan_mask]))				# Standard deviation
#				var.append(np.mean(data[overscan_mask]))			# Mean
#				var.append(np.median(data[overscan_mask]))			# Median

				figctr=figctr+1
		#fig_all.suptitle(image)
		plt.close(fig_all)
		#plt.show()			# Show histogram per image
		

		# STORE COMPUTED VARIABLE
		print("\nNoise in overscan from stddev [ADU]:")
		print(var[0], var[1], var[2], var[3])
		print(varsplot[parplot-1]+" in selected area:")
		print(var_fit[0], var_fit[1], var_fit[2], var_fit[3]) #add to Dataframe

		#valuesDict[header['RUNID']]=[int(header['NSAMP']),round(var_fit[0], 4), round(var_fit[1], 4), round(var_fit[2],4), round(var_fit[3],4)]
		valuesDict[header['RUNID']]=[int(stringval),round(var_fit[0], 4), round(var_fit[1], 4), round(var_fit[2],4), round(var_fit[3],4)]
		#valuesDict[imgNumber]=[int(stringval),round(var_fit[0], 4), round(var_fit[1], 4), round(var_fit[2],4), round(var_fit[3],4)]

		

		#list_var.append([float(stringval), var[0], var[1], var[2], var[3]])
		list_var.append([float(stringval), var_fit[0], var_fit[1], var_fit[2], var_fit[3]])		# To use the var obtained from the gaussian fit

		# deltaH.append(float(header["H1AH"])-float(header["H1AL"]))
		# deltaV.append(float(header["V1AH"])-float(header["V1AL"]))
		# deltaT.append(float(header["TGAH"])-float(header["TGAL"]))
		# deltaSW.append(float(header["SWAH"])-float(header["SWAL"]))
		# try:
		# 	RunID.append(float(header["RUNID"]))
			
		# except:
		# 	#RunID.append(float(hlabel.split("_")[-1].split(".")[0]))
		# 	vFile.append(float(image.split('sinit_')[1].split('_')[0]))

		if len(files) > 1:
			dataframe=pd.DataFrame.from_dict(valuesDict, columns=['delay','Ext 0', '1', '2', '3'], orient='index')
	arr_var=np.array(list_var)
	arr_var=arr_var[np.argsort(arr_var[:, 0])]	# Sort array by values on first column
#	print(arr_var)

	#fig_all.legend(handles_all, labels_all, loc='upper right')
	#plt.legend()
	if len(files)>1:
		print('\n')
		print(varsplot[parplot-1]+" for every image in selected area:")
		#print(dataframe.sort_index())
		print(dataframe.sort_values(by='delay'))
		
		
	
	#plt.savefig(dirname+"/checknoise.png")
	plt.show()
	

	# PLOT
	fig_var, axs_var = plt.subplots(1, 4, figsize=(20, 5))
#	fig_var.tight_layout()

	for k in range(0, 4):
		axs_var[k].plot([row[0] for row in arr_var], [(row[k+1]) for row in arr_var], ".k")
		#axs_var[k].plot([row[0] for row in arr_var], sqrt([row[0] for row in arr_var], arr_var[0, k+1]), "-r")		# Sqrt fit when doing noise vs nsamp
		axs_var[k].set_title('ext '+str(k+1))
		axs_var[k].set_xlabel(string)
		axs_var[k].set_ylabel(varsplot[parplot-1])
		axs_var[k].grid(True)
#		axs_var[k].set_xscale('log')
#		axs_var[k].set_yscale('log')
#		axs_var[k].set_ylim(ymin=1)

# 	#axis=vFile
# 	axis=RunID	
# 	axs_var[1][0].plot(axis,deltaV,".k")
# 	axs_var[1][0].set_title("Delta V")
# 	axs_var[1][0].set_ylabel("Volts")
# 	axs_var[1][0].set_xlabel("NSAMP")
# 	axs_var[1][1].plot(axis,deltaT,".k")
# 	axs_var[1][1].set_title("Delta T")
# 	axs_var[1][1].set_xlabel("NSAMP")
# 	axs_var[1][2].plot(axis,deltaH,".k")
# 	axs_var[1][2].set_title("Delta H")
# 	axs_var[1][2].set_xlabel("NSAMP")
# 	axs_var[1][3].plot(axis,deltaSW,".k")
# 	axs_var[1][3].set_title("Delta SW")
# 	axs_var[1][3].set_xlabel("NSAMP")

	
	plt.savefig(dirname+"/checknoise.png")	# Save plot
	plt.show()				# Show plot
	
	

	# TXT FILE
#	np.savetxt(datafile, arr_var, fmt="%s")	# Save array to datafile
#	datafile.close()			# Close datafile'

else:
	print("To run do: python3 checknoise.py path/img*.fits")
