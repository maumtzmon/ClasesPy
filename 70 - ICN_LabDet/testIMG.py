from astropy.io import fits
import matplotlib.pyplot as plt

hdul = fits.open('/home/oem/datosFits/MicrochipTest_Marzo/datos/22MAY23/proc_skp_m-009_microchip_vTested_T_170__seq_HA_NSAMP_324_NROW_1300_NCOL_1200_EXPOSURE_0_NBINROW_1_NBINCOL_1_img_096.fits')

print('hola')

data=hdul[3].data

print("")

plt.figure(figsize=(20,10))
plt.imshow(data, vmin=40000, vmax=80000)
plt.show()
