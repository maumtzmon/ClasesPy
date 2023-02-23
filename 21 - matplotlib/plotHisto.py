import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/oem/Software/LTA_Automatic_Read')

from  fits import histogram



#file="/home/maumtz/datosFits/testMITLL/16NOV22/proc_skp_module26_MITLL01_externalVr_Vtest_T170_testLeakage__NSAMP1_NROW650_NCOL700_EXPOSURE0_NBINROW1_NBINCOL1_img34.fits"
file="/home/maumtz/datosFits/testMITLL/16NOV22/proc_skp_module26_MITLL01_externalVr_Vtest_T170_testLeakage__NSAMP324_NROW50_NCOL700_EXPOSURE0_NBINROW1_NBINCOL1_img33.fits"

class argv:
    def __init__(self,histogram='',x=[100, 200, 300, 400], y=[100, 200, 300, 400],baseline=False, charge=None, dCurrent=None, eventDet=None, ext=None, pdf=True,):
        self.histogram = file
        self.x = x
        
        self.y = y
        self.baseline = baseline
        self.ext=ext
        self.pdf = pdf

x=[100, 200, 300, 400]
y=[100, 200, 300, 400]

ext=[0]

#argObj=argv(file, [100, 200, 600, 650], [100, 200, 600, 650], False, None, None, None, ext, False)
argObj=argv(file, [1, 24, 25, 49],[1, 250,450, 699], False, None, None, None, ext, False)
histogram(argObj)