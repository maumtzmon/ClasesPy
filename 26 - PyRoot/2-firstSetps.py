#!/snap/bin/pyroot
import ROOT



h = ROOT.TH1F("myHist", "myTitle", 64, -4, 4)
h.FillRandom("gaus")

h.Draw()