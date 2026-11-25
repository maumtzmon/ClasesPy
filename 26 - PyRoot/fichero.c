// https://root-forum.cern.ch/t/passing-a-filename-as-an-argument-to-a-macro/3255

#include <iostream>

void fichero(const char *archivo)
// void fichero()
{
	ifstream inFile;
	inFile.open(archivo);
	
	Int_t lineCount = 0;
	while (inFile.getline(), archivo)) {  // Loop through each line in the file
      		lineCount++;  // Incrementing line count for each line read
      	}
	
	inFile.close();      	
	Int_t NBins = lineCount;
	const int NBinsPlus1 = NBins + 1;
	Double_t    XBins[NBinsPlus1]; 
	Double_t    BinCont[NBinsPlus1];
			
			// ifstream inFile;
			// inFile.open(archivo);
			// // inFile.open("/home/oem/datosFits/histogram_nsamp_100.csv");
			// double val1, val2;
			// char del;
			// int ibin = 0;
			// while ( (inFile >> val1 >> del >> val2) && (del==',') )
			// {
			// //printf("%4d: %2.16f ,%2.19f\n",ibin,val1,val2);
			// XBins[ibin]   = val1;
			// BinCont[ibin] = val2;
			// ibin++;
			// } //while inFile
			
			// // cout << "bines: "<< endl;
			// TH1F  *histo = new TH1F("histo","",NBins,XBins);
			// //for (int ibin=0;ibin<NBins;ibin++) histo->Fill(XBins[ibin]);
			// //histo->Draw();
			// //histo->GetXaxis()->SetTitle("charge (e^{-})");
			// //histo->SetLineWidth(3);
			// //histo->SetLineColor(4);
			// for (int ibin = 0; ibin<NBins; ibin++){
			// histo->SetBinContent(ibin,BinCont[ibin]);
			// }
			// histo->Draw();
			// //double nevents = histo->Integral();
			// //cout <<"events: "<< nevents  << endl;
}
