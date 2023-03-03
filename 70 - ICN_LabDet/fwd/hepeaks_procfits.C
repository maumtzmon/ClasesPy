//-----This script generates a histogram per extension
//-----from an averaged FITS image ("proc_img" generated from skipper2root)
//-----and stores them in a root file (*_hepeaks.root)

//-----Example to run: .x hepeaks_procfits.C ("file")--------

void hepeaks_procfits (char const* file){

//--------IMPORTANT VARIABLES--------

  const int numext = 4;		// Number of extensions
//  const int numext = 16;		// Number of extensions

//  int colmin = 2;	//10
  int colmin = 10;	//540	// Lowest column defining image region of interest
//  int colmax = 108;	//538
  int colmax = 538;		// Highest column defining image region of interest
  int rowmin = 1;		// Lowest row defining image region of interest
  int rowmax = 1000;		// Highest row defining image region of interest

  int nbin = 200;		// Number of bins
  float pixmin = -500;
  float pixmax = 1000;
//  float pixmin = 180000;		// pixmin
//  float pixmax = 240000;		// pixmax

//--------Style--------
  gROOT->Reset();
  gROOT->SetStyle("Plain");
  gStyle->SetOptStat(0);	// Get rid of the stats box in the graphs
  gStyle->SetPalette(1);	// Default = 1 = Rainbow palette
  gStyle->SetOptFit(1);		// SetOptFit(pcev); p-probability, c-chisquare, e-errors, v-valuesofparameters; Default = 1 = 0111

  gStyle->SetTitleFont(132);
  gStyle->SetTitleFont(132, "Y");
  gStyle->SetLabelFont(132);	//  (textfontcode precision); 13 = times-medium-r-normal
  gStyle->SetLabelFont(132, "Y");
  gStyle->SetTitleSize(0.04);
  gStyle->SetTitleSize(0.04, "Y");
  gStyle->SetLabelSize(0.035);
  gStyle->SetLabelSize(0.035, "Y");
  gStyle->SetTitleOffset(1.1);
  gStyle->SetTitleOffset(1.3, "Y");

  gStyle->SetTextSize(0.035);
  gStyle->SetTextFont(132);

  gStyle->SetStatFont(132);
  gStyle->SetStatH(0.1);
  gStyle->SetStatX(0.9);
  gStyle->SetStatY(1.);

  gStyle->SetLegendFont(132);
  gStyle->SetLegendBorderSize(0);

  TGaxis::SetMaxDigits(3);

//--------Define histograms per extension--------

  TH1F *hpix[numext];

  for (int next=0; next<numext; next++){
	hpix[next] = new TH1F(Form("ext%i",next+1), Form("ext%i",next+1), nbin, pixmin, pixmax);
	hpix[next]->Sumw2();
	}

//--------Main loop to open each extension of FITS file and to fill histograms--------

 for (int next=0; next<numext; next++) {

	int ptctr=0;
	float mean=0.;

	//-----Open FITS file-----

 	TFITSHDU *hdu = new TFITSHDU(Form("%s", file), next);
 	if (hdu == 0){
		printf("ERROR: could not access the HDU\n"); return;
   		}
	printf("File successfully open!\n");
//	hdu->Print("");

	//-----Read FITS file as a matrix-----

	TMatrixD *mat = hdu->ReadAsMatrix(0);
//	mat->Print();
	int cols = mat->GetNcols();
	int rows = mat->GetNrows();

	//-----Loop over each row and column-----

	for (int i=0; i<rows; i++){
		if ((i+1)>=rowmin && (i+1)<=rowmax){
			for (int j=0; j<cols; j++){
				if ((j+1)>=colmin && (j+1)<=colmax && mat(i,j)>pixmin && mat(i,j)<pixmax){
//					printf("\n%f", mat(i,j));
					hpix[next]->Fill(mat(i,j));	// Fill histograms
					}
				}
			}
		}

	}


//--------Create the file to store histograms and the canvas to draw them--------

  char fileroot[] = file;
  int length = strlen(fileroot);
  fileroot[length-5] = '\0';		// Throw ".fits" from filename (last 5 characters)

  TFile *histograms = new TFile(Form("%s_hepeaks.root", fileroot), "RECREATE");

//  TCanvas *c = new TCanvas("c","Fit canvas", 500*numext, 500);
//  c->Divide(numext, 1);
  TCanvas *c = new TCanvas("c","Fit canvas", 2000, 500*ceil(numext/4.));
  c->Divide(4, ceil(numext/4.));
  int ctr = 0;

//--------Loop for storing and drawing histograms--------

  for (next=0; next<numext; next++){
	ctr = ctr+1;
	c->cd(ctr);
//	gPad->SetLogy();
	hpix[next]->Draw();
	hpix[next]->Write();
	}

//--------Close and write files--------

  histograms->Close();

  c->Print(Form("%s_hepeaks.pdf", fileroot));

}
