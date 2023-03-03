//-----This script uses the root file (*_hepeaks.root) generated from hepeaks_*.C
//-----and performs a fit to the electron peaks

//--------Define fitting function using convolution of Poisson and Gaussian--------

double myfunc(double *xcoor, double *par){
  double xx = xcoor[0];
  double lambda = par[0];
  double norm = par[1];
  double gain = par[2];
  double noise = par[3];
  double offset = par[4];
  double f = 0;

  //-----Calculate the Poisson distribution with mean = lambda-----
  double p[3];
  p[0]=exp(-lambda);
  for (int i=1; i<3; i++){
	p[i]=lambda*p[i-1]/i;
	}

  //-----Calculate the Gaussian-----
  for (int i=0; i<3; i++){
	f=f+p[i]*exp(-pow((((xx-offset)/gain)-i)/noise,2)/2.)/(gain*noise*sqrt(2*TMath::Pi()));
	}
  f=f*norm;

/*  for (int i=0; i<3; i++){
	f=norm*(TMath::Gaus(xx,offset+i*gain,noise*gain,1)*TMath::Poisson(i,lambda))
	} */

  return f;
}

void noisedc (char const* file){

//--------VARIABLES TO FIT HISTOGRAMS--------

  const int numpeaks = 2;		// Number of peaks to fit starting from the 0e peak (numpeaks=2 fits the 0e and the 1e peak)
  const int numext = 4;			// Number of working extensions
  int goodext[numext] = {1, 2, 3, 4};
//  int goodext[numext] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
//  float expgain[numext] = {1.2, 1.2, 1.2, 1};			// Expected gain in ADU/e-
//  float expgain[numext] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};			// Expected gain in ADU/e-
  float expgain[numext] = {200, 200, 200, 200};			// Expected gain in ADU/e-
//  float expgain[numext] = {600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600};			// Expected gain in ADU/e-
//  float expgain[numext] = {400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400};			// Expected gain in ADU/e-
  const int fitopt = 2;			// 1 for fitting with 2 gaussians, 2 for fitting with convolution

//--------Style--------

  gROOT->Reset();
  gROOT->SetStyle("Plain");
  gStyle->SetOptStat(0);	// Get rid of the stats box in the graphs
  gStyle->SetPalette(1);	// Default = 1 = Rainbow palette
  gStyle->SetOptFit(1);		// SetOptFit(pcev); p-probability, c-chisquare, e-errors, v-valuesofparameters; Default = 1 = 0111

  gStyle->SetTitleFont(132);
  gStyle->SetTitleFont(132, "Y");
  gStyle->SetLabelFont(132);	// (textfontcode precision); 13 = times-medium-r-normal
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

//--------Retrieve histograms from the root file--------

  TFile filehist(Form("%s", file));

  TH1F *hpix[numext];

  for (int next=0; next<numext; next++){
	hpix[next] = (TH1F*)filehist.Get(Form("ext%i", goodext[next]));
	hpix[next]->SetDirectory(0);
	hpix[next]->GetXaxis()->SetTitle("ADU");
	hpix[next]->GetYaxis()->SetTitle("Number of pixels");
	}

  filehist.Close();

//--------Retrieve histograms parameters--------

  float nbin = (hpix[0]->GetSize())-2.;
  float pixmin = hpix[0]->GetXaxis()->GetXmin();
  float pixmax = hpix[0]->GetXaxis()->GetXmax();

  printf("\nnbin = %f, coormin = %f, coormax = %f\n", nbin, pixmin, pixmax);

//--------Define cloned histograms to fit--------

  TH1F *hpix1[numext];
  for (int next=0; next<numext; next++){
	hpix1[next] = (TH1F*)hpix[next]->Clone(Form("hpix1[%i]", next));
	delete hpix[next];
	}

//--------Find peaks to fit and fit them--------

  TF1 *fitfun[numpeaks];
  int fitfunstatus[numext][numpeaks];
  double pars[numpeaks*3], epars[numpeaks*3], sigma[numext];

  TF1 *f2gaus[numext];

  int fitstatus[numext];
  double gain[numext], offset[numext], lambda[numext], noise[numext], norm[numext];
  double egain[numext], eoffset[numext], elambda[numext], enoise[numext], enorm[numext];
  double gain_fit[numext], noise_fit[numext], lambda_fit[numext], offset_fit[numext], norm_fit[numext];
  double egain_fit[numext], enoise_fit[numext], elambda_fit[numext], eoffset_fit[numext], enorm_fit[numext];

  for (int next=0; next<numext; next++){			// Loop of exts

//	offset[next] = hpix1[next]->GetXaxis()->GetBinCenter(hpix1[next]->GetMaximumBin());
	offset[next]=0;

	//--------Gaussian fit of e- peaks--------

	for (int npeak=0; npeak<numpeaks; npeak++){
		float xmin = (npeak*expgain[next])-(expgain[next]*0.3)+offset[next];
		float xmax = (npeak*expgain[next])+(expgain[next]*0.3)+offset[next];
		printf("xmin%f xmax%f", xmin, xmax);
		int binmin = hpix1[next]->GetXaxis()->FindBin(xmin);
		int binmax = hpix1[next]->GetXaxis()->FindBin(xmax);

		hpix1[next]->GetXaxis()->SetRange(binmin, binmax);
		float norm0 = hpix1[next]->GetMaximum();
		float offset0 = hpix1[next]->GetXaxis()->GetBinCenter(hpix1[next]->GetMaximumBin());
		float sigma0 = 0.5*expgain[next];

		fitfun[npeak] = new TF1 (Form("f%i_ext%i", npeak, next+1), "gaus", xmin, xmax);
		fitfun[npeak]->SetLineColor(2);
		fitfun[npeak]->SetParameters(norm0, offset0, sigma0);
//		fitfun[npeak]->SetParLimits(0, 0, 1.5*norm0);
//		fitfun[npeak]->SetParLimits(1, 2*xmin, 2*xmax);
//		fitfun[npeak]->SetParLimits(2, 0, 2*expgain[next]);

		fitfunstatus[next][npeak] = hpix1[next]->Fit(fitfun[npeak], "R+");
		fitfun[npeak]->GetParameters(&pars[npeak*3]);
		epars[0+3*npeak]=fitfun[npeak]->GetParError(0+3*npeak);
		epars[1+3*npeak]=fitfun[npeak]->GetParError(1+3*npeak);
		epars[2+3*npeak]=fitfun[npeak]->GetParError(2+3*npeak);
	
		printf("\next %i: %ie- peak ---> constant: %f | mean: %f | sigma: %f\n \n", goodext[next], npeak, pars[0+(npeak*3)], pars[1+(npeak*3)], pars[2+(npeak*3)]);
		}

	hpix1[next]->GetXaxis()->SetRange();

	norm[next] = pars[0];
	offset[next] = pars[1];
	sigma[next] = pars[2];		// Noise in ADU
	enorm[next] = epars[0];
	eoffset[next] = epars[1];

	//--------Do fit routine if 0e- and 1e- peaks converged--------

	if (numpeaks>=2 && fitfunstatus[next][0]==0 && fitfunstatus[next][1]==0){
//		gain[next] = pars[4]-pars[1];
		gain[next] = expgain[next];
		lambda[next] = pars[3]/pars[0];
		noise[next] = pars[2]/gain[next];
		egain[next] = TMath::Power(TMath::Power(epars[4], 2)+TMath::Power(epars[1], 2), 0.5);
//		elambda[next] = TMath::Power(TMath::Power(epars[3]*pars[0], 2)+TMath::Power(pars[3]*epars[0], 2), 0.5);
		enoise[next] = epars[2]/gain[next];

		printf("Initial values:");
		printf("\ngain: %f ADU/e- | noise: %f e- | lambda: %f e-/pix | offset: %f ADU | norm: %f\n \n", gain[next], noise[next], lambda[next], offset[next], norm[next]);

		if (fitopt==1){
			f2gaus[next] = new TF1(Form("f2gaus_ext%i", next+1), "gaus(0) + gaus(3)", offset[next]-(gain[next]/4), gain[next]+(gain[next]/3)+offset[next]);
			f2gaus[next]->SetParNames("Constant 1", "Mean 1", "Sigma 1", "Constant 2", "Mean 2", "Sigma 2");
			f2gaus[next]->SetParameters(pars[0], pars[1], pars[2], pars[3], pars[4], pars[5]);
//			f2gaus[next]->SetParLimits(4, 0, gain[next]);
//			f2gaus[next]->SetParLimits(4, offset[next]+0.5*gain[next], offset[next]+1.5*gain[next]);
	
			fitstatus[next] = hpix1[next]->Fit(f2gaus[next], "R");
	
			gain_fit[next] = (f2gaus[next]->GetParameter(4))-(f2gaus[next]->GetParameter(1));
			noise_fit[next] = (f2gaus[next]->GetParameter(2))/gain_fit[next];
			lambda_fit[next] = (f2gaus[next]->GetParameter(3))/(f2gaus[next]->GetParameter(0));
			offset_fit[next] = f2gaus[next]->GetParameter(1);

			egain_fit[next] = TMath::Power(TMath::Power(f2gaus[next]->GetParError(4), 2)+TMath::Power(f2gaus[next]->GetParError(1), 2), 0.5);
			enoise_fit[next] = (f2gaus[next]->GetParError(2))/gain_fit[next];
//			elambda_fit[next] = TMath::Power(TMath::Power((f2gaus[next]->GetParError(3))*(f2gaus[next]->GetParameter(0)), 2)+TMath::Power((f2gaus[next]->GetParameter(3))*(f2gaus[next]->GetParError(0)), 2), 0.5);
			eoffset_fit[next] = f2gaus[next]->GetParError(1);

			printf("\nFitted values:");
			printf("\ngain: %f ADU/e- | noise: %f e- | lambda: %f e-/pix | offset: %f ADU\n \n", gain_fit[next], noise_fit[next], lambda_fit[next], offset_fit[next]);
			}

		if (fitopt==2){	
			TF1 *func = new TF1("func", myfunc, pixmin, pixmax, 5);		// Create ROOT function based on 'myfunc'
			func->SetParameters(lambda[next], norm[next], gain[next], noise[next], offset[next]);
			func->SetParNames("lambda","norm","gain","noise","offset");
			func->SetParLimits(0, 0, 1);
//			func->SetParLimits(2, 0.5*gain[next], 1.5*gain[next]);
//			func->SetParLimits(3, 0, gain[next]);
//			func->SetParLimits(4, -0.5*gain[next], 0.5*gain[next]);
			func->SetLineColor(2);
			func->SetLineWidth(1);

			fitstatus[next] = hpix1[next]->Fit("func", "R");
	
			lambda_fit[next] = func->GetParameter(0);
			norm_fit[next] = func->GetParameter(1);
			gain_fit[next] = func->GetParameter(2);
			noise_fit[next] = func->GetParameter(3);
			offset_fit[next] = func->GetParameter(4);
		
			elambda_fit[next] = func->GetParError(0);
			enorm_fit[next] = func->GetParError(1);
			egain_fit[next] = func->GetParError(2);
			enoise_fit[next] = func->GetParError(3);
			eoffset_fit[next] = func->GetParError(4);

			printf("\nFitted values:");
			printf("\ngain: %f ADU/e- | noise: %f e- | lambda: %f e-/pix | offset: %f ADU | norm: %f\n \n", gain_fit[next], noise_fit[next], lambda_fit[next], offset_fit[next], norm_fit[next]);
			}
		}
	}			// End of loop of exts

//--------Retrieve filename without extension--------

  char fileroot[] = file;
  int length = strlen(fileroot);
  fileroot[length-5] = '\0';		// Throw ".root" from filename (last 5 characters)

//--------Draw fitted histograms--------

  TCanvas *c = new TCanvas("c","Fit canvas", 2000, 500*ceil(numext/4.));
  c->Divide(4, ceil(numext/4.));
//  TCanvas *c = new TCanvas("c","Fit canvas", 500*numext, 500);
//  c->Divide(numext, 1);
  int ctr = 0;

  for (int next=0; next<numext; next++){
	ctr = ctr+1;
	c->cd(ctr);
	gPad->SetLogy();
	hpix1[next]->Draw();
	}

  c->Print(Form("%s_noisedc.pdf", fileroot));

//--------Print fits info--------

  char* strimg = "_img";
  char *imgID = strstr(file, strimg);
  imgID=imgID+strlen(strimg);
  strtok(imgID, "_");

  char* string = "_img";
  char *var = strstr(fileroot, string);	// Retrieve substring of string
  var=var+strlen(string);		// Remove characters correspondent to string
  strtok(var, "_");			// Retrieve string before delimiter

  if (numpeaks==1){
	printf("\n#ImgID\t%s\tExt\tNoise (ADU)\n", string);
	}
  if (numpeaks>=2){
	printf("\n#ImgID\t%s\tExt\tNoise (e-)\tGain (ADU/e-)\tLambda (e-/pix)\teNoise (e-)\teGain (ADU/e-)\teLambda (e-/pix)\n", string);
	}

  for (int next=0; next<numext; next++){			// Loop of exts
//	for (int ngood=0; ngood<numgoodext; ngood++){		// Loop of good exts
//		if (next+1==goodext[next]){ 			// Exclude bad exts
			if (numpeaks==1){
				if (fitfunstatus[next][0]==0){
					printf("%s\t%s\t%i\t%f\n", imgID, var, goodext[next], sigma[next]);
					}
				else {
					printf("%s\t%s\t%i\t0\n", imgID, var, goodext[next]);
					}
				}
			if (numpeaks>=2){
				if (fitstatus[next]==0){
					printf("%s\t%s\t%i\t%f\t%f\t%f\t%f\t%f\t%f\n", imgID, var, goodext[next], noise_fit[next], gain_fit[next], lambda_fit[next], enoise_fit[next], egain_fit[next], elambda_fit[next]);
					}
				else if (fitfunstatus[next][0]==0 && fitfunstatus[next][1]==0){
					printf("%s\t%s\t%i\t%f\t%f\t%f\t%f\t%f\t%f\n", imgID, var, goodext[next], noise[next], gain[next], lambda[next], enoise[next], egain[next], elambda[next]);
					}
				else {
					printf("%s\t%s\t%i\t0\t0\t0\n", imgID, var, goodext[next]);
					}
				}
//			}
//		}
	}


/*//OLD CODE
  ofstream datafile;
  datafile.open("noisedc.txt", ofstream::app);
  datafile << "Ext\tPeak\tNorm\tMean\tSigma\n";

  for (int next=0; next<numext; next++){
//	if (next!=0 && next!=1){		//---Exclude bad ohdus
		for (int npeak=0; npeak<numpeaks; npeak++){
			if (datafile.is_open()) {
			 	datafile << next+1 << "\t" << npeak << "\t" << norm[next][npeak] << "\t" << mean[next][npeak] << "\t" << sigma[next][npeak] << "\n";
				}
			}
//		}
	}

  datafile << "\nExt\tGain (ADU/e-)\tNoise (e-)\tLambda (e-)\tOffset (ADU/e-)\tNorm\n";

  for (int next=0; next<numext; next++){
//	if (next!=0 && next!=1){		//---Exclude bad ohdus
		if (datafile.is_open()) {
			datafile << next+1 << "\t" << gain_fit[next] << "\t" << noise_fit[next] << "\t" << lambda_fit[next] << "\t" << offset_fit[next] << "\t" << norm_fit[next] << "\n";
			}
//		}
	}

  datafile.close();
*/
}
