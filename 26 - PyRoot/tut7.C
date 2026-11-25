void tut7()
{
	TH1F *hist = new TH1F("hist","Histogram",90,0,1000);
	TRandom2 *rand = new TRandom2(50);
	
	fstream file;
	file.open("data.txt",ios::out);
	
	for(int j=0; j<3; j++){
		for(int i = 0; i<300; i++)
		{
			double r =rand->Gaus((j+1)*250,60);
			file << r << endl;
		}
	}	
	
	file.close();
	
	file.open("data.txt", ios::in);
	
	double value;
	
	while(1)
	{
		file >> value;
		hist->Fill(value);
		if(file.eof()) break;
	}
	
	file.close();
	
	hist->GetXaxis()->SetTitle("Distribution");
	hist->GetYaxis()->SetTitle("Entries");
	
	TF1 *fit =new TF1("fit","gaus", 150,350); // Range from 150 to 350
	
	fit->SetParameter(0,40); // max value in Y axis, parameter 0
	fit->SetParameter(1,250);  // mean value initial aproximation, parameter 1
	fit->SetParameter(2,60);	 // std Dev, parameter 2
	
	
	TCanvas *c1 = new TCanvas();
	hist->Draw();
	hist->Fit("fit","R");
	
	double mean = fit->GetParameter(1);
	double sigma = fit->GetParameter(2);
	double error = fit->GetParError(1);

	cout << "mean= " << mean << ";   error=" << error << endl;
	cout << "sigma=" << sigma << endl;
	
	double pars[3]; 
	fit->GetParameters(&pars[0]);  // Almacena los datos del fit, comenzando del indice 0 en pars
	
	cout << "pars[0] Max value= " << pars[0] << endl;
	cout << "pars[1] Mean value= " << pars[1] << endl;
	cout << "pars[2] Sigma Value= " << pars[2] << endl;
}
