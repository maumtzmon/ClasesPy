void tut3()
{
	TH1F *hist = new TH1F("hist","Histogram",6,1,10);
	
	fstream file;
	file.open("data_tut3.txt",ios::in);
	
	double value;
	
		
	while(1)
	{
		file >> value;
		hist->Fill(value);
		if(file.eof())break;
	}
	
	file.close();
	
	hist->GetXaxis()->SetTitle("Grade");
	hist->GetXaxis()->SetTitle("Entries");
	
	TCanvas *c1 = new TCanvas();
	hist->Draw();
}
