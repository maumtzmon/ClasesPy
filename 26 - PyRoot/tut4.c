//#include <iostream>

void tut4(char* FileArg)
{
	//cout << FileArg;
	
	TGraph *gr  = new TGraph();
	 
	fstream file;
	file.open(FileArg, ios::in);
	//file.open("data_tut4.csv", ios::in);
	
	while(1)
	{
		double x,y;
		file >>x >> y;
		gr->SetPoint(gr->GetN(),x,y);
		if(file.eof()) break;
	}
	
	file.close();
	
	
	
	TCanvas *c1 = new TCanvas();
	gr->SetFillColor(40);
	gr->Draw("AB");
	
}
