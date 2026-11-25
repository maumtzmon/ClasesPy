#include <math.h>

double multGaussians(double *x, double *par){
	double q 	 =	x[0];  // valores de pixel
	double norm  =  par[0];
	double mu    =  par[1];
	double sigma =  par[2];
	double gain  =  par[3];
	double con   =  par[4];
	double val=0;

	double cPi = TMath::Pi();

	for(int w=0; w<3; w++){
		val+=(norm/sqrt(2*cPi*pow(sigma,2)))*(exp(-0.5*pow(q-mu-w*gain,2)/pow(sigma,2)));
	}
	val+=con;
	return val;
}

	//  pow(x,2), elevar al cuadrado
	//  sqrt(x), raiz cuadrada
	//  a*exp(b*x), funcion exponencial

void tut7_3peaksFit()
{
	TRandom2 *rand = new TRandom2(50);
	
	fstream file;
	file.open("data.txt",ios::out);
	
	
	file.close();
	
	//Crear histograma TH1F (aqui esta vacio, se llena despues)
	double loq = 0;
	double hiq = 1000;
	int    Nbins = 130;
	double bw = (hiq-loq)/Nbins;
	TH1F *hist = new TH1F("hist","Histogram",Nbins,loq,hiq); //Aqui se genera el histograma, #bines, rango de valores


	//se crea un TF1 a partir de la funcion multigausian
	//se establecen parametros iniciales ṕara llenar el histograma *hist
	int Npar = 5;
	TF1 *func = new TF1("func", multGaussians,loq,hiq,Npar); // Npar = numero de parametros de la funcion multiGaussians necesita 5 parametros
	func->SetParameter(0,1);   // establecer parametros iniciales (#parametro, valor del parametro)
	func->SetParameter(1,250);
	func->SetParameter(2,60);
	func->SetParameter(3,250);
	func->SetParameter(4,0.001);
	double integ0 = func->Integral(-2*hiq,2*hiq); // checar que es
	cout << "Integ0 =" << integ0 << endl;


	int Nevents = 50000;
    for(int i=0; i<Nevents; i++)
		{
		//se lena el histograma
	 	double r = func->GetRandom(); // aqui se genera el histograma con la funcion multigaussians
		hist->Fill(r);
	 	}



	//Initial values for the Fit		
	func->SetParameter(0,23000);
	func->SetParameter(1,250);
	func->SetParameter(2,60);
	func->SetParameter(3,250);
	func->SetParameter(4,10);
	
	TCanvas *c0 = new TCanvas();
	hist->Fit("func","R");
	hist->Draw("same");
	
	// Do Fit
	double norm  = func->GetParameter(0);
	double mean  = func->GetParameter(1);
	double sigma = func->GetParameter(2);
	double gain  = func->GetParameter(3);
	double con   = func->GetParameter(4);

	double norme  = func->GetParError(0);
	double meane  = func->GetParError(1);
	double sigmae = func->GetParError(2);
	double gaine  = func->GetParError(3);
	double cone   = func->GetParError(4);

	printf("Nevt per Gaussian = %6.1f +- %6.1f \n",norm/bw, norme/bw);


	TF1 *fcte = new TF1("fcte", multGaussians,loq,hiq,Npar);
	fcte->SetLineStyle(2);
	fcte->SetParameter(0,0*norm); // Valor constante entre 0 y 1
	fcte->SetParameter(1,mean);
	fcte->SetParameter(2,sigma);
	fcte->SetParameter(3,gain);
	fcte->SetParameter(4,con);


	//--Drawing section 

	fcte->Draw("same");
	


	// TH1F *frame = new TH1F("frame", "", 100, loq, hiq);
	// frame->SetMaximum(30);




	

	// file.open("data.txt", ios::in);
	
	// double value;
	
	// while(1)
	// {
	// 	file >> value;
	// 	hist->Fill(value);
	// 	if(file.eof()) break;
	// }
	
	// file.close();
	
	// hist->GetXaxis()->SetTitle("Distribution");
	// hist->GetYaxis()->SetTitle("Entries");
	
	// //TF1 *fit =new TF1("fit","gaus", 150,350);
	// TF1 *fit = new TF1("fit", multGaussians, 400, 1100,5); //rango de las gaussianas y #parametros
	

	
	// fit->SetParameter(0,40); // max value in Y axis, parameter 0
	// fit->SetParameter(1,250);  // mean value initial aproximation, parameter 1 mu
	// fit->SetParameter(2,60);	 // std Dev, parameter 2
	// fit->SetParameter(3,250); // initial gain
	// fit->SetParameter(4,0);	//initial const
	

	
	// TCanvas *c1 = new TCanvas();
	// hist->Draw();
	// hist->Fit("fit","R");
	
	// double mean = fit->GetParameter(1);
	// double sigma = fit->GetParameter(2);
	// double error = fit->GetParError(1);

	// cout << "mean= " << mean << ";   error=" << error << endl;
	// cout << "sigma=" << sigma << endl;
	
	// double pars[5]; 
	// fit->GetParameters(&pars[0]);  // Almacena los datos del fit, comenzando del indice 0 en pars
	
	// cout << "pars[0] norm value= " << pars[0] << endl;
	// cout << "pars[1] Mean value= " << pars[1] << endl;
	// cout << "pars[2] Sigma Value= " << pars[2] << endl;
	// cout << "pars[3] gain Value= " << pars[3] << endl;
	// cout << "pars[4] const Value= " << pars[4] << endl;
}
