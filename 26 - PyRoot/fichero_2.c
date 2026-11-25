
#include <iostream>

//void fichero_2(const char *archivo)
void fichero_2(const char *archivo)
{
    printf("%s\n",archivo);


    //string inFileName = "/home/oem/datosFits/histogram_nsamp_100.csv";
    string inFileName = archivo;
    ifstream inFile;

    //--------------------------------------------------------------
    // open and read once to get number of lines from 0 to nlines -1
    inFile.open(inFileName.c_str());
    //count lines in file
    int nlines = 0;
    string aline;

    while ( getline(inFile, aline) ) ++nlines;
    cout << "Number of lines in text file: " << nlines;
    cout << endl;
    inFile.close(); // CLose input file
    //--------------------------------------------------------------

    // define histogram variable binning
    Int_t     NBins = nlines; 
    const int NBinsPlus1 = NBins + 1;
	double    XBins[NBinsPlus1]; 
	double    BinCont[NBins];

    double x,y;
    char sep;

    //--------------------------------------------------------------
    //read file again to get data
    inFile.open(inFileName.c_str());

    if (inFile.is_open())
    {
        for (int i = 0; i < nlines+1; i++)
        {
            inFile >> x >> sep >> y;
            //cout << i+1 << " " << x << sep << y << endl;


            XBins[i]=x;
            if (i<nlines-1) BinCont[i] = y;
            //else BinCont[i]=-1111;

        }

        inFile.close(); // CLose input file
    }
    else { //Error message
        cerr << "Can't find input file " << inFileName << endl;
    }

    cout << endl;


    //--------------------------------------------------------------
    // create and fill variable bin histogram

    TH1F  *histo = new TH1F("histo","",NBins,XBins);

    //Fill histogram
    for (int i = 0; i < NBins+1; i++) {
       //cout << i+1 << ": " << XBins[i] << endl; 
        histo->SetBinContent(i+1,BinCont[i]);

    }

    //--------------------------------------------------------------

    //TTree *T = new TTree("ntuple","data from ascii file");
    // Long64_t nlines = T->ReadFile(Form("%s",archivo),"x:y");
    // printf(" found %lld points\n",nlines);    



    //Drawing section

    TCanvas *c1 = new TCanvas("c1","The FillRandom example",700,400);

    //gPad->SetLogY();
    histo->Draw("");
    c1->Print("plot.pdf");
 



}