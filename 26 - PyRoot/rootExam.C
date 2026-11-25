#include "TH1F.h"
#include "TCanvas.h"
#include "TAxis.h"
#include <iostream>

void rootExam() {
    // Create a sample histogram
    TH1F *h = new TH1F("h", "Example Histogram", 100, 0, 10);

    // Fill with some data
    for (int i = 0; i < 10000; ++i) {
        h->Fill(gRandom->Gaus(5, 1));
    }

    // Define the range of interest
    double x_min_range = 3.0;
    double x_max_range = 6.0;

    // Set the X-axis range for finding the maximum
    int bin_min = h->FindBin(x_min_range);
    int bin_max = h->FindBin(x_max_range);
    h->GetXaxis()->SetRange(bin_min, bin_max);

    // Get the maximum value within this range
    double max_value_in_range = h->GetMaximum();
    std::cout << "Maximum value in the range [" << x_min_range << ", " << x_max_range << "]: " << max_value_in_range << std::endl;

    // Optionally, reset the X-axis range
    h->GetXaxis()->SetRange(0, 0);

    // Get the global maximum (over the entire histogram)
    double global_max_value = h->GetMaximum();
    std::cout << "Global maximum value: " << global_max_value << std::endl;

    // Draw the histogram
    TCanvas *c1 = new TCanvas("c1", "Histogram");
    h->Draw();
}
