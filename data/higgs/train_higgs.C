#include "TFile.h"
#include "TString.h"

#include "TMVA/Factory.h"
#include "TMVA/Tools.h"
#include "TMVA/DataLoader.h"
#include "TMVA/TMVAGui.h"

void train_higgs( )
{
    TMVA::Tools::Instance();

    TString fname="higgs.root";
    TFile *input = TFile::Open(fname);
    TMVA::DataLoader *loader=new TMVA::DataLoader("higgs");
}
