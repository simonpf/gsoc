#include "TFile.h"
#include "TString.h"

#include "TMVA/Factory.h"
#include "TMVA/Tools.h"
#include "TMVA/DataLoader.h"
#include "TMVA/TMVAGui.h"

void train_higgs( )
{
    TMVA::Tools::Instance();

    // File input.

    TString infilename = "higgs.root";
    TFile *input = TFile::Open(infilename);

    TString outfilename = "TMVA_p3.root";
    TFile *output = TFile::Open(outfilename, "RECREATE");

    TMVA::DataLoader *loader=new TMVA::DataLoader("higgs");
    TMVA::Factory *factory = new TMVA::Factory( "TMVAClassification",
                                                output,
                                                "AnalysisType=Classification");

    // Register data with loader.

    TTree *signal     = (TTree *) input->Get("TreeS");
    TTree *background = (TTree *) input->Get("TreeB");

    loader->AddVariable("lepton_pT",'F');
    loader->AddVariable("lepton_eta",'F');
    loader->AddVariable("lepton_phi",'F');
    loader->AddVariable("missing_energy_magnitude",'F');
    loader->AddVariable("missing_energy_phi",'F');
    loader->AddVariable("jet_1_pt",'F');
    loader->AddVariable("jet_1_eta",'F');
    loader->AddVariable("jet_1_phi",'F');
    loader->AddVariable("jet_1_b_tag",'F');
    loader->AddVariable("jet_2_pt",'F');
    loader->AddVariable("jet_2_eta",'F');
    loader->AddVariable("jet_2_phi",'F');
    loader->AddVariable("jet_2_b_tag",'F');
    loader->AddVariable("jet_3_pt",'F');
    loader->AddVariable("jet_3_eta",'F');
    loader->AddVariable("jet_3_phi",'F');
    loader->AddVariable("jet_3_b_tag",'F');
    loader->AddVariable("jet_4_pt",'F');
    loader->AddVariable("jet_4_eta",'F');
    loader->AddVariable("jet_4_phi",'F');
    loader->AddVariable("jet_4_b_tag",'F');
    loader->AddVariable("m_jj",'F');
    loader->AddVariable("m_jjj",'F');
    loader->AddVariable("m_lv",'F');
    loader->AddVariable("m_jlv",'F');
    loader->AddVariable("m_bb",'F');
    loader->AddVariable("m_wbb",'F');
    loader->AddVariable("m_wwbb",'F');

    Double_t signalWeight = 1.0;
    Double_t backgroundWeight = 1.0;
    loader->AddSignalTree    (signal,     signalWeight);
    loader->AddBackgroundTree(background, backgroundWeight);

    TString dataString = "nTrain_Signal=1000000:"
                         "nTrain_Background=1000000:"
                         "nTest_Signal=1000:"
                         "nTest_Background=1000:"
                         "SplitMode=Random:"
                         "NormMode=NumEvents:"
                         "!V";

    loader->PrepareTrainingAndTestTree("", "", dataString);

    // Network configuration.

    TString configString = "!H:V";

    configString += ":VarTransform=G";

    configString += ":ErrorStrategy=CROSSENTROPY";

    configString += ":WeightInitialization=XAVIERUNIFORM";

    // Network layout.
    TString layoutString = "Layout=RELU|100,RELU|50,RELU|10,LINEAR";

    // Training strategy.
    TString trainingString1 = "TrainingStrategy="
                              "LearningRate=1e-1,"
                              "Momentum=0.5,"
                              "Repetitions=1,"
                              "ConvergenceSteps=300,"
                              "BatchSize=20,"
                              "DropConfig=0.0+0.5+0.5+0.0,"  // Dropout
                              "DropRepetitions=5,"
                              "WeightDecay=0.001,"
                              "Regularization=L2,"
                              "TestRepetitions=15,"
                              "Multithreading=True";

    TString trainingString2 = "|LearningRate=1e-2,"
                              "Momentum=0.1,"
                              "Repetitions=1,"
                              "ConvergenceSteps=300,"
                              "BatchSize=20,"
                              "DropConfig=0.0+0.1+0.1+0.0,"  // Dropout
                              "DropRepetitions=5,"
                              "WeightDecay=0.001,"
                              "Regularization=L2,"
                              "TestRepetitions=15,"
                              "Multithreading=True";

    TString trainingString3 = "|LearningRate=1e-3"
                              "Momentum=0.0,"
                              "Repetitions=1,"
                              "ConvergenceSteps=300,"
                              "BatchSize=50,"
                              "WeightDecay=0.001,"
                              "Regularization=L2,"
                              "TestRepetitions=15,"
                              "Multithreading=True";

    configString += ":" + layoutString + ":" + trainingString1 + trainingString2 + trainingString3;
    factory->BookMethod(loader, TMVA::Types::kDNN, "DNN 1", configString);

    // configString += trainingString2;
    // factory->BookMethod(loader, TMVA::Types::kDNN, "DNN 2", configString);

    // configString += trainingString3;
    // factory->BookMethod(loader, TMVA::Types::kDNN, "DNN 3", configString);

    factory->TrainAllMethods();
    factory->TestAllMethods();
    factory->EvaluateAllMethods();

    output->Close();

    delete factory;
    delete loader;

}
