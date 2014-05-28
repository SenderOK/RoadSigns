#include <map>
#include <string>
#include <fstream>
#include <iostream>

#include "argvparser/argvparser.h"

using std::map;
using std::string;
using std::ifstream;
using std::cout;
using std::endl;

using CommandLineProcessing::ArgvParser;

typedef map<string, string> TLabels;

void LoadLabels(const string& data_file, TLabels* labels) {
    ifstream stream(data_file.c_str());

    string filename;
    int label;

    int char_idx = data_file.size() - 1;
    for (; char_idx >= 0; --char_idx)
        if (data_file[char_idx] == '/' || data_file[char_idx] == '\\')
            break;
    string data_path = data_file.substr(0,char_idx+1);

    while(!stream.eof() && !stream.fail()) {
        stream >> filename >> label;
        if (filename.size())
            (*labels)[data_path + filename] = label;
    }

    stream.close();
}

void TestLabels(const TLabels& gt_labels, const TLabels& predicted_labels) {
    int correct_predictions = 0;

    if (gt_labels.size() != predicted_labels.size()) {
        cout << "Error! Files with predicted and ground truth labels "
                "have different number of samples." << endl;
        return;
    }

    if (!gt_labels.size()) {
        cout << "Error! Dataset is empty.";
        return;
    }

    for (TLabels::const_iterator predicted_it = predicted_labels.begin();
         predicted_it != predicted_labels.end();
         ++predicted_it) {
        string sample = predicted_it->first;
        TLabels::const_iterator gt_it = gt_labels.find(sample);
        if (gt_it == gt_labels.end()) {
            cout << "Error! File " << sample << " has no ground truth label."
                 << endl;
            return;
        }

        if (predicted_it->second == gt_it->second)
            ++correct_predictions;
    }
    cout << "Precision: " << double(correct_predictions) / gt_labels.size() << endl;
}

int main(int argc, char** argv) {
    ArgvParser cmd;
    cmd.setIntroductoryDescription("Machine graphics course, program for testing task 2. CMC MSU, 2013.");
    cmd.setHelpOption("h", "help", "Print this help message");
    cmd.defineOption("gt_labels", "File with ground truth labels",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("predicted_labels", "File with predicted labels",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);

    cmd.defineOptionAlternative("gt_labels", "g");
    cmd.defineOptionAlternative("predicted_labels", "p");

    int result = cmd.parse(argc, argv);
    if (result) {
        cout << cmd.parseErrorDescription(result) << endl;
        return result;
    }

    TLabels gt_labels, predicted_labels;
    LoadLabels(cmd.optionValue("gt_labels"), &gt_labels);
    LoadLabels(cmd.optionValue("predicted_labels"), &predicted_labels);

    TestLabels(gt_labels, predicted_labels);
}