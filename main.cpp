#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>

#include "classifier.h"
#include "EasyBMP/EasyBMP.h"
#include "liblinear-1.93/linear.h"
#include "argvparser/argvparser.h"
#include "io.h"
#include "filters.h"

using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;

using CommandLineProcessing::ArgvParser;

//_______________________________________________________________________________________
//PARAMETERS ARE DESCRIBED HERE
//_______________________________________________________________________________________
const int N_DIRECTIONS = 13;

const double OVERLAP = 0.7;
const int BLOCK_SIZE = 4;
const int N_BLOCKS = 3;
const int METABLOCK_STRIDE = BLOCK_SIZE * N_BLOCKS * (1 - OVERLAP);
const Matrix<double> filter_x = Matrix<double>({ {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} });
const Matrix<double> filter_y = Matrix<double>({ {1, 2, 1}, {0, 0, 0}, {-1, -2, -1} });
const int SVM_SOLVER = L2R_L2LOSS_SVC_DUAL;
const double SVM_C = 0.2;

enum NORM {
    L1,
    L2,
    LINF,
    L2Hys
};

const int CURR_NORM = L2Hys;

const bool KERNEL_IS_ON = false;
const float KERNEL_L = 0.25;
const int KERNEL_N = 1;

//________________________________________________________________________________________
//________________________________________________________________________________________

const double PI = 3.14159265358979323;

typedef vector<pair<string, int> > TFileList;
typedef vector<pair<vector<float>, int> > TFeatures;

// Load list of files and its labels from 'data_file' and
// stores it in 'file_list'
void LoadFileList(const string& data_file, TFileList* file_list) {
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
            file_list->push_back(make_pair(data_path + filename, label));
    }

    stream.close();
}

// Save result of prediction to file
void SavePredictions(const TFileList& file_list,
                     const TLabels& labels, 
                     const string& prediction_file) {
        // Check that list of files and list of labels has equal size 
    assert(file_list.size() == labels.size());
        // Open 'prediction_file' for writing
    ofstream stream(prediction_file.c_str());

        // Write file names and labels to stream
    for (size_t image_idx = 0; image_idx < file_list.size(); ++image_idx)
        stream << file_list[image_idx].first << " " << labels[image_idx] << endl;
    stream.close();
}

vector<float> get_histogram(const Matrix<double> &image)
{    
    Matrix<double> sobel_x = image.unary_map(CustomFilter(filter_x));
    Matrix<double> sobel_y = image.unary_map(CustomFilter(filter_y));
   
    Matrix<double> grad_vals = binary_map(CalcGradValFilter(), sobel_y, sobel_x);
    Matrix<double> grad_dirs = binary_map(CalcGradDirFilter(), sobel_y, sobel_x);
    
    vector<float> histogram(N_DIRECTIONS, 0);
    
    for (uint i = 0; i < image.n_rows; ++i) {
        for (uint j = 0; j < image.n_cols; ++j) {
            uint sector_num = uint(((grad_dirs(i, j) / PI + 1) / 2) * N_DIRECTIONS);          
            histogram[(sector_num == N_DIRECTIONS) ? N_DIRECTIONS - 1 : sector_num] += grad_vals(i, j);
        }
    }
    
    return histogram;
}

const float EPS = 0.0001;

void NormL1(vector<float> &v)
{
    float sum = 0;
    for (auto it = v.begin(); it < v.end(); ++it)
        sum += *it;
      
    sum += EPS;
        
    for (auto it = v.begin(); it < v.end(); ++it)
        *it /= sum;
}

void NormL2(vector<float> &v)
{
    float sum_sqr = 0;
    for (auto it = v.begin(); it < v.end(); ++it)
        sum_sqr += (*it) * (*it);
      
    sum_sqr = sqrt(sum_sqr + EPS * EPS);
       
    for (auto it = v.begin(); it < v.end(); ++it)
        *it /= sum_sqr;
}

void NormL2Hys(vector<float> &v)
{
    float sum_sqr = 0;
    for (auto it = v.begin(); it < v.end(); ++it)
        sum_sqr += (*it) * (*it);
      
    sum_sqr = sqrt(sum_sqr + EPS * EPS);
       
    for (auto it = v.begin(); it < v.end(); ++it) {
        *it /= sum_sqr;
        if (*it > 0.2) {
            *it = 0.2;
        }
    }
    NormL2(v);
}

void NormLINF(vector<float> &v)
{
    float max_element = 0;
    for (auto it = v.begin(); it < v.end(); ++it)
        max_element = std::max(max_element, *it);
      
    if (max_element < EPS)
        return;
        
    for (auto it = v.begin(); it < v.end(); ++it)
        *it /= max_element;
}

vector<float> chi_sqare_kernel(float x)
{
    vector<float> result;
    result.reserve(2 * (2 * KERNEL_N + 1));
    if (x < EPS) {
        for (int i = -KERNEL_N; i <= KERNEL_N; ++i) {
            result.push_back(0);
            result.push_back(0);
        }
    } else {
        for (int i = -KERNEL_N; i <= KERNEL_N; ++i) {
            float lambda = KERNEL_L * i;
            float coeff = sqrt((x / cosh(PI * lambda)));
            float tmp = lambda * log(x);
            result.push_back(cos(tmp) * coeff);
            result.push_back(-sin(tmp) * coeff);
        }
    }
    return result;
}

// Extract features from dataset.
void ExtractFeatures(const TFileList& file_list, TFeatures& features) {

    typedef void (*norm_ptr) (vector<float> &);
    norm_ptr functions[] = 
    {
        NormL1,
        NormL2,
        NormLINF,
        NormL2Hys,
    };
    const norm_ptr norm_f = functions[CURR_NORM];
    
    features.reserve(file_list.size());
    for (size_t image_idx = 0; image_idx < file_list.size(); ++image_idx) {
    
        PreciseImage im = load_image<double>(file_list[image_idx].first.c_str());
        
        Matrix<double> image = im.unary_map(WeightPixelFilter(0.299, 0.587, 0.114));
                
        vector<float> one_image_features;        
        for (uint i = 0; i + N_BLOCKS * BLOCK_SIZE < image.n_rows; i += METABLOCK_STRIDE) {
            for (uint j = 0; j + N_BLOCKS * BLOCK_SIZE < image.n_cols; j += METABLOCK_STRIDE) {
            
                vector<float> one_metablock_features;
		one_metablock_features.reserve(N_BLOCKS * N_BLOCKS * N_DIRECTIONS * ((KERNEL_IS_ON) ? 2 * (2 * KERNEL_N + 1) : 1));

                for (uint row = 0; row < N_BLOCKS; ++row) {
                    for (uint col = 0; col < N_BLOCKS; ++col) {

			Matrix<double> part = image.submatrix(i + row * BLOCK_SIZE, j + col * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
			vector<float> one_block_hist;

	                if (KERNEL_IS_ON) {
        	            vector<float> hist = get_histogram(part);
                	    one_block_hist.reserve(2 * (2 * KERNEL_N + 1) * hist.size());
	                    for (int k = 0; k < hist.size(); ++k) {
        	                vector<float> tmp = chi_sqare_kernel(hist[k]);
                	        one_block_hist.insert(one_block_hist.end(), tmp.begin(), tmp.end());
	                    }
	                } else {
        	            one_block_hist = get_histogram(part);
                	}
			one_metablock_features.insert(one_metablock_features.end(), one_block_hist.begin(), one_block_hist.end());
		    }
                }
                
                norm_f(one_metablock_features);
                
                one_image_features.insert(one_image_features.end(), one_metablock_features.begin(), one_metablock_features.end());
            }
        }
        
        features.push_back(make_pair(one_image_features, file_list[image_idx].second));
    }
}

// Train SVM classifier using data from 'data_file' and save trained model
// to 'model_file'
void TrainClassifier(const string& data_file, const string& model_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TFeatures features;
        // Model which would be trained
    TModel model;
        // Parameters of classifier
    TClassifierParams params;
        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Extract features from images
    ExtractFeatures(file_list, features);
    
    params.C = SVM_C;
    params.solver_type = SVM_SOLVER;
    

    TClassifier classifier(params);
        // Train classifier
    classifier.Train(features, &model);
        // Save model to file
    model.Save(model_file);
}

// Predict data from 'data_file' using model from 'model_file' and
// save predictions to 'prediction_file'
void PredictData(const string& data_file,
                 const string& model_file,
                 const string& prediction_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TFeatures features;
        // List of image labels
    TLabels labels;

        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    ExtractFeatures(file_list, features);

        // Classifier 
    TClassifier classifier = TClassifier(TClassifierParams());
        // Trained model
    TModel model;
        // Load model from file
    model.Load(model_file);
        // Predict images by its features using 'model' and store predictions
        // to 'labels'
    classifier.Predict(features, model, &labels);

        // Save predictions
    SavePredictions(file_list, labels, prediction_file);
}

int main(int argc, char** argv) {
        // Command line options parser
    ArgvParser cmd;
        // Description of program
    cmd.setIntroductoryDescription("Machine graphics course, task 2. CMC MSU, 2013.");
        // Add help option
    cmd.setHelpOption("h", "help", "Print this help message");
        // Add other options
    cmd.defineOption("data_set", "File with dataset",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("model", "Path to file to save or load model",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("predicted_labels", "Path to file to save prediction results",
        ArgvParser::OptionRequiresValue);
    cmd.defineOption("train", "Train classifier");
    cmd.defineOption("predict", "Predict dataset");
        
        // Add options aliases
    cmd.defineOptionAlternative("data_set", "d");
    cmd.defineOptionAlternative("model", "m");
    cmd.defineOptionAlternative("predicted_labels", "l");
    cmd.defineOptionAlternative("train", "t");
    cmd.defineOptionAlternative("predict", "p");

        // Parse options
    int result = cmd.parse(argc, argv);

        // Check for errors or help option
    if (result) {
        cout << cmd.parseErrorDescription(result) << endl;
        return result;
    }
    
       // Get values 
    string data_file = cmd.optionValue("data_set");
    string model_file = cmd.optionValue("model");
    bool train = cmd.foundOption("train");
    bool predict = cmd.foundOption("predict");

        // If we need to train classifier
    if (train) {
        TrainClassifier(data_file, model_file);
    }
        // If we need to predict data
    if (predict) {
            // You must declare file to save images
        if (!cmd.foundOption("predicted_labels")) {
            cerr << "Error! Option --predicted_labels not found!" << endl;
            return 1;
        }
            // File to save predictions
        string prediction_file = cmd.optionValue("predicted_labels");
            // Predict data
        PredictData(data_file, model_file, prediction_file);
    }
}
