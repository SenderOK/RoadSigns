#ifndef CLASSIFIER_H_
#define CLASSIFIER_H_

#include <vector>
#include <string>
#include <cstdlib>
#include <iostream>
#include <memory>

#include "liblinear-1.93/linear.h"

using std::vector;
using std::pair;
using std::string;
using std::auto_ptr;

typedef vector<pair<vector<float>, int> > TFeatures;
typedef vector<int> TLabels;

// Model of classifier to be trained
// Encapsulates 'struct model' from liblinear
class TModel {
        // Pointer to liblinear model;
    auto_ptr<struct model> model_;
 public:
        // Basic constructor
    TModel(): model_(NULL) {}
        // Construct class by liblinear model
    TModel(struct model* model): model_(model) {}
        // Operator = for liblinear model
    TModel& operator=(struct model* model) {
        model_ = auto_ptr<struct model>(model);
        return *this;
    }
        // Save model to file
    void Save(const string& model_file) const {
        assert(model_.get());
        save_model(model_file.c_str(), model_.get());
    }
        // Load model from file
    void Load(const string& model_file) {
        model_ = auto_ptr<struct model>(load_model(model_file.c_str()));
    }
        // Get pointer to liblinear model
    struct model* get() const {
        return model_.get();
    }
};

// Parameters for classifier training
// Read more about it in liblinear documentation
struct TClassifierParams {
    double bias;
    int solver_type;
    double C;
    double eps;
    int nr_weight;
    int* weight_label;
    double* weight;

    TClassifierParams() {
        bias = -1;
        solver_type = L2R_L2LOSS_SVC_DUAL;
        C = 0.1;
        eps = 1e-4;
        nr_weight = 0;
        weight_label = NULL;
        weight = NULL;
    }
};

// Classifier. Encapsulates liblinear classifier.
class TClassifier {
        // Parameters of classifier
    TClassifierParams params_;

 public:
        // Basic constructor
    TClassifier(const TClassifierParams& params): params_(params) {}

        // Train classifier
    void Train(TFeatures& features, TModel* model) {
            // Number of samples and features must be nonzero
        size_t number_of_samples = features.size();
        assert(number_of_samples > 0);

        size_t number_of_features = features[0].first.size();
        assert(number_of_features > 0);

            // Description of one problem
        struct problem prob;
        prob.l = number_of_samples;
        prob.bias = -1;
        prob.n = number_of_features;
        prob.y = new double[number_of_samples];
        prob.x = new struct feature_node*[number_of_samples];
        
            // Fill struct problem
        for (size_t sample_idx = 0; sample_idx < number_of_samples; ++sample_idx)
        {
            prob.x[sample_idx] = new struct feature_node[number_of_features + 1];
            for (int feature_idx = 0; feature_idx < number_of_features; feature_idx++)
            {
                prob.x[sample_idx][feature_idx].index = feature_idx + 1;
                prob.x[sample_idx][feature_idx].value = features[sample_idx].first[feature_idx];
            }
            prob.x[sample_idx][number_of_features].index = -1;
            prob.y[sample_idx] = features[sample_idx].second;
            
            vector<float>().swap(features[sample_idx].first);
        }

            // Fill param structure by values from 'params_'
        struct parameter param;
        param.solver_type = params_.solver_type;
        param.C = params_.C;      // try to vary it
        param.eps = params_.eps;
        param.nr_weight = params_.nr_weight;
        param.weight_label = params_.weight_label;
        param.weight = params_.weight;

            // Train model
        *model = train(&prob, &param);

            // Clear param structure
        destroy_param(&param);
            // clear problem structure
        delete[] prob.y;
        for (int sample_idx = 0; sample_idx < number_of_samples; ++sample_idx)
            delete[] prob.x[sample_idx];
        delete[] prob.x;
    }

        // Predict data
    void Predict(const TFeatures& features, const TModel& model, TLabels* labels) {
            // Number of samples and features must be nonzero
        size_t number_of_samples = features.size();
        assert(number_of_samples > 0);
        size_t number_of_features = features[0].first.size();
        assert(number_of_features > 0);

            // Fill struct problem
        struct feature_node* x = new struct feature_node[number_of_features + 1];
        for (size_t sample_idx = 0; sample_idx < features.size(); ++sample_idx) {
            for (int feature_idx = 0; feature_idx < number_of_features; ++feature_idx) {
                x[feature_idx].index = feature_idx + 1;
                x[feature_idx].value = features[sample_idx].first[feature_idx];
            }
            x[number_of_features].index = -1;
                // Add predicted label to labels structure
            labels->push_back(predict(model.get(), x));
        }
    }
};

#endif
