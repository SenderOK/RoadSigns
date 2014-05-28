#pragma once

#include <tuple>
using std::tuple;
using std::get;
using std::tie;
using std::make_tuple;


#include "matrix.h"
#include "io.h"

typedef unsigned int uint;

class FilterDimensions
{
public:
    const uint r_horz;
    const uint r_vert;
    
    FilterDimensions(uint _r_horz, uint _r_vert);
    virtual ~FilterDimensions();
};

class CustomFilter : public FilterDimensions
{
    const Matrix<double> kernel;
public:
    CustomFilter(const Matrix<double> &_kernel);
    double operator()(const Matrix<double> &m) const;
};

class WeightPixelFilter : public FilterDimensions
{
    const double r_weight;
    const double g_weight;
    const double b_weight;
public:
    WeightPixelFilter(double _r_weight, double _g_weight, double _b_weight);
    double operator()(const PreciseImage &m) const;
};

class ClipPixelFilter : public FilterDimensions
{
public:
    ClipPixelFilter();
    double operator()(const Matrix<double> &m) const;
};

class MakePreciseFilter : public FilterDimensions
{
public:
    MakePreciseFilter();
    PrecisePixel operator()(const Image &m) const;
};

class CalcGradValFilter
{
public:
    double operator()(double x, double y) const;
};

class CalcGradDirFilter
{
public:
    double operator()(double x, double y) const;
};

class UniteThreeFilter
{
public:
    PrecisePixel operator()(double x, double y, double z) const;
};

class MedianFilter : public FilterDimensions
{
public:
    MedianFilter(uint _r_horz = 1, uint _r_vert = 1);
    Pixel operator()(const Image &m) const;
};


Matrix<double> GenGaussKernel(uint r_vert, uint r_horz, double sigma);

Image LinearMedianFilter(const Image &m, uint r);
Image ConstantMedianFilter(const Image &m, uint radius);
