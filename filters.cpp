#include <cmath>
#include <string>

#include "filters.h"

using std::vector;

//______________________________________________________________________________

FilterDimensions::FilterDimensions(uint _r_horz, uint _r_vert) : 
    r_horz(_r_horz), 
    r_vert(_r_vert) 
{ }


FilterDimensions::~FilterDimensions() 
{ }

//______________________________________________________________________________

CustomFilter::CustomFilter(const Matrix<double> &_kernel) : 
    FilterDimensions(_kernel.n_cols / 2, _kernel.n_rows / 2),
    kernel(_kernel)
{ 
    if(_kernel.n_cols % 2 == 0 || _kernel.n_rows % 2 == 0){
        throw std::string("Error in CustomFilter::CustomFilter: the kernel must have odd dimensions");
    }
}

double CustomFilter::operator()(const Matrix<double> &m) const
{
    double ans = 0;
    for (uint i = 0; i < 2 * r_vert + 1; ++i)
        for (uint j = 0; j < 2 * r_horz + 1; ++j)
            ans += m(i, j) * kernel(i, j);
    return ans;
}

//______________________________________________________________________________

WeightPixelFilter::WeightPixelFilter(double _r_weight, double _g_weight, double _b_weight) : 
    FilterDimensions(0, 0),
    r_weight(_r_weight),
    g_weight(_g_weight),
    b_weight(_b_weight)
{ }

double WeightPixelFilter::operator()(const PreciseImage &m) const
{
    double r, g, b;
    tie(r, g, b) = m(0, 0);
    return r * r_weight + g * g_weight + b * b_weight;
}


//______________________________________________________________________________

double CalcGradValFilter::operator()(double x, double y) const
{
    return sqrt(x * x + y * y);
}

double CalcGradDirFilter::operator()(double x, double y) const
{
    return atan2(y, x);
}

PrecisePixel UniteThreeFilter::operator()(double x, double y, double z) const
{
    return make_tuple(x, y, z);
}

//______________________________________________________________________________

ClipPixelFilter::ClipPixelFilter() :
    FilterDimensions(0, 0)
{ }

double ClipPixelFilter::operator()(const Matrix<double> &m) const
{
    return std::min(std::max(m(0, 0), 0.0), 255.0);
}

MakePreciseFilter::MakePreciseFilter() :
    FilterDimensions(0, 0)
{ }

PrecisePixel MakePreciseFilter::operator()(const Image &m) const
{
    uint r, g, b;
    tie(r, g, b) = m(0, 0);
    return make_tuple(double(r), double(g), double(b));
}

//______________________________________________________________________________

class Histogram {
    static const uint COLOR_DEPTH = 256;
    
    vector<uint> r_hist;
    vector<uint> g_hist;
    vector<uint> b_hist;
    uint n_values;
    
    uint find_median(const vector<uint> &hist) const;
public:
    Histogram();
    
    void add_pixel(const Pixel &pixel);
    void subtract_pixel(const Pixel &pixel);
    void add_histogram(const Histogram &h);
    void subtract_histogram(const Histogram &h);
    Pixel get_median() const;  
};

uint Histogram::find_median(const vector<uint> &hist) const
{
    uint median_pos = (n_values / 2) + 1;
    uint curr_n = 0;
    for (uint i = 0; i < COLOR_DEPTH; ++i) {
        curr_n = curr_n + hist[i];
        if (curr_n >= median_pos) {
            return i;
        }
    }
    return 0;
}

Histogram::Histogram() : 
    r_hist(COLOR_DEPTH), 
    g_hist(COLOR_DEPTH),  
    b_hist(COLOR_DEPTH),
    n_values(0) 
{ }

void Histogram::add_pixel(const Pixel &pixel)
{
    uint r, g, b;
    tie(r, g, b) = pixel;
    ++r_hist[r];
    ++g_hist[g];
    ++b_hist[b];
    ++n_values;
}

void Histogram::subtract_pixel(const Pixel &pixel)
{
    uint r, g, b;
    tie(r, g, b) = pixel;
    
    if (n_values == 0 || r_hist[r] == 0 || g_hist[g] == 0 || b_hist[b] == 0) 
        throw std::string("Error in subtract_pixel");
        
    --r_hist[r];
    --g_hist[g];
    --b_hist[b];
    --n_values;
}

void Histogram::add_histogram(const Histogram &h)
{
    for (uint i = 0; i < COLOR_DEPTH; ++i) {
        r_hist[i] += h.r_hist[i];
        g_hist[i] += h.g_hist[i];
        b_hist[i] += h.b_hist[i];
    }
    n_values += h.n_values;
}

void Histogram::subtract_histogram(const Histogram &h)
{
    for (uint i = 0; i < COLOR_DEPTH; ++i) {
        if (r_hist[i] < h.r_hist[i] || g_hist[i] < h.g_hist[i] || b_hist[i] < h.b_hist[i])
            throw std::string("Error in subtract_histogram");
        r_hist[i] -= h.r_hist[i];
        g_hist[i] -= h.g_hist[i];
        b_hist[i] -= h.b_hist[i];
    }
    
    if (n_values < h.n_values) 
        throw std::string("Error in subtract_histogram");
        
    n_values -= h.n_values;
}

Pixel Histogram::get_median() const
{
    return make_tuple(find_median(r_hist), find_median(g_hist), find_median(b_hist));
}

//______________________________________________________________________________

MedianFilter::MedianFilter(uint _r_horz, uint _r_vert) :
    FilterDimensions(_r_horz, _r_vert)
{ }

Pixel MedianFilter::operator()(const Image &m) const
{
    Histogram hist;
    
    for (uint i = 0; i < 2 * r_vert + 1; ++i)
        for (uint j = 0; j < 2 * r_horz + 1; ++j)
            hist.add_pixel(m(i, j));
    
    return hist.get_median();
}

Image LinearMedianFilter(const Image &m, uint radius)
{
    if (m.n_rows < radius || m.n_cols < radius)
        return Image(0, 0);

    Image im = m.MirrorExpand(radius, radius);
    Image result(m.n_rows, m.n_cols);
    
    for (uint i = radius; i < im.n_rows - radius; ++i) {
        Histogram hist;
    
        for (uint j = 0; j < 2 * radius; ++j)
            for (uint k = 0; k < 2 * radius + 1; ++k)
                hist.add_pixel(im(i + k - radius, j));
                
        for (uint j = 2 * radius; j < im.n_cols; ++j){
            for (uint k = 0; k < 2 * radius + 1; ++k)
                hist.add_pixel(im(i + k - radius, j));
            
            result(i - radius, j - 2 * radius) = hist.get_median();
            
            for (uint k = 0; k < 2 * radius + 1; ++k)
                hist.subtract_pixel(im(i + k - radius, j - 2 * radius));
        }
    }
    
    return result;
}

Image ConstantMedianFilter(const Image &m, uint radius)
{
    if (m.n_rows < radius || m.n_cols < radius)
        return Image(0, 0);
        
    Image im = m.MirrorExpand(radius, radius);
    Image result(m.n_rows, m.n_cols);
    
    vector<Histogram> hists(im.n_cols);
    
    for (uint i = 0; i < 2 * radius; ++i){
        for (uint j = 0; j < im.n_cols; ++j)
            hists[j].add_pixel(im(i, j));
    }
    
    for (uint i = 2 * radius; i < im.n_rows; ++i) {

        for (uint j = 0; j < im.n_cols; ++j)
            hists[j].add_pixel(im(i, j));

        Histogram total_hist;
    
        for (uint j = 0; j < 2 * radius; ++j)
            total_hist.add_histogram(hists[j]);
            
        for (uint j = 2 * radius; j < im.n_cols; ++j) {
            total_hist.add_histogram(hists[j]);
            result(i - 2 * radius, j - 2 * radius) = total_hist.get_median();
            total_hist.subtract_histogram(hists[j - 2 * radius]);
        }
        
        for (uint j = 0; j < im.n_cols; ++j)
            hists[j].subtract_pixel(im(i - 2 * radius, j));

    }
 
    return result;    
}

//______________________________________________________________________________

Matrix<double> GenGaussKernel(uint r_vert, uint r_horz, double sigma)
{
    Matrix<double> result(2 * r_vert + 1, 2 * r_horz + 1);
    double sum = 0;
    for (uint i = 0; i < 2 * r_vert + 1; ++i) {
        for (uint j = 0; j < 2 * r_horz + 1; ++j ) {
            double power = -1.0 / (2.0 * sigma * sigma) * 
                            ((double(i) - r_vert) * (double(i) - r_vert) + 
                             (double(j) - r_horz) * (double(j) - r_horz));
            double a = exp(power);
            result(i, j) = a;
            sum += a;
        }
    }
    
    for (uint i = 0; i < 2 * r_vert + 1; ++i) {
        for (uint j = 0; j < 2 * r_horz + 1; ++j ) {
            result(i, j) = result(i, j) / sum;
        }
    }
    
    return result;
}

