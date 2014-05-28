#include <string>
using std::string;

using std::tuple;
using std::make_tuple;
using std::tie;

template <typename T>
Matrix<std::tuple<T, T, T>> load_image(const char *path)
{
    BMP in;

    if (!in.ReadFromFile(path))
        throw string("Error reading file ") + string(path);

    Matrix<std::tuple<T, T, T>> res(in.TellHeight(), in.TellWidth());

    for (uint i = 0; i < res.n_rows; ++i) {
        for (uint j = 0; j < res.n_cols; ++j) {
            RGBApixel *p = in(j, i);
            res(i, j) = make_tuple(T(p->Red), T(p->Green), T(p->Blue));
        }
    }

    return res;
}

template <typename T>
uint clip(T v)
{
    return uint(std::min(std::max(v, T(0)), T(255)));
}

template <typename T>
void save_image(const Matrix<std::tuple<T, T, T>> &im, const char *path)
{
    BMP out;
    out.SetSize(im.n_cols, im.n_rows);

    T r, g, b;
    RGBApixel p;
    p.Alpha = 255;
    for (uint i = 0; i < im.n_rows; ++i) {
        for (uint j = 0; j < im.n_cols; ++j) {
            tie(r, g, b) = im(i, j);
            p.Red = clip(r); p.Green = clip(g); p.Blue = clip(b);
            out.SetPixel(j, i, p);
        }
    }

    if (!out.WriteToFile(path))
        throw string("Error writing file ") + string(path);
}
