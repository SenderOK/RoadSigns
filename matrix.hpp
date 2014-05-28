template<typename ValueT>
template<typename T>
inline
T& Matrix<ValueT>::make_rw(const T& val) const
{
    return const_cast<T&>(val);
}

template<typename ValueT>
Matrix<ValueT>::Matrix(uint row_count, uint col_count):
    n_rows{row_count},
    n_cols{col_count},
    stride{n_cols},
    pin_row{0},
    pin_col{0},
    _data{}
{
    auto size = n_cols * n_rows;
    if (size)
        _data.reset(new ValueT[size], std::default_delete<ValueT[]>());
}

template<typename ValueT>
Matrix<ValueT>::Matrix(std::initializer_list<ValueT> lst):
    n_rows{1},
    n_cols(lst.size()), // FIXME: narrowing.
    stride{n_cols},
    pin_row{0},
    pin_col{0},
    _data{}
{
    if (n_cols) {
        _data.reset(new ValueT[n_cols], std::default_delete<ValueT[]>());
        std::copy(lst.begin(), lst.end(), _data.get());
    }
}

template<typename ValueT>
Matrix<ValueT> Matrix<ValueT>::deep_copy() const
{
    Matrix<ValueT> tmp(n_rows, n_cols);
    for (uint i = 0; i < n_rows; ++i)
        for (uint j = 0; j < n_cols; ++j)
            tmp(i, j) = (*this)(i, j);
    return tmp;
}

template<typename ValueT>
const Matrix<ValueT> &Matrix<ValueT>::operator = (const Matrix<ValueT> &m)
{
    make_rw(n_rows) = m.n_rows;
    make_rw(n_cols) = m.n_cols;
    make_rw(stride) = m.stride;
    make_rw(pin_row) = m.pin_row;
    make_rw(pin_col) = m.pin_col;
    _data = m._data;
    return *this;
}
template<typename ValueT>
Matrix<ValueT>::Matrix(std::initializer_list<std::initializer_list<ValueT>> lsts):
    n_rows(lsts.size()), // FIXME: narrowing.
    n_cols{0},
    stride{n_cols},
    pin_row{0},
    pin_col{0},
    _data{}
{
    // check if no action is needed.
    if (n_rows == 0)
        return;

    // initializing columns count using first row.
    make_rw(n_cols) = lsts.begin()->size();
    make_rw(stride) = n_cols;

    // lambda function to check sublist length.
    // local block to invalidate stack variables after it ends.
    {
        auto local_n_cols = n_cols;
        auto chk_length = [local_n_cols](const std::initializer_list<ValueT> &l) {
            return l.size() == local_n_cols;
        };
        // checking that all row sizes are equal.
        if (not std::all_of(lsts.begin(), lsts.end(), chk_length))
            throw std::string("Initialization rows must have equal length");
    }

    if (n_cols == 0)
        return;

    // allocating matrix memory.
    _data.reset(new ValueT[n_cols * n_rows], std::default_delete<ValueT[]>());

    // copying matrix data.
    {
        auto write_ptr = _data.get();
        auto ptr_delta = n_cols;
        auto copier = [&write_ptr, ptr_delta](const std::initializer_list<ValueT> &l) {
            std::copy(l.begin(), l.end(), write_ptr);
            write_ptr += ptr_delta;
        };
        for_each(lsts.begin(), lsts.end(), copier);
    }
}

template<typename ValueT>
Matrix<ValueT>::Matrix(const Matrix &src):
    n_rows{src.n_rows},
    n_cols{src.n_cols},
    stride{src.stride},
    pin_row{src.pin_row},
    pin_col{src.pin_col},
    _data{src._data}
{
}

template<typename ValueT>
Matrix<ValueT>::Matrix(Matrix &&src):
    n_rows{src.n_rows},
    n_cols{src.n_cols},
    stride{src.stride},
    pin_row{src.pin_row},
    pin_col{src.pin_col},
    _data{src._data}
{
    // resetting state of donor object.
    make_rw(src.n_rows) = 0;
    make_rw(src.n_cols) = 0;
    make_rw(src.stride) = 0;
    make_rw(src.pin_row) = 0;
    make_rw(src.pin_col) = 0;
    src._data.reset();
}


template<typename ValueT>
ValueT &Matrix<ValueT>::operator()(uint row, uint col)
{
    if (row >= n_rows or col >= n_cols)
        throw std::string("Out of bounds while indexing");
    row += pin_row;
    col += pin_col;
    return _data.get()[row * stride + col];
}

template<typename ValueT>
const ValueT &Matrix<ValueT>::operator()(uint row, uint col) const
{
    if (row >= n_rows or col >= n_cols)
        throw std::string("Out of bounds while indexing");
    row += pin_row;
    col += pin_col;
    return _data.get()[row * stride + col];
}

template<typename ValueT>
Matrix<ValueT>::~Matrix()
{}

template<typename ValueT>
const Matrix<ValueT> Matrix<ValueT>::submatrix(uint prow, uint pcol,
                                               uint rows, uint cols) const
{
    if (prow + rows > n_rows or pcol + cols > n_cols)
        throw std::string("Out of bounds in submatrix");
    // copying requested data to submatrix.
    Matrix<ValueT> tmp(*this);
    make_rw(tmp.n_rows) = rows;
    make_rw(tmp.n_cols) = cols;
    make_rw(tmp.pin_row) = pin_row + prow;
    make_rw(tmp.pin_col) = pin_col + pcol;
    return tmp;
}

template<typename ValueT>
template<typename UnaryMatrixOperator>
Matrix<typename std::result_of<UnaryMatrixOperator(Matrix<ValueT>)>::type>
Matrix<ValueT>::unary_map(const UnaryMatrixOperator &op) const
{
    // Let's typedef return type of function for ease of usage
    typedef typename std::result_of<UnaryMatrixOperator(Matrix<ValueT>)>::type ReturnT;

    const auto r_vert = op.r_vert;
    const auto r_horz = op.r_horz;
    
    if (n_rows < r_horz || n_cols < r_vert)
        return Matrix<ReturnT>(0, 0);
        
    Matrix<ValueT> tmp = (r_vert == 0 && r_horz == 0) ? (*this) : MirrorExpand(r_vert, r_horz);

    Matrix<ReturnT> ans(n_rows, n_cols);
    
    const auto size_v = 2 * r_vert + 1;
    const auto size_h = 2 * r_horz + 1;

    const auto start_i = r_vert;
    const auto end_i = tmp.n_rows - r_vert;
    const auto start_j = r_horz;
    const auto end_j = tmp.n_cols - r_horz;

    for (uint i = start_i; i < end_i; ++i) {
        for (uint j = start_j; j < end_j; ++j) {
            auto neighbourhood = tmp.submatrix(i - r_vert, j - r_horz, size_v, size_h);
            ans(i - start_i, j - start_j) = op(neighbourhood);
        }
    }
    return ans;
}

template<typename ValueT>
Matrix<ValueT> Matrix<ValueT>::MirrorExpand(uint n_new_rows, uint n_new_cols) const
{
    if (n_rows < n_new_rows || n_cols < n_new_cols)
        throw std::string("Can't expand so much in MirrorExpand");
    
    Matrix<ValueT> result(n_rows + 2 * n_new_rows, n_cols + 2 * n_new_cols);
    
    //main part
    for (uint i = 0; i < n_rows; ++i) {
        for (uint j = 0; j < n_cols; ++j) {
            result(i + n_new_rows, j + n_new_cols) = (*this)(i, j);
        }
    }
    
    // top left
    for (uint i = 0; i < n_new_rows; ++i) {
        for (uint j = 0; j < n_new_cols; ++j) {
            result(i, j) = result(n_new_rows + (n_new_rows - i) - 1, n_new_cols + (n_new_cols - j) - 1);
        }
    }
    
    // top middle
    for (uint i = 0; i < n_new_rows; ++i) {
        for (uint j = n_new_cols; j < n_new_cols + n_cols; ++j) {
            result(i, j) = result(n_new_rows + (n_new_rows - i) - 1, j);
        }
    }
    
    // top right
    for (uint i = 0; i < n_new_rows; ++i) {
        for (uint j = n_new_cols + n_cols; j < n_new_cols + n_cols + n_new_cols; ++j) {
            result(i, j) = result(n_new_rows + (n_new_rows - i) - 1, n_cols + n_new_cols - (j - (n_cols + n_new_cols)) - 1);
        }
    }
    
    // left
    for (uint i = n_new_rows; i < n_new_rows + n_rows; ++i) {
        for (uint j = 0; j < n_new_cols; ++j) {
            result(i, j) = result(i, n_new_cols + (n_new_cols - j) - 1);
        }
    }
    
    // right
    for (uint i = n_new_rows; i < n_new_rows + n_rows; ++i) {
        for (uint j = n_new_cols + n_cols; j < n_new_cols + n_cols + n_new_cols; ++j) {
            result(i, j) = result(i, n_cols + n_new_cols - (j - (n_cols + n_new_cols)) - 1);
        }
    }
    
    // bottom left
    for (uint i = n_new_rows + n_rows; i < n_new_rows + n_rows + n_new_rows; ++i) {
        for (uint j = 0; j < n_new_cols; ++j) {
            result(i, j) = result(n_new_rows + n_rows - (i - (n_new_rows + n_rows)) - 1, n_new_cols + (n_new_cols - j) - 1);
        }
    }
    
    // bottom middle
    for (uint i = n_new_rows + n_rows; i < n_new_rows + n_rows + n_new_rows; ++i) {
        for (uint j = n_new_cols; j < n_new_cols + n_cols; ++j) {
            result(i, j) = result(n_new_rows + n_rows - (i - (n_new_rows + n_rows)) - 1, j);
        }
    }
    
    // bottom right
    for (uint i = n_new_rows + n_rows; i < n_new_rows + n_rows + n_new_rows; ++i) {
        for (uint j = n_new_cols + n_cols; j < n_new_cols + n_cols + n_new_cols; ++j) {
            result(i, j) = result(n_new_rows + n_rows - (i - (n_new_rows + n_rows)) - 1, n_cols + n_new_cols - (j - (n_cols + n_new_cols)) - 1);
        }
    }
    
    return result;
            
}

template <typename ValueT>
Matrix<ValueT> Matrix<ValueT>::rotate1() const
{
    Matrix<ValueT> result(n_cols, n_rows);
    
    for (uint i = 0; i < n_rows; ++i)
        for (uint j = 0; j < n_cols; ++j)
            result(j, n_rows - i - 1) = (*this)(i, j);
            
    return result;
}

template <typename ValueT>
Matrix<ValueT> Matrix<ValueT>::rotate_clockwise(int n) const
{
    if(n < 0 || n > 3 ) 
        throw std::string("RotateClockwise: number of rotations must be 0, 1, 2, or 3");
        
    Matrix<ValueT> result = *this;
    for (int i = 0; i < n; ++i)
        result = result.rotate1();
        
    return result;   
}

