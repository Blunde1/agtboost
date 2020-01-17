// Order gamma

#ifndef __SORTING_HPP_INCLUDED__
#define __SORTING_HPP_INCLUDED__


template <typename T>
Tvec<size_t> sort_indexes(const Tvec<T> &v) {
    
    // Initialize
    Tvec<size_t> idx(v.size());
    std::iota(idx.data(), idx.data()+idx.size(), 0);
    
    // Sort with lambda functionality
    std::sort(idx.data(), idx.data() + idx.size(),
              [&v](int i1, int i2){return v[i1] < v[i2];});
    
    // Return
    return idx;
}


#endif