#ifndef PROB_H
#define PROB_H

#include <memory>
#include <vector>

#include "BCStruct.H"

namespace virta {

enum class Staggering {
    noStag,
    iStag,
    jStag
};

template<typename Real, typename FieldXD>
class Prob {
public:
    const size_t ni;
    const size_t nj;
    
    std::vector<std::unique_ptr<FieldXD>> fields;

    Prob(size_t ni_, size_t nj_) : ni(ni_), nj(nj_) {}

    FieldXD& add(Real value, virta::BCStruct<Real, 2> BC, int gcm, int stag = -1) {
        auto ptr = std::make_unique<FieldXD>(ni, nj, value, BC, gcm, stag);
        FieldXD& ref = *ptr;
        fields.push_back(std::move(ptr));
        return ref;
    }
    
    inline void setBC() {
        for (auto& fptr : fields) {
            FieldXD& f = *fptr;
            f.setBC();
        }
    }

};

} // namespace virta

#endif
