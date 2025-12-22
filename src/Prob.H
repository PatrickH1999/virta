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
    const size_t ni_;
    const size_t nj_;
    
    std::vector<std::unique_ptr<FieldXD>> fields;

    Prob(size_t ni, size_t nj) : ni_(ni), nj_(nj) {}

    template<Staggering stag = Staggering::noStag>
    FieldXD& add(Real value, virta::BCStruct<Real, 2> BC, int gcm) {
        int ni = ni_;
        int nj = nj_;
        if constexpr (stag == Staggering::iStag) {
            ni++;
        } else if constexpr (stag == Staggering::jStag) {
            nj++;
        }
        auto ptr = std::make_unique<FieldXD>(ni, nj, value, BC, gcm);
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
