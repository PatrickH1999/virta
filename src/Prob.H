#ifndef PROB_H
#define PROB_H

#include <memory>
#include <vector>

namespace virta {

template<typename Real, typename FieldXD>
class Prob {
public:
    std::vector<std::unique_ptr<FieldXD>> fields;

    Prob(size_t ni, size_t nj) : ni_(ni), nj_(nj) {}

    FieldXD& add(Real value_) {
        auto ptr = std::make_unique<FieldXD>(ni_, nj_, value_);
        FieldXD& ref = *ptr;
        fields.push_back(std::move(ptr));
        return ref;
    }

private:
    const size_t ni_;
    const size_t nj_;

};

} // namespace virta

#endif
