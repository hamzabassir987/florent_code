#include "florent_layout.hpp"
#include <cmath>
#include <limits>
#include <vector>
namespace florent_rl {
struct PolicyMath {
 static std::vector<float> masked_log_softmax(const std::vector<float>& logits,const std::vector<uint8_t>& legal,float& entropy) {
  float mx=-std::numeric_limits<float>::infinity(); int n=0; for(size_t i=0;i<logits.size();++i) if(legal[i]) {mx=std::max(mx,logits[i]);++n;}
  std::vector<float> out(logits.size(),-std::numeric_limits<float>::infinity()); if(!n){entropy=0;return out;}
  float z=0; for(size_t i=0;i<logits.size();++i) if(legal[i]) z+=std::exp(logits[i]-mx); float lz=mx+std::log(z); entropy=0;
  for(size_t i=0;i<logits.size();++i) if(legal[i]) {out[i]=logits[i]-lz; float p=std::exp(out[i]); entropy-=p*out[i];} return out;
 }
};
}
