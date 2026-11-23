#include "../sim/florent_game.hpp"
#include "../train/rollout.hpp"
#include <fstream>
#include <iostream>
#include <random>
#include <unordered_map>

namespace {
struct TablePolicy {
 std::unordered_map<int,float> logit;
 int choose(const std::vector<florent_rl::Candidate>& cs, const fe::Game& g, int a, std::mt19937& rng) {
  std::vector<int> legal; for(int i=0;i<(int)cs.size();++i) if(florent_rl::legal_candidate(g,a,cs[i])) legal.push_back(i);
  if(legal.empty()) return -1; int best=legal[0]; for(int i:legal) if(logit[i]>logit[best]) best=i;
  if((rng()%100)<20) best=legal[rng()%legal.size()]; return best;
 }
 void update(const std::vector<int>& chosen,float reward) { for(int i:chosen) logit[i]+=0.01f*reward; }
 void save(const std::string& p) const { std::ofstream f(p); for(auto&kv:logit) f<<kv.first<<" "<<kv.second<<"\n"; }
};
}
int main(int argc,char**argv){
 if(argc<3){std::cerr<<"usage: florent_train MAP ITERATIONS [OUT]\n";return 2;}
 std::string map=argv[1], out=argc>3?argv[3]:"florent_policy.tbl"; int it=std::atoi(argv[2]); TablePolicy p; std::mt19937 rng(7); long wins=0;
 for(int n=0;n<it;++n){ florent_rl::Match m; if(!m.load(map,n+1))return 3; m.reset(); std::vector<int> picks; for(int r=0;r<fe::MAX_TURNS&&!m.game.over;++r){ for(int a:m.acting_units()){ std::vector<florent_rl::Candidate> cs; florent_rl::enumerate_candidates(m.game,a,cs); int k=p.choose(cs,m.game,a,rng); if(k<0)continue; if(m.act(a,cs[k])!=fe::OK) continue; picks.push_back(k); } m.game.end_round(); }
  int reward=m.terminal_reward(0); if(reward>0)++wins; p.update(picks,(float)reward); if((n+1)%10==0)std::cout<<"iteration="<<n+1<<" reward="<<reward<<" wins="<<wins<<"\n";
 }
 p.save(out); std::cout<<"TRAIN COMPLETE wins="<<wins<<"/"<<it<<" output="<<out<<"\n";
}
