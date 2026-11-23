#include "rollout.hpp"
#include <iostream>
int main(int argc,char**argv){ if(argc<2)return 2; florent_rl::Rollout r; florent_rl::RolloutConfig c; c.map=argv[1]; c.rounds=argc>2?std::atoi(argv[2]):32; int e=r.run_random(c); if(e){std::cerr<<"rollout error "<<e<<"\n";return 1;} std::cout<<"ROLLOUT OK steps="<<r.trajectory.steps.size()<<"\n"; }
