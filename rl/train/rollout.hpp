#pragma once
#include "../sim/florent_game.hpp"
#include "trajectory.hpp"
#include <random>
#include <string>
namespace florent_rl {
struct RolloutConfig { int rounds=32; uint64_t seed=1; std::string map; };
struct Rollout {
    Match match; Trajectory trajectory; std::mt19937 rng;
    bool start(const RolloutConfig& c){ trajectory.clear(); rng.seed((uint32_t)c.seed); if(!match.load(c.map,c.seed)) return false; match.reset(); return true; }
    int run_random(const RolloutConfig& c){
        if(!start(c)) return -1;
        for(int r=0;r<c.rounds && !match.game.over;++r){
            match.round_critic(); RoundValue rv; rv.team=0; rv.round=match.game.round;
            for(int a:match.acting_units()){
                std::vector<Candidate> cs; enumerate_candidates(match.game,a,cs); std::vector<int> legal;
                for(int i=0;i<(int)cs.size();++i) if(legal_candidate(match.game,a,cs[i])) legal.push_back(i);
                if(legal.empty()) continue;
                int k=legal[(size_t)(rng()%legal.size())]; if(match.act(a,cs[k])!=fe::OK) return 4;
                Step s; s.team=match.game.ents[a].team; s.round=match.game.round; s.entity=a; s.policy=policy_for_type(match.game.ents[a].type); trajectory.steps.push_back(s);
            }
            match.game.end_round(); rv.terminal=match.game.over; rv.reward=rv.terminal?float(match.terminal_reward(0)):0.f; trajectory.rounds.push_back(rv);
        }
        trajectory.finish(); return 0;
    }
};
}
