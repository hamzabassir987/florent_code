#pragma once
#include "florent_layout.hpp"
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace florent_rl {

struct Step {
    int team=0, round=0, entity=-1, turn=0;
    UnitPolicy policy=UnitPolicy::None;
    float old_logp=0, value=0, reward=0, advantage=0, ret=0;
    bool terminal=false;
};

struct RoundValue {
    int team=0, round=0;
    float value=0, reward=0, next_value=0;
    float advantage=0, ret=0;
    bool terminal=false;
};

// Compute round-level GAE. There is exactly one transition per team and round;
// actor decisions in the same round inherit that transition's advantage.
inline void finish_gae(std::vector<RoundValue*>& seq, float gamma=GAMMA,
                       float lambda=LAMBDA, bool normalize=true) {
    if (seq.empty()) return;
    float gae=0.f;
    for (int i=(int)seq.size()-1; i>=0; --i) {
        RoundValue& r=*seq[(size_t)i];
        const float bootstrap=r.terminal ? 0.f : r.next_value;
        const float delta=r.reward + gamma*bootstrap-r.value;
        gae=delta + gamma*lambda*gae;
        r.advantage=gae;
        r.ret=r.value+gae;
    }
    if (normalize) {
        double mean=0.; for(auto* r:seq) mean += r->advantage;
        mean/=seq.size();
        double var=0.; for(auto* r:seq) { double d=r->advantage-mean; var+=d*d; }
        const float inv=1.f/std::sqrt((float)(var/seq.size())+1e-8f);
        for(auto* r:seq) r->advantage=(r->advantage-(float)mean)*inv;
    }
}

struct Trajectory {
    std::vector<Step> steps;
    std::vector<RoundValue> rounds;
    void clear(){steps.clear(); rounds.clear();}
    void add_round(const RoundValue& r){rounds.push_back(r);}

    // Supports both team-perspective rows and the legacy one-row-per-step
    // caller. Terminal transitions never bootstrap; truncated rollouts do.
    void finish(float terminal_value=0, bool normalize=true) {
        if (rounds.empty() && !steps.empty()) {
            rounds.reserve(steps.size());
            for(size_t i=0;i<steps.size();++i) {
                const Step& s=steps[i];
                rounds.push_back({s.team,s.round,s.value,s.reward,
                    s.terminal?0.f:(i+1<steps.size()?steps[i+1].value:terminal_value),
                    0.f,0.f,s.terminal});
            }
        }
        std::vector<RoundValue*> by_team[2];
        for(auto& r:rounds) if(r.team>=0 && r.team<2) by_team[r.team].push_back(&r);
        for(int t=0;t<2;++t) finish_gae(by_team[t],GAMMA,LAMBDA,normalize);
        std::unordered_map<long long,RoundValue*> lookup;
        for(auto& r:rounds) lookup[(static_cast<long long>(r.team)<<32) | (unsigned)r.round]=&r;
        for(auto& s:steps) {
            auto it=lookup.find((static_cast<long long>(s.team)<<32)|(unsigned)s.round);
            if(it==lookup.end()) continue;
            s.advantage=it->second->advantage;
            s.ret=it->second->ret;
        }
    }
};
}
