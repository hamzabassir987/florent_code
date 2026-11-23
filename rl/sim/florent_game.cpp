#include "florent_game.hpp"
namespace florent_rl {
bool Match::load(const std::string& p,uint64_t s){ map_path=p; seed=s; return game.load_map(p); }
void Match::reset(){ game.reset(seed); memories.clear(); critic_round=-1; }
std::vector<int> Match::acting_units() const { return game.turn_order(); }
bool Match::round_critic(){ if(critic_round==game.round) return false; fe::encode_critic(game,critic_channels,critic_scalars); critic_round=game.round; return true; }
int Match::terminal_reward(int team) const { if(!game.over || game.winner<0) return 0; return game.winner==team ? 1 : -1; }
int Match::act(int entity,const Candidate& c){ return apply_candidate(game,entity,c); }
void Match::observe(int entity) {
    if (!memories.count(entity)) { fe::ObsMemory m; m.init(game, game.ents[entity].team); memories.emplace(entity, std::move(m)); }
    memories.at(entity).update(game, entity);
}
void Match::encode_only(int entity, std::vector<float>& out) {
    auto& mem=memories.at(entity);
    std::vector<int32_t> ch, sc; fe::encode_actor(game, mem, entity, ch, sc);
    out.resize(ch.size()+sc.size()); for(size_t i=0;i<ch.size();++i) out[i]=static_cast<float>(ch[i]);
    for(size_t i=0;i<sc.size();++i) out[ch.size()+i]=static_cast<float>(sc[i]);
}
void Match::actor_observation(int entity, std::vector<float>& out) {
    if (!memories.count(entity)) { fe::ObsMemory m; m.init(game, game.ents[entity].team); memories.emplace(entity, std::move(m)); }
    auto& mem=memories.at(entity); mem.update(game, entity);
    std::vector<int32_t> ch, sc; fe::encode_actor(game, mem, entity, ch, sc);
    out.resize(ch.size()+sc.size()); for(size_t i=0;i<ch.size();++i) out[i]=static_cast<float>(ch[i]);
    for(size_t i=0;i<sc.size();++i) out[ch.size()+i]=static_cast<float>(sc[i]);
}
}
