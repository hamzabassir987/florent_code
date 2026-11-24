#include "florent_game.hpp"
namespace florent_rl {
bool Match::load(const std::string& p,uint64_t s){ map_path=p; seed=s; return game.load_map(p); }
bool Match::load_bytes(const std::string& b,uint64_t s){ map_path="(generated)"; seed=s; return game.load_map_bytes(reinterpret_cast<const uint8_t*>(b.data()), b.size()); }
void Match::reset(){ game.reset(seed); memories.clear(); critic_round=-1; }
std::vector<int> Match::acting_units() const { return game.turn_order(); }
bool Match::round_critic(){ if(critic_round==game.round) return false; fe::encode_critic(game,critic_channels,critic_scalars); critic_round=game.round; return true; }
int Match::terminal_reward(int team) const { if(!game.over || game.winner<0) return 0; return game.winner==team ? 1 : -1; }
int Match::act(int entity,const Candidate& c){ return apply_candidate(game,entity,c); }
void Match::observe(int entity) {
    if (!memories.count(entity)) { fe::ObsMemory m; m.init(game, game.ents[entity].team); memories.emplace(entity, std::move(m)); }
    fe::ObsMemory& mem = memories.at(entity);
    // team report bus (as bots/econ/encoder.py): merge the store snapshot, see, post one report
    const int team = game.ents[entity].team;
    mem.merge_store(game.teams[team].store.data());
    mem.update(game, entity);
    if (const uint32_t rep = mem.next_report())
        game.write_store(team, int((uint32_t(game.ents[entity].id) + uint32_t(game.round)) % 16u), int64_t(rep));
}
void Match::encode_only(int entity, std::vector<float>& out) {
    auto& mem=memories.at(entity);
    std::vector<int32_t> ch, sc; fe::encode_actor(game, mem, entity, ch, sc);
    out.resize(ch.size()+sc.size()); for(size_t i=0;i<ch.size();++i) out[i]=static_cast<float>(ch[i]);
    for(size_t i=0;i<sc.size();++i) out[ch.size()+i]=static_cast<float>(sc[i]);
}
long Match::encode_compact(int entity, uint8_t* sp, int16_t* sc) {
    fe::encode_actor(game, memories.at(entity), entity, enc_ch, enc_sc);
    long oob = 0;
    const int32_t* c = enc_ch.data();
    for (int i = 0; i < fe::ACTOR_SPATIAL; i++) {
        int32_t v = c[i];
        if (v < 0 || v > 255) { oob++; v = v < 0 ? 0 : 255; }
        sp[i] = uint8_t(v);
    }
    for (int i = 0; i < fe::ACTOR_SCALARS; i++) {
        int32_t v = enc_sc[i];
        sc[i] = int16_t(v < -32768 ? -32768 : v > 32767 ? 32767 : v);
    }
    return oob;
}
void Match::actor_observation(int entity, std::vector<float>& out) {
    if (!memories.count(entity)) { fe::ObsMemory m; m.init(game, game.ents[entity].team); memories.emplace(entity, std::move(m)); }
    auto& mem=memories.at(entity); mem.update(game, entity);
    std::vector<int32_t> ch, sc; fe::encode_actor(game, mem, entity, ch, sc);
    out.resize(ch.size()+sc.size()); for(size_t i=0;i<ch.size();++i) out[i]=static_cast<float>(ch[i]);
    for(size_t i=0;i<sc.size();++i) out[ch.size()+i]=static_cast<float>(sc[i]);
}
}
