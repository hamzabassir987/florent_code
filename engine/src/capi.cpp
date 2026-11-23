// Plain C interface over fe::Game, for ctypes (verifier) and other languages.
#include "fengine.hpp"
#include "obs.hpp"
using fe::Game;
extern "C" {
Game* fe_new() { return new Game(); }
void fe_free(Game* g) { delete g; }
int fe_load_map(Game* g, const char* path) { return g->load_map(path) ? 1 : 0; }
void fe_reset(Game* g, unsigned long long seed) { g->reset(seed); }
int fe_width(Game* g) { return g->W; }
int fe_height(Game* g) { return g->H; }
int fe_round(Game* g) { return g->round; }
int fe_winner(Game* g) { return g->winner; }
void fe_win_condition(Game* g, char* out, int n) { int k = 0; for (char c : g->win_condition) { if (k + 1 >= n) break; out[k++] = c; } out[k] = 0; }
int fe_n_entities(Game* g) { return int(g->ents.size()); }
int fe_find(Game* g, int id) { return g->find(id); }
// id,type,team,x,y,hp,max_hp,alive,dir,acd,mcd,stored,res
void fe_entity(Game* g, int i, int* o) {
    const fe::Entity& e = g->ents[i];
    int v[13] = {e.id, e.type, e.team, e.x, e.y, e.hp, e.max_hp, e.alive, e.dir, e.acd, e.mcd, e.stored, e.res};
    for (int k = 0; k < 13; k++) o[k] = v[k];
}
// ti,vo,ammo,ti_collected,vo_collected,scale,units,core_index
void fe_team(Game* g, int t, int* o) {
    const fe::Team& m = g->teams[t];
    int v[8] = {m.ti, m.vo, m.ammo, m.ti_collected, m.vo_collected, m.scale, m.units, m.core};
    for (int k = 0; k < 8; k++) o[k] = v[k];
}
int fe_building_at(Game* g, int x, int y) { return g->inb(x, y) ? g->bld[g->idx(x, y)] : -1; }
int fe_bot_at(Game* g, int x, int y) { return g->inb(x, y) ? g->bot[g->idx(x, y)] : -1; }
int fe_turn_order(Game* g, int* out, int max) {
    auto v = g->turn_order(); int n = 0;
    for (int i : v) { if (n >= max) break; out[n++] = i; }
    return n;
}
int fe_move(Game* g, int a, int d) { return g->move(a, d); }
int fe_build(Game* g, int a, int t, int x, int y, int d) { return g->build(a, t, x, y, d); }
int fe_attack(Game* g, int a, int x, int y) { return g->attack(a, x, y); }
int fe_heal(Game* g, int a, int x, int y) { return g->heal(a, x, y); }
int fe_destroy(Game* g, int a, int x, int y) { return g->destroy(a, x, y); }
int fe_self_destruct(Game* g, int a) { return g->self_destruct(a); }
int fe_spawn(Game* g, int c, int x, int y) { return g->spawn(c, x, y); }
int fe_convert_ammo(Game* g, int c, int amount) { return g->convert_ammo(c, amount); }
int fe_fire(Game* g, int a, int x, int y) { return g->fire(a, x, y); }
int fe_rotate(Game* g, int a, int d) { return g->rotate(a, d); }
int fe_launch(Game* g, int a, int bx, int by, int tx, int ty) { return g->launch(a, bx, by, tx, ty); }
int fe_can_move(Game* g, int a, int d) { return g->can_move(a, d); }
int fe_can_build(Game* g, int a, int t, int x, int y, int d) { return g->can_build(a, t, x, y, d); }
int fe_can_attack(Game* g, int a, int x, int y) { return g->can_attack(a, x, y); }
int fe_can_heal(Game* g, int a, int x, int y) { return g->can_heal(a, x, y); }
int fe_can_destroy(Game* g, int a, int x, int y) { return g->can_destroy(a, x, y); }
int fe_can_spawn(Game* g, int c, int x, int y) { return g->can_spawn(c, x, y); }
int fe_can_convert(Game* g, int c, int amount) { return g->can_convert(c, amount); }
int fe_can_fire(Game* g, int a, int x, int y) { return g->can_fire(a, x, y); }
int fe_can_rotate(Game* g, int a, int d) { return g->can_rotate(a, d); }
int fe_can_launch(Game* g, int a, int bx, int by, int tx, int ty) { return g->can_launch(a, bx, by, tx, ty); }
int fe_vis_tiles(Game* g, int a, int r2, int* out, int max) {
    std::vector<int> v; g->nearby_tiles(a, r2, v); int n = 0;
    for (int t : v) { if (n >= max) break; out[2 * n] = t % g->W; out[2 * n + 1] = t / g->W; n++; }
    return n;
}
int fe_vis_ids(Game* g, int a, int r2, int kind, int* out, int max) {
    std::vector<int> v; g->nearby_ids(a, r2, kind, v); int n = 0;
    for (int id : v) { if (n >= max) break; out[n++] = id; }
    return n;
}
// empty, passable, building id (-1), bot id (-1), env
void fe_vis_tile(Game* g, int a, int x, int y, int* o) {
    int b = g->tile_building(a, x, y), u = g->bot[g->idx(x, y)];
    o[0] = g->tile_empty(a, x, y); o[1] = g->tile_passable(a, x, y);
    o[2] = b >= 0 ? g->ents[b].id : -1; o[3] = u >= 0 ? g->ents[u].id : -1; o[4] = g->env[g->idx(x, y)];
}
int fe_write_store(Game* g, int t, int i, long long v) { return g->write_store(t, i, v); }
long long fe_get_store(Game* g, int t, int i) { return g->teams[t].store[i]; }
void fe_end_round(Game* g) { g->end_round(); }
int fe_last_moves(Game* g, int* out, int max) {
    int n = 0;
    for (auto& m : g->last_moves) { if (n >= max) break; int* o = out + 6 * n++; o[0] = m.fx; o[1] = m.fy; o[2] = m.tx; o[3] = m.ty; o[4] = m.res; o[5] = int(m.stack_id); }
    return n;
}
// ---- debug hooks used by the verifier to resynchronise after a mismatch
void fe_dbg_set_stored(Game* g, int x, int y, int res, int sid) { int b = g->bld[g->idx(x, y)]; if (b >= 0) { g->ents[b].stored = (uint8_t)res; g->ents[b].stack_id = (uint32_t)sid; } }
void fe_dbg_set_hcd(Game* g, int i, int v) { g->ents[i].hcd = v; }
void fe_dbg_set_team(Game* g, int t, int ti, int vo, int ammo) { g->teams[t].ti = ti; g->teams[t].vo = vo; g->teams[t].ammo = ammo; }
void fe_dbg_set_hp(Game* g, int i, int hp) { g->ents[i].hp = hp; }
void fe_dbg_kill(Game* g, int i) { g->kill(i); }
void fe_dbg_set_sides(Game* g, int i, const int* in4, const int* out4) { for (int k = 0; k < 4; k++) { g->ents[i].in_last[k] = in4[k]; g->ents[i].out_last[k] = out4[k]; } }
void fe_dbg_set_last_out(Game* g, int i, int r) { g->ents[i].last_out = r; }

// ---- observation encoder
void* fe_obs_new(Game* g, int team) {
    fe::ObsMemory* m = new fe::ObsMemory();
    m->init(*g, team);
    return m;
}
void fe_obs_free(void* p) { delete reinterpret_cast<fe::ObsMemory*>(p); }
void fe_obs_update(void* p, Game* g, int actor) { reinterpret_cast<fe::ObsMemory*>(p)->update(*g, actor); }
int fe_obs_ch() { return fe::ACTOR_CH; }
int fe_obs_patch() { return fe::PATCH; }
int fe_obs_spatial() { return fe::ACTOR_SPATIAL; }
int fe_obs_nscalars() { return fe::ACTOR_SCALARS; }
void fe_obs_encode(void* p, Game* g, int actor, int* out, int* scal) {
    std::vector<int32_t> ch, sc;
    fe::encode_actor(*g, *reinterpret_cast<fe::ObsMemory*>(p), actor, ch, sc);
    for (size_t i = 0; i < ch.size(); i++) out[i] = ch[i];
    for (size_t i = 0; i < sc.size(); i++) scal[i] = sc[i];
}
int fe_critic_ch() { return fe::CRITIC_CH; }
int fe_critic_pad() { return fe::CRITIC_PAD; }
int fe_critic_spatial() { return fe::CRITIC_SPATIAL; }
int fe_critic_nscalars() { return fe::CRITIC_SCALARS; }
void fe_critic_encode(Game* g, int* out, int* scal) {
    std::vector<int32_t> ch, sc;
    fe::encode_critic(*g, ch, sc);
    for (size_t i = 0; i < ch.size(); i++) out[i] = ch[i];
    for (size_t i = 0; i < sc.size(); i++) scal[i] = sc[i];
}
}
