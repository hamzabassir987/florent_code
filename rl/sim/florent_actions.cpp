#include "florent_actions.hpp"
#include <algorithm>

namespace florent_rl {
using namespace fe;

UnitPolicy policy_for_type(int t) {
    switch (t) {
    case CORE: return UnitPolicy::Core;
    case BUILDER: return UnitPolicy::Builder;
    case GUNNER: return UnitPolicy::Gunner;
    case SENTINEL: return UnitPolicy::Sentinel;
    case LAUNCHER: return UnitPolicy::Launcher;
    case SPIKE: return UnitPolicy::Spike;
    default: return UnitPolicy::None;
    }
}
const char* policy_name(UnitPolicy p) {
    switch (p) { case UnitPolicy::Core:return "core"; case UnitPolicy::Builder:return "builder";
    case UnitPolicy::Gunner:return "gunner"; case UnitPolicy::Sentinel:return "sentinel";
    case UnitPolicy::Launcher:return "launcher"; case UnitPolicy::Spike:return "spike"; default:return "none"; }
}

static void offsets(std::vector<std::pair<int,int>>& v, int radius=1, bool include_origin=false) {
    for (int y=-radius;y<=radius;++y)
        for (int x=-radius;x<=radius;++x)
            if (include_origin || x || y) v.emplace_back(x,y);
}

// Candidate positions are fixed windows around the actor. Coordinates are
// absolute in the candidate payload, but the index/order is independent of
// map dimensions and therefore remains checkpoint compatible.
void enumerate_candidates(const Game& g, int a, std::vector<Candidate>& out) {
    out.clear(); out.push_back({ActionKind::Wait});
    if (a < 0 || a >= (int)g.ents.size() || !g.ents[a].alive) return;
    const auto& e=g.ents[a]; std::vector<std::pair<int,int>> os;
    if (e.type==BUILDER) {
        // Movement is cardinal and mutually exclusive with every other action.
        for (int d : {N,E,S,W}) out.push_back({ActionKind::Move,d});
        offsets(os,1);
        const int types[] = {CONVEYOR,SPLITTER,HARVESTER,GUNNER,SENTINEL,LAUNCHER,SPIKE,BARRIER,MINE};
        for (int t:types) for (auto [dx,dy]:os) for (int d=C; d<=NW; ++d)
            out.push_back({ActionKind::Build,t,e.x+dx,e.y+dy,d});
        for (auto [dx,dy]:os) {
            out.push_back({ActionKind::Attack,e.x+dx,e.y+dy});
            out.push_back({ActionKind::Heal,e.x+dx,e.y+dy});
            out.push_back({ActionKind::Destroy,e.x+dx,e.y+dy});
        }
    } else if (e.type==CORE) {
        offsets(os,2);
        for (auto [dx,dy]:os) out.push_back({ActionKind::Spawn,e.x+dx,e.y+dy});
        for (int amount: {1,2,4,8,16}) out.push_back({ActionKind::Convert,amount});
    } else if (e.type==GUNNER || e.type==SENTINEL || e.type==SPIKE) {
        offsets(os,5);
        for (auto [dx,dy]:os) out.push_back({ActionKind::Fire,e.x+dx,e.y+dy});
        if (e.type==GUNNER) for (int d=N;d<=NW;++d) out.push_back({ActionKind::Rotate,d});
    } else if (e.type==LAUNCHER) {
        std::vector<std::pair<int,int>> dest; offsets(dest,5);
        offsets(os,1);
        for (auto [bx,by]:os) for (auto [dx,dy]:dest)
            out.push_back({ActionKind::Launch,e.x+bx,e.y+by,e.x+dx,e.y+dy});
    }
}
bool legal_candidate(const Game& g,int a,const Candidate& c) {
    switch(c.kind) {
    case ActionKind::Wait:return true;
    case ActionKind::Move:return g.can_move(a,c.a)==OK;
    case ActionKind::Spawn:return g.can_spawn(a,c.a,c.b)==OK;
    case ActionKind::Convert:return g.can_convert(a,c.a)==OK;
    case ActionKind::Build:return g.can_build(a,c.a,c.b,c.c,c.d)==OK;
    case ActionKind::Attack:return g.can_attack(a,c.a,c.b)==OK;
    case ActionKind::Heal:return g.can_heal(a,c.a,c.b)==OK;
    case ActionKind::Destroy:return g.can_destroy(a,c.a,c.b)==OK;
    case ActionKind::Fire:return g.can_fire(a,c.a,c.b)==OK;
    case ActionKind::Rotate:return g.can_rotate(a,c.a)==OK;
    case ActionKind::Launch:return g.can_launch(a,c.a,c.b,c.c,c.d)==OK;
    }
    return false;
}
int apply_candidate(Game& g,int a,const Candidate& c) {
    switch(c.kind) {
    case ActionKind::Wait:return OK;
    case ActionKind::Move:return g.move(a,c.a);
    case ActionKind::Spawn:return g.spawn(a,c.a,c.b);
    case ActionKind::Convert:return g.convert_ammo(a,c.a);
    case ActionKind::Build:return g.build(a,c.a,c.b,c.c,c.d);
    case ActionKind::Attack:return g.attack(a,c.a,c.b);
    case ActionKind::Heal:return g.heal(a,c.a,c.b);
    case ActionKind::Destroy:return g.destroy(a,c.a,c.b);
    case ActionKind::Fire:return g.fire(a,c.a,c.b);
    case ActionKind::Rotate:return g.rotate(a,c.a);
    case ActionKind::Launch:return g.launch(a,c.a,c.b,c.c,c.d);
    }
    return E_BAD_ARG;
}
}
