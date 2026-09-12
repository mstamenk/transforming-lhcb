"""Compiled, PID-blind geometry helpers for the compact transformer dataset."""

from __future__ import annotations


CPP_ML_HELPERS = r"""
#ifndef LHCB_MASKED_PID_ML_RDF_HELPERS
#define LHCB_MASKED_PID_ML_RDF_HELPERS

#include <ROOT/RVec.hxx>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <set>
#include <tuple>
#include <vector>

namespace lhcb_ml {

using ROOT::RVecF;
using ROOT::RVecI;

constexpr float kMissing = -999.f;
constexpr double kPionMassMeV = 139.57039;

struct Vec3 { double x=0., y=0., z=0.; };
Vec3 operator+(const Vec3 &a,const Vec3 &b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
Vec3 operator-(const Vec3 &a,const Vec3 &b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
Vec3 operator*(double s,const Vec3 &a){return {s*a.x,s*a.y,s*a.z};}
double dot(const Vec3&a,const Vec3&b){return a.x*b.x+a.y*b.y+a.z*b.z;}
double norm(const Vec3&a){return std::sqrt(dot(a,a));}
Vec3 unit(const Vec3&a){const double n=norm(a);return n>0?(1./n)*a:Vec3{};}

struct Fit {
  bool valid=false;
  Vec3 position{};
  double rms=kMissing, max_doca=kMissing;
};

bool solve3(const double a[3][3],const double b[3],Vec3 &out){
  const double d=a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])
    -a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])
    +a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]);
  if(!std::isfinite(d)||std::abs(d)<1e-12)return false;
  const double inv[3][3]={
    {(a[1][1]*a[2][2]-a[1][2]*a[2][1])/d,(a[0][2]*a[2][1]-a[0][1]*a[2][2])/d,(a[0][1]*a[1][2]-a[0][2]*a[1][1])/d},
    {(a[1][2]*a[2][0]-a[1][0]*a[2][2])/d,(a[0][0]*a[2][2]-a[0][2]*a[2][0])/d,(a[0][2]*a[1][0]-a[0][0]*a[1][2])/d},
    {(a[1][0]*a[2][1]-a[1][1]*a[2][0])/d,(a[0][1]*a[2][0]-a[0][0]*a[2][1])/d,(a[0][0]*a[1][1]-a[0][1]*a[1][0])/d}};
  out={inv[0][0]*b[0]+inv[0][1]*b[1]+inv[0][2]*b[2],
       inv[1][0]*b[0]+inv[1][1]*b[1]+inv[1][2]*b[2],
       inv[2][0]*b[0]+inv[2][1]*b[1]+inv[2][2]*b[2]};
  return std::isfinite(out.x)&&std::isfinite(out.y)&&std::isfinite(out.z);
}

Fit fit_lines(const std::vector<Vec3>&r,const std::vector<Vec3>&p){
  if(r.size()<2||r.size()!=p.size())return {};
  double a[3][3]={},b[3]={};
  std::vector<Vec3> u; u.reserve(p.size());
  for(std::size_t i=0;i<p.size();++i){
    u.push_back(unit(p[i])); if(norm(u.back())==0)return {};
    const double uu[3]={u.back().x,u.back().y,u.back().z};
    const double rr[3]={r[i].x,r[i].y,r[i].z};
    for(int row=0;row<3;++row)for(int col=0;col<3;++col){
      const double q=(row==col?1.:0.)-uu[row]*uu[col];
      a[row][col]+=q; b[row]+=q*rr[col];
    }
  }
  Vec3 v; if(!solve3(a,b,v))return {};
  double ss=0.,md=0.;
  for(std::size_t i=0;i<r.size();++i){
    const Vec3 d=v-r[i]; const Vec3 t=d-dot(d,u[i])*u[i];
    const double distance=norm(t); ss+=distance*distance; md=std::max(md,distance);
  }
  return {true,v,std::sqrt(ss/r.size()),md};
}

double signed_flight(const Vec3&a,const Vec3&b,const Vec3&p){return dot(b-a,unit(p));}
double pointing(const Vec3&a,const Vec3&b,const Vec3&p){
  const Vec3 f=b-a; const double d=norm(f)*norm(p); return d>0?dot(f,p)/d:-2.;
}
double line_ip(const Vec3&point,const Vec3&origin,const Vec3&direction){
  const Vec3 d=point-origin,u=unit(direction); return norm(d-dot(d,u)*u);
}
double mass_pions(const std::vector<Vec3>&momenta){
  Vec3 p{}; double e=0.;
  for(const auto&m:momenta){p=p+m;e+=std::sqrt(dot(m,m)+kPionMassMeV*kPionMassMeV);}
  return std::sqrt(std::max(0.,e*e-dot(p,p)));
}
double corrected_mass(double mass,const Vec3&momentum,const Vec3&flight){
  const Vec3 direction=unit(flight);
  if(norm(direction)==0)return mass;
  const Vec3 transverse=momentum-dot(momentum,direction)*direction;
  const double pt=norm(transverse);
  return std::sqrt(std::max(0.,mass*mass+pt*pt))+pt;
}

std::uint64_t mix64(std::uint64_t x){
  x+=0x9e3779b97f4a7c15ULL; x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
  x=(x^(x>>27))*0x94d049bb133111ebULL; return x^(x>>31);
}
std::uint64_t event_uid(unsigned int source,unsigned int run,unsigned long long event){
  return mix64(mix64(static_cast<std::uint64_t>(source))^
               mix64(static_cast<std::uint64_t>(run))^
               mix64(static_cast<std::uint64_t>(event)));
}
int event_split(unsigned int source,unsigned int run,unsigned long long event,unsigned long long seed,
                double train_fraction,double validation_fraction,double analysis_fraction){
  const auto h=mix64(event_uid(source,run,event)^seed);
  const double u=static_cast<double>(h>>11)*(1.0/9007199254740992.0);
  if(u<train_fraction)return 0;
  if(u<train_fraction+validation_fraction)return 1;
  if(u<train_fraction+validation_fraction+analysis_fraction)return 3;
  return 2;
}

template<class T> float first_or(const ROOT::VecOps::RVec<T>&x,float fallback=kMissing){
  return x.empty()?fallback:static_cast<float>(x[0]);
}

struct VertexCandidate {
  std::array<int,4> tracks{{-1,-1,-1,-1}};
  int n=0; Fit fit{}; Vec3 momentum{}; double mass=0.,corrected_mass=0.,flight=0.,ip=0.,point=0.;
  double min_ipchi2=0.,sum_ipchi2=0.,proxy=-1e30; int charge=0;
};
struct ChainCandidate {
  VertexCandidate child{}; int child_index=-1,bachelor=-1; Fit parent{}; Vec3 parent_p{};
  double parent_mass=0.,parent_corrected_mass=0.,parent_flight=0.,parent_ip=0.,parent_point=0.;
  double child_flight=0.,child_point=0.,child_ip=0.,child_corrected_mass=0.,proxy=-1e30;
};
bool better_vertex(const VertexCandidate&a,const VertexCandidate&b){
  if(a.proxy!=b.proxy)return a.proxy>b.proxy;
  if(a.n!=b.n)return a.n<b.n;
  return a.tracks<b.tracks;
}
bool better_chain(const ChainCandidate&a,const ChainCandidate&b){
  if(a.proxy!=b.proxy)return a.proxy>b.proxy;
  if(a.child.tracks!=b.child.tracks)return a.child.tracks<b.child.tracks;
  return a.bachelor<b.bachelor;
}

struct JetRecord {
  RVecI particle_valid,original_index,charge,has_track,has_pid,has_muon_pid,has_calo;
  RVecI target_species,target_reco_id,target_valid_e,target_valid_k,target_valid_p,target_valid_pi,target_valid_mu;
  RVecF log_pt,log_p,log_e,pt_fraction,e_fraction,delta_eta,delta_phi,px,py,pz,energy;
  RVecF ip,ip_raw,log1p_ipchi2,track_chi2,qoverp,state_dx,state_dy,state_dz,dir_x,dir_y,dir_z;
  RVecF calo_ecal,calo_hcal2ecal,calo_e49,calo_prs;
  RVecF target_nne,target_nnk,target_nnp,target_nnpi,target_nnmu;
  RVecI vertex_valid,vertex_n_tracks,vertex_track0,vertex_track1,vertex_track2,vertex_track3,vertex_charge;
  RVecF vertex_x,vertex_y,vertex_z,vertex_rms,vertex_max_doca,vertex_flight_pv,vertex_ip_pv,vertex_pointing;
  RVecF vertex_px,vertex_py,vertex_pz,vertex_pt,vertex_mass_pi,vertex_corrected_mass_pi,vertex_min_ipchi2,vertex_sum_ipchi2,vertex_fit_proxy;
  RVecI chain_valid,chain_child_n_tracks,chain_track0,chain_track1,chain_track2,chain_track3;
  RVecI chain_bachelor,chain_child_vertex_index,chain_charge;
  RVecF chain_parent_x,chain_parent_y,chain_parent_z,chain_parent_rms,chain_parent_max_doca;
  RVecF chain_parent_flight_pv,chain_parent_ip_pv,chain_parent_pointing,chain_child_flight,chain_child_pointing,chain_child_ip_pv;
  RVecF chain_child_mass_pi,chain_child_corrected_mass_pi,chain_parent_mass_pi,chain_parent_corrected_mass_pi,chain_parent_pt,chain_fit_proxy;
  int n_particles_input=0,n_particles_stored=0,n_track_lines=0;
  int n_pair_total=0,n_triplet_total=0,n_quad_total=0,n_chain_total=0;
  int vertices_truncated=0,chains_truncated=0,particles_truncated=0;
};

template<class T> T at_or(const ROOT::VecOps::RVec<T>&x,std::size_t i,T fallback){return i<x.size()?x[i]:fallback;}
float finite_or(float x,float fallback=kMissing){return std::isfinite(x)?x:fallback;}
bool valid_state(std::size_t i,const RVecF&q,const RVecF&x,const RVecF&y,const RVecF&z,
                 const RVecF&px,const RVecF&py,const RVecF&pz){
  if(i>=q.size()||(q[i]!=1.f&&q[i]!=-1.f)||i>=x.size()||i>=y.size()||i>=z.size())return false;
  if(!std::isfinite(x[i])||!std::isfinite(y[i])||!std::isfinite(z[i])||x[i]<=-900||y[i]<=-900||z[i]<=-900)return false;
  return i<px.size()&&i<py.size()&&i<pz.size()&&std::isfinite(px[i])&&std::isfinite(py[i])&&std::isfinite(pz[i])
    &&(px[i]*px[i]+py[i]*py[i]+pz[i]*pz[i])>0;
}

VertexCandidate make_vertex(const std::array<int,4>&ids,int n,const RVecF&px,const RVecF&py,const RVecF&pz,
    const RVecF&q,const RVecF&ipchi2,const RVecF&x,const RVecF&y,const RVecF&z,const Vec3&pv){
  VertexCandidate c; c.tracks=ids;c.n=n;
  std::vector<Vec3> r,p; r.reserve(n);p.reserve(n);
  c.min_ipchi2=std::numeric_limits<double>::infinity();
  for(int j=0;j<n;++j){const int i=ids[j];r.push_back({x[i],y[i],z[i]});p.push_back({px[i],py[i],pz[i]});
    c.momentum=c.momentum+p.back();c.charge+=static_cast<int>(q[i]);
    const double ip2=i<static_cast<int>(ipchi2.size())&&std::isfinite(ipchi2[i])?std::max(0.f,ipchi2[i]):0.;
    c.min_ipchi2=std::min(c.min_ipchi2,ip2);c.sum_ipchi2+=ip2;}
  c.fit=fit_lines(r,p); if(!c.fit.valid)return c;
  c.mass=mass_pions(p);c.flight=signed_flight(pv,c.fit.position,c.momentum);
  c.point=pointing(pv,c.fit.position,c.momentum);c.ip=line_ip(pv,c.fit.position,c.momentum);
  c.corrected_mass=corrected_mass(c.mass,c.momentum,c.fit.position-pv);
  c.proxy=std::log1p(c.sum_ipchi2)+0.35*std::log1p(std::max(0.,c.flight))-2.*std::log1p(c.fit.max_doca/0.05);
  return c;
}

JetRecord build_jet_record(
  const RVecF&e,const RVecF&pt,const RVecF&id,const RVecF&px,const RVecF&py,const RVecF&pz,
  const RVecF&eta,const RVecF&phi,const RVecF&q,const RVecF&ip,const RVecF&ipraw,const RVecF&ipchi2,
  const RVecF&nne,const RVecF&nnk,const RVecF&nnp,const RVecF&nnpi,const RVecF&nnmu,
  const RVecF&chi2,const RVecF&qoverp,const RVecF&tx,const RVecF&ty,const RVecF&tz,
  const RVecF&tvx,const RVecF&tvy,const RVecF&tvz,const RVecF&cecal,const RVecF&ch2e,const RVecF&ce49,const RVecF&cprs,
  double jet_pt,double jet_eta,double jet_phi,double jet_e,double pv_x,double pv_y,double pv_z,
  int max_particles,int max_pair_vertices,int max_triplet_vertices,int max_quad_vertices,int max_chains,
  double max_vertex_doca,double min_vertex_flight,double min_chain_pointing,double max_abs_state_z,
  double min_vertex_track_pt,double min_vertex_track_ipchi2){
  JetRecord o; const int n=static_cast<int>(std::min({e.size(),pt.size(),px.size(),py.size(),pz.size(),eta.size(),phi.size(),q.size()}));
  o.n_particles_input=n;o.n_particles_stored=std::min(n,max_particles);o.particles_truncated=n>max_particles;
  auto pad_i=[&](RVecI&v,int x=0){v=RVecI(max_particles,x);}; auto pad_f=[&](RVecF&v,float x=kMissing){v=RVecF(max_particles,x);};
  pad_i(o.particle_valid);pad_i(o.original_index,-1);pad_i(o.charge);pad_i(o.has_track);pad_i(o.has_pid);pad_i(o.has_muon_pid);pad_i(o.has_calo);
  pad_i(o.target_species,-1);pad_i(o.target_reco_id);pad_i(o.target_valid_e);pad_i(o.target_valid_k);pad_i(o.target_valid_p);pad_i(o.target_valid_pi);pad_i(o.target_valid_mu);
  for(auto*v:{&o.log_pt,&o.log_p,&o.log_e,&o.pt_fraction,&o.e_fraction,&o.delta_eta,&o.delta_phi,&o.px,&o.py,&o.pz,&o.energy,
      &o.ip,&o.ip_raw,&o.log1p_ipchi2,&o.track_chi2,&o.qoverp,&o.state_dx,&o.state_dy,&o.state_dz,&o.dir_x,&o.dir_y,&o.dir_z,
      &o.calo_ecal,&o.calo_hcal2ecal,&o.calo_e49,&o.calo_prs,&o.target_nne,&o.target_nnk,&o.target_nnp,&o.target_nnpi,&o.target_nnmu})pad_f(*v);
  const Vec3 pv{pv_x,pv_y,pv_z}; std::vector<int> tracks;
  for(int i=0;i<o.n_particles_stored;++i){
    o.particle_valid[i]=1;o.original_index[i]=i;o.charge[i]=static_cast<int>(std::lrint(q[i]));
    const double p=std::sqrt(px[i]*px[i]+py[i]*py[i]+pz[i]*pz[i]);
    o.log_pt[i]=std::log1p(std::max(0.f,pt[i]));o.log_p[i]=std::log1p(std::max(0.,p));o.log_e[i]=std::log1p(std::max(0.f,e[i]));
    o.pt_fraction[i]=jet_pt>0?pt[i]/jet_pt:kMissing;o.e_fraction[i]=jet_e>0?e[i]/jet_e:kMissing;
    o.delta_eta[i]=eta[i]-jet_eta;double dp=phi[i]-jet_phi;while(dp> M_PI)dp-=2*M_PI;while(dp<=-M_PI)dp+=2*M_PI;o.delta_phi[i]=dp;
    o.px[i]=px[i];o.py[i]=py[i];o.pz[i]=pz[i];o.energy[i]=e[i];
    // Preserve every valid stored track state as particle context. The state-z
    // window is a combinatoric fit-quality requirement, not missingness.
    const bool ht=valid_state(i,q,tx,ty,tz,px,py,pz);o.has_track[i]=ht;
    if(ht){o.ip[i]=finite_or(at_or(ip,i,kMissing));o.ip_raw[i]=finite_or(at_or(ipraw,i,kMissing));
      const float ip2=at_or(ipchi2,i,kMissing);o.log1p_ipchi2[i]=std::isfinite(ip2)&&ip2>=0?std::log1p(ip2):kMissing;
      o.track_chi2[i]=finite_or(at_or(chi2,i,kMissing));o.qoverp[i]=finite_or(at_or(qoverp,i,kMissing));
      o.state_dx[i]=tx[i]-pv_x;o.state_dy[i]=ty[i]-pv_y;o.state_dz[i]=tz[i]-pv_z;
      o.dir_x[i]=p>0?px[i]/p:kMissing;o.dir_y[i]=p>0?py[i]/p:kMissing;o.dir_z[i]=p>0?pz[i]/p:kMissing;
      if(std::abs(tz[i])<=max_abs_state_z&&pt[i]>=min_vertex_track_pt&&std::isfinite(ip2)&&ip2>=min_vertex_track_ipchi2)tracks.push_back(i);}
    const bool hp=i<static_cast<int>(nne.size())&&i<static_cast<int>(nnk.size())&&i<static_cast<int>(nnp.size())&&i<static_cast<int>(nnpi.size())
      &&nne[i]>=0&&nnk[i]>=0&&nnp[i]>=0&&nnpi[i]>=0;
    o.has_pid[i]=hp;o.has_muon_pid[i]=i<static_cast<int>(nnmu.size())&&nnmu[i]>=0;
    if(hp){o.target_nne[i]=nne[i];o.target_nnk[i]=nnk[i];o.target_nnp[i]=nnp[i];o.target_nnpi[i]=nnpi[i];
      o.target_valid_e[i]=o.target_valid_k[i]=o.target_valid_p[i]=o.target_valid_pi[i]=1;}
    if(o.has_muon_pid[i]){o.target_nnmu[i]=nnmu[i];o.target_valid_mu[i]=1;}
    const int rid=i<static_cast<int>(id.size())?static_cast<int>(std::lrint(id[i])):0;o.target_reco_id[i]=rid;
    const int aid=std::abs(rid);
    o.target_species[i]=aid==11?1:aid==13?2:aid==22?3:aid==211?4:aid==321?5:
      aid==2212?6:aid==111?7:(aid==130||aid==310)?8:aid==3122?9:0;
    const bool hc=i<static_cast<int>(cecal.size())&&cecal[i]>-900;o.has_calo[i]=hc;
    if(hc){o.calo_ecal[i]=cecal[i];o.calo_hcal2ecal[i]=at_or(ch2e,i,kMissing);o.calo_e49[i]=at_or(ce49,i,kMissing);o.calo_prs[i]=at_or(cprs,i,kMissing);}
  }
  o.n_track_lines=tracks.size();
  std::vector<VertexCandidate> pairs,triplets,quads;
  for(std::size_t a=0;a<tracks.size();++a)for(std::size_t b=a+1;b<tracks.size();++b){
    std::array<int,4> ids{{tracks[a],tracks[b],-1,-1}};auto c=make_vertex(ids,2,px,py,pz,q,ipchi2,tx,ty,tz,pv);
    if(c.fit.valid&&c.fit.max_doca<=max_vertex_doca&&c.flight>=min_vertex_flight)pairs.push_back(c);}
  o.n_pair_total=pairs.size();std::sort(pairs.begin(),pairs.end(),better_vertex);
  const int pair_seed=std::min<int>(pairs.size(),std::max(max_pair_vertices,2*max_triplet_vertices));
  std::set<std::array<int,4>> seen3;
  for(int s=0;s<pair_seed;++s)for(int k:tracks){if(k==pairs[s].tracks[0]||k==pairs[s].tracks[1])continue;
    std::array<int,4> ids{{pairs[s].tracks[0],pairs[s].tracks[1],k,-1}};std::sort(ids.begin(),ids.begin()+3);
    if(!seen3.insert(ids).second)continue;int charge_sum=static_cast<int>(q[ids[0]]+q[ids[1]]+q[ids[2]]);if(std::abs(charge_sum)>1)continue;
    auto c=make_vertex(ids,3,px,py,pz,q,ipchi2,tx,ty,tz,pv);if(c.fit.valid&&c.fit.max_doca<=max_vertex_doca&&c.flight>=min_vertex_flight)triplets.push_back(c);}
  o.n_triplet_total=triplets.size();std::sort(triplets.begin(),triplets.end(),better_vertex);
  const int triple_seed=std::min<int>(triplets.size(),std::max(max_triplet_vertices,2*max_quad_vertices));std::set<std::array<int,4>> seen4;
  for(int s=0;s<triple_seed;++s)for(int k:tracks){auto ids=triplets[s].tracks;if(k==ids[0]||k==ids[1]||k==ids[2])continue;ids[3]=k;std::sort(ids.begin(),ids.end());
    if(!seen4.insert(ids).second)continue;int charge_sum=0;for(int j=0;j<4;++j)charge_sum+=static_cast<int>(q[ids[j]]);if(charge_sum!=0)continue;
    auto c=make_vertex(ids,4,px,py,pz,q,ipchi2,tx,ty,tz,pv);if(c.fit.valid&&c.fit.max_doca<=max_vertex_doca&&c.flight>=min_vertex_flight)quads.push_back(c);}
  o.n_quad_total=quads.size();std::sort(quads.begin(),quads.end(),better_vertex);
  if(static_cast<int>(pairs.size())>max_pair_vertices||static_cast<int>(triplets.size())>max_triplet_vertices||static_cast<int>(quads.size())>max_quad_vertices)o.vertices_truncated=1;
  pairs.resize(std::min<int>(pairs.size(),max_pair_vertices));triplets.resize(std::min<int>(triplets.size(),max_triplet_vertices));quads.resize(std::min<int>(quads.size(),max_quad_vertices));
  std::vector<VertexCandidate> vertices;vertices.insert(vertices.end(),pairs.begin(),pairs.end());vertices.insert(vertices.end(),triplets.begin(),triplets.end());vertices.insert(vertices.end(),quads.begin(),quads.end());
  const int nv=max_pair_vertices+max_triplet_vertices+max_quad_vertices;
  auto pv_i=[&](RVecI&v,int x=0){v=RVecI(nv,x);};auto pv_f=[&](RVecF&v,float x=kMissing){v=RVecF(nv,x);};
  for(auto*v:{&o.vertex_valid,&o.vertex_n_tracks,&o.vertex_track0,&o.vertex_track1,&o.vertex_track2,&o.vertex_track3,&o.vertex_charge})pv_i(*v);
  std::fill(o.vertex_track0.begin(),o.vertex_track0.end(),-1);std::fill(o.vertex_track1.begin(),o.vertex_track1.end(),-1);std::fill(o.vertex_track2.begin(),o.vertex_track2.end(),-1);std::fill(o.vertex_track3.begin(),o.vertex_track3.end(),-1);
  for(auto*v:{&o.vertex_x,&o.vertex_y,&o.vertex_z,&o.vertex_rms,&o.vertex_max_doca,&o.vertex_flight_pv,&o.vertex_ip_pv,&o.vertex_pointing,&o.vertex_px,&o.vertex_py,&o.vertex_pz,&o.vertex_pt,&o.vertex_mass_pi,&o.vertex_corrected_mass_pi,&o.vertex_min_ipchi2,&o.vertex_sum_ipchi2,&o.vertex_fit_proxy})pv_f(*v);
  for(std::size_t j=0;j<vertices.size();++j){const auto&c=vertices[j];o.vertex_valid[j]=1;o.vertex_n_tracks[j]=c.n;o.vertex_track0[j]=c.tracks[0];o.vertex_track1[j]=c.tracks[1];o.vertex_track2[j]=c.tracks[2];o.vertex_track3[j]=c.tracks[3];o.vertex_charge[j]=c.charge;
    o.vertex_x[j]=c.fit.position.x;o.vertex_y[j]=c.fit.position.y;o.vertex_z[j]=c.fit.position.z;o.vertex_rms[j]=c.fit.rms;o.vertex_max_doca[j]=c.fit.max_doca;o.vertex_flight_pv[j]=c.flight;o.vertex_ip_pv[j]=c.ip;o.vertex_pointing[j]=c.point;
    o.vertex_px[j]=c.momentum.x;o.vertex_py[j]=c.momentum.y;o.vertex_pz[j]=c.momentum.z;o.vertex_pt[j]=std::hypot(c.momentum.x,c.momentum.y);o.vertex_mass_pi[j]=c.mass;o.vertex_corrected_mass_pi[j]=c.corrected_mass;o.vertex_min_ipchi2[j]=c.min_ipchi2;o.vertex_sum_ipchi2[j]=c.sum_ipchi2;o.vertex_fit_proxy[j]=c.proxy;}
  std::vector<ChainCandidate> chains;const int n_child=pairs.size()+triplets.size();
  for(int ci=0;ci<n_child;++ci){const auto&child=vertices[ci];for(int k:tracks){bool used=false;for(int j=0;j<child.n;++j)used|=child.tracks[j]==k;if(used)continue;
    std::vector<Vec3> rr{{child.fit.position.x,child.fit.position.y,child.fit.position.z},{tx[k],ty[k],tz[k]}};
    std::vector<Vec3> pp{{child.momentum.x,child.momentum.y,child.momentum.z},{px[k],py[k],pz[k]}};Fit pf=fit_lines(rr,pp);if(!pf.valid||pf.max_doca>max_vertex_doca)continue;
    ChainCandidate c;c.child=child;c.child_index=ci;c.bachelor=k;c.parent=pf;c.parent_p=child.momentum+pp[1];
    c.parent_flight=signed_flight(pv,pf.position,c.parent_p);c.parent_ip=line_ip(pv,pf.position,c.parent_p);c.parent_point=pointing(pv,pf.position,c.parent_p);
    c.child_flight=signed_flight(pf.position,child.fit.position,child.momentum);c.child_point=pointing(pf.position,child.fit.position,child.momentum);c.child_ip=line_ip(pv,child.fit.position,child.momentum);
    if(c.parent_flight<min_vertex_flight||c.child_flight<0||c.parent_point<min_chain_pointing||c.child_point<min_chain_pointing)continue;
    std::vector<Vec3> pm;for(int j=0;j<child.n;++j)pm.push_back({px[child.tracks[j]],py[child.tracks[j]],pz[child.tracks[j]]});pm.push_back(pp[1]);c.parent_mass=mass_pions(pm);
    c.child_corrected_mass=corrected_mass(child.mass,child.momentum,child.fit.position-pf.position);
    c.parent_corrected_mass=corrected_mass(c.parent_mass,c.parent_p,pf.position-pv);
    c.proxy=child.proxy+std::log1p(std::max(0.f,at_or(ipchi2,k,0.f)))-2.*std::log1p(pf.max_doca/0.05)+0.5*std::log1p(c.child_flight);chains.push_back(c);}}
  o.n_chain_total=chains.size();std::sort(chains.begin(),chains.end(),better_chain);if(static_cast<int>(chains.size())>max_chains)o.chains_truncated=1;chains.resize(std::min<int>(chains.size(),max_chains));
  auto pc_i=[&](RVecI&v,int x=0){v=RVecI(max_chains,x);};auto pc_f=[&](RVecF&v,float x=kMissing){v=RVecF(max_chains,x);};
  for(auto*v:{&o.chain_valid,&o.chain_child_n_tracks,&o.chain_track0,&o.chain_track1,&o.chain_track2,&o.chain_track3,&o.chain_bachelor,&o.chain_child_vertex_index,&o.chain_charge})pc_i(*v);
  for(auto*v:{&o.chain_track0,&o.chain_track1,&o.chain_track2,&o.chain_track3,&o.chain_bachelor,&o.chain_child_vertex_index})std::fill(v->begin(),v->end(),-1);
  for(auto*v:{&o.chain_parent_x,&o.chain_parent_y,&o.chain_parent_z,&o.chain_parent_rms,&o.chain_parent_max_doca,&o.chain_parent_flight_pv,&o.chain_parent_ip_pv,&o.chain_parent_pointing,&o.chain_child_flight,&o.chain_child_pointing,&o.chain_child_ip_pv,&o.chain_child_mass_pi,&o.chain_child_corrected_mass_pi,&o.chain_parent_mass_pi,&o.chain_parent_corrected_mass_pi,&o.chain_parent_pt,&o.chain_fit_proxy})pc_f(*v);
  for(std::size_t j=0;j<chains.size();++j){const auto&c=chains[j];o.chain_valid[j]=1;o.chain_child_n_tracks[j]=c.child.n;o.chain_track0[j]=c.child.tracks[0];o.chain_track1[j]=c.child.tracks[1];o.chain_track2[j]=c.child.tracks[2];o.chain_track3[j]=c.child.tracks[3];o.chain_bachelor[j]=c.bachelor;o.chain_child_vertex_index[j]=c.child_index;o.chain_charge[j]=c.child.charge+static_cast<int>(q[c.bachelor]);
    o.chain_parent_x[j]=c.parent.position.x;o.chain_parent_y[j]=c.parent.position.y;o.chain_parent_z[j]=c.parent.position.z;o.chain_parent_rms[j]=c.parent.rms;o.chain_parent_max_doca[j]=c.parent.max_doca;o.chain_parent_flight_pv[j]=c.parent_flight;o.chain_parent_ip_pv[j]=c.parent_ip;o.chain_parent_pointing[j]=c.parent_point;o.chain_child_flight[j]=c.child_flight;o.chain_child_pointing[j]=c.child_point;o.chain_child_ip_pv[j]=c.child_ip;o.chain_child_mass_pi[j]=c.child.mass;o.chain_child_corrected_mass_pi[j]=c.child_corrected_mass;o.chain_parent_mass_pi[j]=c.parent_mass;o.chain_parent_corrected_mass_pi[j]=c.parent_corrected_mass;o.chain_parent_pt[j]=std::hypot(c.parent_p.x,c.parent_p.y);o.chain_fit_proxy[j]=c.proxy;}
  return o;
}

} // namespace lhcb_ml
#endif
"""


def declare_ml_helpers(ROOT: object) -> None:
    if not ROOT.gInterpreter.Declare(CPP_ML_HELPERS):
        raise RuntimeError("ROOT failed to declare ML post-processing helpers")
