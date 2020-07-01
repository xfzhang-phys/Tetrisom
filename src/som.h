#ifndef _SOM_H_
#define _SOM_H_

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <complex>
#include <mpi.h>


// The spectrum function A(om) here is defined as: A(om) = \sum_t \eta_t(om),
// where \eta_t(om) = h_t,  when (c_t - w_t/2) <= om <= (c_t + w_t/2);
// else  \eta_t(om) = 0.
class conf_t {
public:
    double h;
    double w;
    double c;

    conf_t(double _h, double _w, double _c) : h(_h), w(_w), c(_c) {
    }
};

// update type
enum update_t {up_shift, up_change_width, up_change_weight, up_add, up_remove, up_split, up_glue};

// correlation function type
enum corr_t {cf_auto, cf_fermion, cf_boson, cf_imtime};

class SOM {
public:
    uint32_t Lmax;   // total number of attempts
    uint32_t Ngrid;  // size of im-time or im-freq grid
    uint32_t Nf;     // number of global updates for an attempt
    uint32_t Tmax;   // number of elementary updates in a global update
    uint32_t Kmax;   // maximum size of configuration set
    uint32_t nwout;  // number of frequency points used to output the spectrum function
    double Smin;    // minimum area (weight) of a rectangle
    double ommax, ommin;    // frequency range want to know
    double wmin;  // minimum weight of rectangle
    double gamma;   // control the sampling displacement
    double dmax;    // control the acceptance
    double alpha_good;  // control the quality of fitted spectrum function
    double beta;    // inversion temperature
    double norm_spectrum;  // norm of spectral function
    bool monitor_fit_quality;   // monitor fit quality to choose Nf
    std::string corr_type;   // correlation data type: imtime/fermion/boson/auto

    std::vector<double> grid;    // imaginary-time or matsubara-frequencies grid
    std::vector<std::complex<double>> Gf; // averaged Green's function on im-time or im-freq grid
    std::vector<std::complex<double>> Sigma;   // standard deviation (or another error scale) of Green's function
    std::vector<double> dev; // deviation set to fit A(omega)
    std::vector<std::vector<conf_t>> conf;    // configuration set to fit A(omega)

    SOM(uint32_t _Lmax = 1000, uint32_t _Ngrid = 65, uint32_t _Nf = 100, uint32_t _Tmax = 50, uint32_t _Kmax = 60, uint32_t _nwout = 100,
        double _Smin = 1.e-3, double _wmin = 1.e-3, double _gamma = 2., double _dmax = 2., double _ommax = 10,
        double _ommin = 0, double _alpha_good = 1.5, double _beta = 1., double _norm_spectrum = 1.0,
        bool _monitor_fit_quality = false, std::string _corr_type = "imtime");
    ~SOM();

    void attempt(const uint32_t& _counter, std::mt19937_64* rng);
    void output(const uint32_t& mpi_rank, const uint32_t& mpi_size);

private:
    uint32_t kappa_count;    // for monitor fit quality
    double attmpt_dev;  // deviation for attempt config
    double tmp_dev; // deviation for tmporary config
    double new_dev;
    double dacc;    // control the accepted probability by P = (Dev/Dev_new)^(1+dacc), if Dev_new > Dev
    corr_t idx_corr_type;    // index of correlation function type for switch. auto: 0; fermion: 1; boson: 2; imtime: 3
    std::vector<conf_t> attmpt_conf; // configuration for an attempt step
    std::vector<conf_t> tmp_conf;    // termporary configuration for a global update
    std::vector<conf_t> new_conf;
    std::vector<std::complex<double>> attmpt_elem_dev;    // dev element for each iwn contributed by each rectangle
    std::vector<std::complex<double>> elem_dev;    // dev element for each iwn contributed by each rectangle
    std::vector<std::complex<double>> new_elem_dev;
    std::vector<uint64_t> trial_steps;
    std::vector<uint64_t> accepted_steps;

    void get_input();
    void get_gf();
    void init_config(std::mt19937_64* rng);
    void global_update(std::mt19937_64* rng);
    void update_shift_rectangle(std::mt19937_64* rng);
    void update_change_width(std::mt19937_64* rng);
    void update_change_weight(std::mt19937_64* rng);
    void update_add_rectangle(std::mt19937_64* rng);
    void update_remove_rectangle(std::mt19937_64* rng);
    void update_split_rectangle(std::mt19937_64* rng);
    void update_glue_rectangle(std::mt19937_64* rng);
    void calc_kappa(const std::vector<conf_t>& _conf);
    void calc_dev_rec(const conf_t& _conf, const uint32_t& _ik, std::vector<std::complex<double>>& _elem_dev);
    double calc_dev(const std::vector<std::complex<double>>& _elem_dev, const uint32_t& _nk);
    double Pdx(const double& xmin, const double& xmax, std::mt19937_64* rng);
    double get_norm();
};

#endif
