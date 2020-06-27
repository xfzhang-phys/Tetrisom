#include "som.h"


SOM::SOM(uint32_t _Lmax, uint32_t _Ngrid, uint32_t _Nf, uint32_t _Tmax, uint32_t _Kmax, uint32_t _nwout,
    double _Smin, double _wmin, double _gamma, double _dmax, double _ommax,
    double _ommin, double _alpha_good, double _beta, double _norm_spectrum,
    bool _monitor_fit_quality, string _corr_type) :
    Lmax(_Lmax), Ngrid(_Ngrid), Nf(_Nf), Tmax(_Tmax), Kmax(_Kmax), nwout(_nwout), Smin(_Smin),
    ommax(_ommax), ommin(_ommin), wmin(_wmin), gamma(_gamma), dmax(_dmax), alpha_good(_alpha_good),
    beta(_beta), norm_spectrum(_norm_spectrum), monitor_fit_quality(_monitor_fit_quality),
    corr_type(_corr_type) {

    // get input parameters from params.in file
    get_input();

    grid.resize(Ngrid, 0.0);
    Gf.resize(Ngrid, complex<double>(0.0, 0.0));
    Sigma.resize(Ngrid, complex<double>(0.0, 0.0));

    dev.resize(Lmax);
    accepted_steps.resize(7, 0); trial_steps.resize(7, 0);

    conf.resize(Lmax);
    for (auto& vc : conf) vc.reserve(Kmax);

    attmpt_conf.reserve(Kmax); tmp_conf.reserve(Kmax);
    attmpt_elem_dev.reserve(Ngrid * Kmax); elem_dev.reserve(Ngrid * Kmax);
    new_conf.reserve(Kmax); new_elem_dev.reserve(Ngrid * Kmax);

    // obtain imaginary-time correlation function Gtau
    get_gf();
    if (norm_spectrum < 0.0) norm_spectrum = get_norm();
    for (uint32_t ig = 0; ig < Ngrid; ig++) {
        Gf[ig] /= norm_spectrum;
        Sigma[ig] /= norm_spectrum;
    }

    // corr type index
    if (corr_type == "auto") idx_corr_type = cf_auto;
    else if (corr_type == "fermion") idx_corr_type = cf_fermion;
    else if (corr_type == "boson") idx_corr_type = cf_boson;
    else idx_corr_type = cf_imtime;

    // initialize kappa_count for monitoring
    if (monitor_fit_quality) kappa_count = 0;
}

SOM::~SOM() {
}

void SOM::attempt(const uint32_t& _counter, mt19937_64* rng) {
    init_config(rng);
    for (uint32_t _f = 0; _f < Nf; _f++) {
        global_update(rng);
    }
    conf[_counter] = attmpt_conf;
    dev[_counter] = attmpt_dev;
    if (monitor_fit_quality)
        calc_kappa(attmpt_conf);
}

void SOM::output(const uint32_t& mpi_rank, const uint32_t& mpi_size) {  
    // transform vector<<vector<conf_t>> to vector<double> for MPI communication
    vector<double> _tmp_conf;
    for (auto& vc : conf) {
        vc.resize(Kmax, conf_t(0.0, 0.0, 0.0));
        for (const auto& v : vc) {
            _tmp_conf.push_back(v.h);
            _tmp_conf.push_back(v.w);
            _tmp_conf.push_back(v.c);

        }
    }
    // Gather data by MPI
    vector<double> total_dev(mpi_size * Lmax, 0.0);
    vector<double> total_conf(mpi_size * Lmax * Kmax * 3, 0.0);

    MPI_Gather(dev.data(), Lmax, MPI_DOUBLE, total_dev.data(), Lmax, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(_tmp_conf.data(), Lmax * Kmax * 3, MPI_DOUBLE, total_conf.data(), Lmax * Kmax * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        FILE* fp;
        uint32_t Lgood = 0;
        double dev_min = *(std::min_element(total_dev.begin(), total_dev.end()));

        vector<double> Aom(nwout, 0.0);
        for (uint32_t il = 0; il < mpi_size * Lmax; il++) {
            if (alpha_good * dev_min - total_dev[il] > 0) {
                Lgood++;
                for (uint32_t iw = 0; iw < nwout; iw++) {
                    double _omega = ommin + iw * (ommax - ommin) / nwout;
                    for (uint32_t ik = 0; ik < Kmax; ik++) {
                        uint32_t _idx = il * Kmax * 3 + ik * 3;
                        if (_omega >= (total_conf[_idx + 2] - total_conf[_idx + 1] / 2)
                            && _omega <= (total_conf[_idx + 2] + total_conf[_idx + 1] / 2)) {
                            Aom[iw] += total_conf[_idx];
                        }
                    }
                }
            }
        }
        // output the spectral function: A(w)
        fp = fopen("Aw.dat", "w");
        fprintf(fp, "#               w                A(w)\n");
        for (uint32_t iw = 0; iw < nwout; iw++) {
            double _omega = ommin + iw * (ommax - ommin) / nwout;
            fprintf(fp, "%22.12lf    %22.12lf\n", _omega, Aom[iw] * norm_spectrum / Lgood);
        }
        fclose(fp);

        std::ofstream _outfile;
        _outfile.open("Som.out", std::ios::out | std::ios::trunc);
        // Acceptance ratio
        _outfile << "Acceptance Ratio:" << std::endl;
        _outfile << "\tShift rectangle                = " << 1.0 * accepted_steps[up_shift] / trial_steps[up_shift] << std::endl;
        _outfile << "\tChange rectangle width         = " << 1.0 * accepted_steps[up_change_width] / trial_steps[up_change_width] << std::endl;
        _outfile << "\tChange two rectangles' weights = " << 1.0 * accepted_steps[up_change_weight] / trial_steps[up_change_weight] << std::endl;
        _outfile << "\tAdd rectangle                  = " << 1.0 * accepted_steps[up_add] / trial_steps[up_add] << std::endl;
        _outfile << "\tRemove rectangle               = " << 1.0 * accepted_steps[up_remove] / trial_steps[up_remove] << std::endl;
        _outfile << "\tSplit rectangle                = " << 1.0 * accepted_steps[up_split] / trial_steps[up_split] << std::endl;
        _outfile << "\tGlue two rectangles            = " << 1.0 * accepted_steps[up_glue] / trial_steps[up_glue] << std::endl;
        _outfile << std::endl;

        // simplly output the quality of attempts
        if (monitor_fit_quality) {
            _outfile << "kappa_good / L_total = " << 1.0 * kappa_count / (mpi_size * Lmax) << std::endl;
        }
        _outfile << "D_min = " << dev_min << std::endl;
        _outfile << "L_good / L_total = " << 1.0 * Lgood / (mpi_size * Lmax) << std::endl;
        _outfile.close();
    }
}

void SOM::get_gf() {
    uint32_t icount;
    string _grid, _gre, _gim, _line;

    // read Gf from Gf.dat file
    std::ifstream _infile1("Gf.dat", std::ios::in);
    icount = 0;
    while (getline(_infile1, _line)) {
        std::istringstream iss(_line);
        iss >> _grid >> _gre >> _gim;
        grid[icount] = std::stod(_grid);
        Gf[icount] = complex<double>(std::stod(_gre), std::stod(_gim));
        icount++;
    }
    _infile1.close();

    // read Sigma from Sigma.dat file
    std::ifstream _infile2("Sigma.dat", std::ios::in);
    icount = 0;
    while (getline(_infile2, _line)) {
        std::istringstream iss(_line);
        iss >> _grid >> _gre >> _gim;
        Sigma[icount] = complex<double>(std::stod(_gre), std::stod(_gim));
        icount++;
    }
    _infile2.close();
}

void SOM::init_config(mt19937_64* rng) {
    double w, h, c;
    uniform_real_distribution<double> uniform_dist(0, 1);
    uniform_int_distribution<uint32_t> uniform_dist_int(2, Kmax - 1);

    // choose Know for initializing configuration
    uint32_t _Know = uniform_dist_int(*rng);
    // generate weights with their summation equal to 1
    vector<double> weight(_Know, 0.0);
    for (uint32_t i = 0; i < _Know - 1; i++) {
        weight[i] = uniform_dist(*rng);
    }
    weight[_Know - 1] = 1.0;
    sort(weight.begin(), weight.end());
    adjacent_difference(weight.begin(), weight.end(), weight.begin());
    sort(weight.begin(), weight.end());
    uint32_t plus_count = 0;
    uint32_t minus_count = _Know - 1;
    while (weight[plus_count] < Smin) {
        while(weight[minus_count] < 2 * Smin) {
            minus_count--;
        }
        weight[plus_count] += Smin;
        weight[minus_count] -= Smin;
        plus_count++;
    }

    // initialize configuration
    attmpt_conf.clear();
    attmpt_elem_dev.assign(Ngrid * Kmax, 0.0);
    for (uint32_t ik = 0; ik < _Know; ik++) {
        c = ommin + wmin / 2 + (ommax - ommin - wmin) * uniform_dist(*rng);
        w = wmin + (fmin(2 * (c - ommin), 2 * (ommax - c)) - wmin) * uniform_dist(*rng);
        h = weight[ik] / w;
        // attmpt_conf.insert(attmpt_conf.end(), conf_t(h, w, c));
        attmpt_conf.emplace_back(h, w, c);
        calc_dev_rec(conf_t(h, w, c), ik, attmpt_elem_dev);
    }

    // get dev
    attmpt_dev = calc_dev(attmpt_elem_dev, _Know);
}

void SOM::global_update(mt19937_64* rng) {
    uniform_int_distribution<uint32_t> uniform_dist_int(0, Tmax - 1);
    uniform_int_distribution<uint32_t> uniform_dist_update(0, 6);
    uniform_real_distribution<double> uniform_dist(0, 1);

    // choose T1, d1 and d2 for a global update
    uint32_t T1 = uniform_dist_int(*rng);
    double d1 = uniform_dist(*rng);
    double d2 = 1 + (dmax - 1) * uniform_dist(*rng);
    // copy the attempt config to a temporary config
    tmp_conf = attmpt_conf;
    tmp_dev = attmpt_dev;
    elem_dev = attmpt_elem_dev;
    // elementary updates
    for (uint32_t iter = 0; iter < T1; iter++) {
        // let dacc = d1
        dacc = d1;
        switch(uniform_dist_update(*rng)) {
            case up_shift: update_shift_rectangle(rng); break;
            case up_change_width: update_change_width(rng); break;
            case up_change_weight: if (tmp_conf.size() > 1) update_change_weight(rng); break;
            case up_add: if (tmp_conf.size() < (Kmax - 1)) update_add_rectangle(rng); break;
            case up_remove: if (tmp_conf.size() > 1) update_remove_rectangle(rng); break;
            case up_split: if (tmp_conf.size() < (Kmax - 1)) update_split_rectangle(rng); break;
            case up_glue: if (tmp_conf.size() > 1) update_glue_rectangle(rng); break;
        }
    }
    for (uint32_t iter = T1; iter < Tmax; iter++) {
        // let dacc = d2
        dacc = d2;
        switch(uniform_dist_update(*rng)) {
            case up_shift: update_shift_rectangle(rng); break;
            case up_change_width: update_change_width(rng); break;
            case up_change_weight: if (tmp_conf.size() > 1) update_change_weight(rng); break;
            case up_add: if (tmp_conf.size() < (Kmax - 1)) update_add_rectangle(rng); break;
            case up_remove: if (tmp_conf.size() > 1) update_remove_rectangle(rng); break;
            case up_split: if (tmp_conf.size() < (Kmax - 1)) update_split_rectangle(rng); break;
            case up_glue: if (tmp_conf.size() > 1) update_glue_rectangle(rng); break;
        }
    }
    if (tmp_dev < attmpt_dev) {
        attmpt_conf = tmp_conf;
        attmpt_dev = tmp_dev;
        attmpt_elem_dev = elem_dev;
    }
}

void SOM::update_shift_rectangle(mt19937_64* rng) {
    uniform_int_distribution<uint32_t> uniform_dist_int(0, tmp_conf.size() - 1);
    uniform_real_distribution<double> uniform_dist(0, 1);

    // pick up a rectangle randomly from [0, tmp_Know-1]
    uint32_t t = uniform_dist_int(*rng);
    // shift ct in range [ommin+wt/2, ommax-wt/2]
    double dx_min = ommin + tmp_conf[t].w / 2.0 - tmp_conf[t].c;
    double dx_max = ommax - tmp_conf[t].w / 2.0 - tmp_conf[t].c;
    if (dx_max <= dx_min) { return ; }
    double dc = Pdx(dx_min, dx_max, rng);
    // trial step
    uint32_t _conf_size = tmp_conf.size();
    new_conf = tmp_conf;
    new_elem_dev = elem_dev;
    new_conf[t].c += dc;
    calc_dev_rec(new_conf[t], t, new_elem_dev);
    double new_dev = calc_dev(new_elem_dev, _conf_size);  // new dev with xi -> xi + dx
    if (uniform_dist(*rng) < pow((tmp_dev / new_dev), 1 + dacc)) {
        tmp_conf = new_conf;
        tmp_dev = new_dev;
        elem_dev = new_elem_dev;
        accepted_steps[up_shift]++;
    }
    trial_steps[up_shift]++;
}

void SOM::update_change_width(mt19937_64* rng) {
    uniform_int_distribution<uint32_t> uniform_dist_int(0, tmp_conf.size() - 1);
    uniform_real_distribution<double> uniform_dist(0, 1);

    // pick up a rectangle randomly from [0, tmp_Know-1]
    uint32_t t = uniform_dist_int(*rng);
    // keep ht*wt = const and change wt in range [wmin, min{2(ct-ommin), 2(ommax-ct)}]
    double weight = tmp_conf[t].h * tmp_conf[t].w;
    double dx_min = wmin - tmp_conf[t].w;
    double dx_max = fmin(2 * (tmp_conf[t].c - ommin), 2 * (ommax - tmp_conf[t].c)) - tmp_conf[t].w;
    if (dx_max <= dx_min) { return ; }
    double dw = Pdx(dx_min, dx_max, rng);
    // trial step
    uint32_t _conf_size = tmp_conf.size();
    new_conf = tmp_conf;
    new_elem_dev = elem_dev;
    new_conf[t].w += dw; new_conf[t].h = weight / new_conf[t].w;
    calc_dev_rec(new_conf[t], t, new_elem_dev);
    double new_dev = calc_dev(new_elem_dev, _conf_size);  // new dev with xi -> xi + dx
    if (uniform_dist(*rng) < pow((tmp_dev / new_dev), 1 + dacc)) {
        tmp_conf = new_conf;
        tmp_dev = new_dev;
        elem_dev = new_elem_dev;
        accepted_steps[up_change_width]++;
    }
    trial_steps[up_change_width]++;
}

void SOM::update_change_weight(mt19937_64* rng) {
    uniform_int_distribution<uint32_t> uniform_dist_int(0, tmp_conf.size() - 1);
    uniform_real_distribution<double> uniform_dist(0, 1);

    // pick up two rectangles randomly from [0, tmp_Know-1] to change their weight
    uint32_t t1 = uniform_dist_int(*rng);
    uint32_t t2 = uniform_dist_int(*rng);
    if (t1 == t2) t2 = (t1 + 1) % tmp_conf.size();
    // keep norm-conserving
    double w1 = tmp_conf[t1].w;
    double w2 = tmp_conf[t2].w;
    double h1 = tmp_conf[t1].h;
    double h2 = tmp_conf[t2].h;
    double dx_min = Smin / w1 - h1;
    double dx_max = (h2 - Smin / w2) * w2 / w1;
    if (dx_max <= dx_min) { return ; }
    double dh = Pdx(dx_min, dx_max, rng);
    // trial step
    uint32_t _conf_size = tmp_conf.size();
    new_conf = tmp_conf;
    new_elem_dev = elem_dev;
    new_conf[t1].h += dh; new_conf[t2].h -= dh * w1 / w2;
    calc_dev_rec(new_conf[t1], t1, new_elem_dev);
    calc_dev_rec(new_conf[t2], t2, new_elem_dev);
    double new_dev = calc_dev(new_elem_dev, _conf_size);  // new dev with xi -> xi + dx
    if (uniform_dist(*rng) < pow((tmp_dev / new_dev), 1 + dacc)) {
        tmp_conf = new_conf;
        tmp_dev = new_dev;
        elem_dev = new_elem_dev;
        accepted_steps[up_change_weight]++;
    }
    trial_steps[up_change_weight]++;
}

void SOM::update_add_rectangle(mt19937_64* rng) {
    uniform_int_distribution<uint32_t> uniform_dist_int(0, tmp_conf.size() - 1);
    uniform_real_distribution<double> uniform_dist(0, 1);

    // pick up a rectangle randomly from [0, tmp_Know-1] to share its weight with new rectangle
    uint32_t t = uniform_dist_int(*rng);
    if (tmp_conf[t].h * tmp_conf[t].w <= 2 * Smin) { return ; }
    double dx_min = Smin;
    double dx_max = tmp_conf[t].h * tmp_conf[t].w - Smin;
    if (dx_max <= dx_min) { return ; }
    double h, w;
    double c = (ommin + wmin / 2) + (ommax - ommin - wmin) * uniform_dist(*rng);
    double w_new_max = 2.0 * fmin(ommax - c, c - ommin);
    double dx = Pdx(dx_min, dx_max, rng);
    double r = uniform_dist(*rng);
    // trial step
    uint32_t _conf_size = tmp_conf.size();
    new_conf = tmp_conf;
    new_elem_dev = elem_dev;
    h = dx / w_new_max + (dx / wmin - dx / w_new_max) * r; w = dx / h;
    // new_conf.insert(new_conf.end(), conf_t(h, w, c));
    new_conf.emplace_back(h, w, c);
    new_conf[t].h -= dx / new_conf[t].w;
    calc_dev_rec(new_conf[t], t, new_elem_dev);
    calc_dev_rec(new_conf[_conf_size], _conf_size, new_elem_dev);
    double new_dev = calc_dev(new_elem_dev, _conf_size + 1);  // new dev with xi -> xi + dx
    if (uniform_dist(*rng) < pow((tmp_dev / new_dev), 1 + dacc)) {
        tmp_conf = new_conf;
        tmp_dev = new_dev;
        elem_dev = new_elem_dev;
        accepted_steps[up_add]++;
    }
    trial_steps[up_add]++;
}

void SOM::update_remove_rectangle(mt19937_64* rng) {
    uniform_int_distribution<uint32_t> uniform_dist_int(0, tmp_conf.size() - 1);
    uniform_real_distribution<double> uniform_dist(0, 1);

    // pick up two rectangles randomly from [0, tmp_Know-1]
    uint32_t t1 = uniform_dist_int(*rng);
    uint32_t t2 = uniform_dist_int(*rng);
    if (t1 == t2) t2 = (t1 + 1) % tmp_conf.size();
    // trial step
    uint32_t _conf_size = tmp_conf.size();
    double dx = tmp_conf[t1].h * tmp_conf[t1].w;
    new_conf = tmp_conf;
    new_elem_dev = elem_dev;
    new_conf[t2].h += dx / new_conf[t2].w;
    new_conf[t1] = new_conf.back();
    new_conf.pop_back();
    if (t1 < _conf_size - 1) calc_dev_rec(new_conf[t1], t1, new_elem_dev);
    if (t2 < _conf_size - 1) calc_dev_rec(new_conf[t2], t2, new_elem_dev);
    double new_dev = calc_dev(new_elem_dev, _conf_size - 1);  // new dev with xi -> xi + dx
    if (uniform_dist(*rng) < pow((tmp_dev / new_dev), 1 + dacc)) {
        tmp_conf = new_conf;
        tmp_dev = new_dev;
        elem_dev = new_elem_dev;
        accepted_steps[up_remove]++;
    }
    trial_steps[up_remove]++;
}

void SOM::update_split_rectangle(mt19937_64* rng) {
    uniform_int_distribution<uint32_t> uniform_dist_int(0, tmp_conf.size() - 1);
    uniform_real_distribution<double> uniform_dist(0, 1);

    // pick up two rectangles randomly from [0, tmp_Know-1]
    uint32_t t = uniform_dist_int(*rng);
    conf_t old_conf = tmp_conf[t];
    if ((old_conf.w <= 2 * wmin) || (old_conf.w * old_conf.h <= 2 * Smin)) { return ; }
    double h = old_conf.h;
    double w1 = wmin + (old_conf.w - 2 * wmin) * uniform_dist(*rng);
    double w2 = old_conf.w - w1;
    if (w1 > w2) { double tmpw = w1; w1 = w2; w2 = tmpw; }
    double c1 = old_conf.c - old_conf.w / 2 + w1 / 2;
    double c2 = old_conf.c + old_conf.w / 2 - w2 / 2;
    // shift c1 in range [ommin+w1/2, ommax-w1/2]
    double dx_min = ommin + w1 / 2.0 - c1;
    double dx_max = ommax - w1 / 2.0 - c1;
    if (dx_max <= dx_min) { return ; }
    double dc1 = Pdx(dx_min, dx_max, rng);
    // trial step
    uint32_t _conf_size = tmp_conf.size();
    new_conf = tmp_conf;
    new_elem_dev = elem_dev;
    double dc2 = -1.0 * w1 * dc1 / w2;
    if ((c1 + dc1 >= ommin + w1 / 2) && (c1 + dc1 <= ommax - w1 / 2)
        && (c2 + dc2 >= ommin + w2 / 2) && (c2 + dc2 <= ommax - w2 / 2)) {
        new_conf[t] = new_conf.back();
        new_conf.pop_back();
        // new_conf.insert(new_conf.end(), conf_t(h, w1, c1 + dc1));
        // new_conf.insert(new_conf.end(), conf_t(h, w2, c2 + dc2));
        new_conf.emplace_back(h, w1, c1 + dc1);
        new_conf.emplace_back(h, w2, c2 + dc2);
        if (t < _conf_size - 1) calc_dev_rec(new_conf[t], t, new_elem_dev);
        calc_dev_rec(new_conf[_conf_size - 1], _conf_size - 1, new_elem_dev);
        calc_dev_rec(new_conf[_conf_size], _conf_size, new_elem_dev);
        double new_dev = calc_dev(new_elem_dev, _conf_size + 1);  // new dev with xi -> xi + dx
        if (uniform_dist(*rng) < pow((tmp_dev / new_dev), 1 + dacc)) {
            tmp_conf = new_conf;
            tmp_dev = new_dev;
            elem_dev = new_elem_dev;
            accepted_steps[up_split]++;
        }
    }
    trial_steps[up_split]++;
}

void SOM::update_glue_rectangle(mt19937_64* rng) {
    uniform_int_distribution<uint32_t> uniform_dist_int(0, tmp_conf.size() - 1);
    uniform_real_distribution<double> uniform_dist(0, 1);

    // pick up two rectangles randomly from [0, tmp_Know-1]
    uint32_t t1 = uniform_dist_int(*rng);
    uint32_t t2 = uniform_dist_int(*rng);
    if (t1 == t2) t2 = (t1 + 1) % tmp_conf.size();
    conf_t old_conf1 = tmp_conf[t1];
    conf_t old_conf2 = tmp_conf[t2];
    double weight = old_conf1.h * old_conf1.w + old_conf2.h * old_conf2.w;
    double w_new = 0.5 * (old_conf1.w + old_conf2.w);
    double h_new = weight / w_new;
    double c_new = old_conf1.c + (old_conf2.c - old_conf1.c) * old_conf2.h * old_conf2.w / weight;
    double dx_min = ommin + w_new / 2.0 - c_new;
    double dx_max = ommax - w_new / 2.0 - c_new;
    if (dx_max <= dx_min) { return ; }
    double dc = Pdx(dx_min, dx_max, rng);
    // trial step
    uint32_t _conf_size = tmp_conf.size();
    new_conf = tmp_conf;
    new_elem_dev = elem_dev;
    new_conf[t1] = new_conf.back();
    new_conf.pop_back();
    new_conf[t2] = new_conf.back();
    new_conf.pop_back();
    // new_conf.insert(new_conf.end(), conf_t(h_new, w_new, c_new + dc));
    new_conf.emplace_back(h_new, w_new, c_new + dc);
    if (t1 < _conf_size - 2) calc_dev_rec(new_conf[t1], t1, new_elem_dev);
    if (t2 < _conf_size - 2) calc_dev_rec(new_conf[t2], t2, new_elem_dev);
    calc_dev_rec(new_conf[_conf_size - 2], _conf_size - 2, new_elem_dev);
    double new_dev = calc_dev(new_elem_dev, _conf_size - 1);  // new dev with xi -> xi + dx
    if (uniform_dist(*rng) < pow((tmp_dev / new_dev), 1 + dacc)) {
        tmp_conf = new_conf;
        tmp_dev = new_dev;
        elem_dev = new_elem_dev;
        accepted_steps[up_glue]++;
    }
    trial_steps[up_glue]++;
}

void SOM::calc_kappa(const vector<conf_t>& _conf) {
    constexpr double _inv_pi = 1.0 / 3.141592653589793;
    constexpr complex<double> _cmplx_i(0.0, 1.0);
    double kappa = 0.0;
    complex<double> delta_m, delta_m_old;

    if (corr_type == "auto") {  // autocorrelator for hermitians
        complex<double> Gsample = 0.0;
        for (uint32_t ic = 0; ic < _conf.size(); ic++) {
                double _c = _conf[ic].c; double _w = _conf[ic].w;
                Gsample += _conf[ic].h * (_w + grid[0] * (atan((_c - 0.5 * _w) / grid[0]) - atan((_c + 0.5 * _w) / grid[0])));
        }
        Gsample *= 2.0 * _inv_pi;
        delta_m_old = (Gsample - Gf[0]) / Sigma[0];
        for (uint32_t iw = 0; iw < Ngrid; iw++) {
            Gsample = 0.0;
            for (uint32_t ic = 0; ic < _conf.size(); ic++) {
                double _c = _conf[ic].c; double _w = _conf[ic].w;
                Gsample += _conf[ic].h * (_w + grid[iw] * (atan((_c - 0.5 * _w) / grid[iw]) - atan((_c + 0.5 * _w) / grid[iw])));
            }
            Gsample *= 2.0 * _inv_pi;
            delta_m = (Gsample - Gf[iw]) / Sigma[iw];
            kappa += 0.5 * (1.0 - cos(arg(delta_m) - arg(delta_m_old)));
            delta_m_old = delta_m;
        }
        kappa /= Ngrid - 1.0;
    }
    else if (corr_type == "imtime") {   // imaginary-time version for autocorrelator of hermitians
        // for tau = 0
        complex<double> Gsample = 1.0;
        delta_m_old = (Gsample - Gf[0]) / Sigma[0];
        // for tau > 0
        for (uint32_t it = 1; it < Ngrid; it++) {
            double ftau = grid[it];
            Gsample = 0.0;
            for (uint32_t ic = 0; ic < _conf.size(); ic++) {
                Gsample += _conf[ic].h * exp(-1 * ftau * _conf[ic].c) * sinh(0.5 * ftau * _conf[ic].w);
            }
            Gsample *= 2.0 / ftau;
            delta_m = (Gsample - Gf[it]) / Sigma[it];
            kappa += 0.5 * (1.0 - cos(arg(delta_m) - arg(delta_m_old)));
            delta_m_old = delta_m;
        }
        kappa /= Ngrid - 1.0;
    }
    else if (corr_type == "boson") {    // boson-like correlator
        complex<double> Gsample = 0.0;
        for (uint32_t ic = 0; ic < _conf.size(); ic++) {
                double _c = _conf[ic].c; double _w = _conf[ic].w; complex<double> _iom = _cmplx_i * grid[0];
                Gsample += _conf[ic].h * (_w + _iom * log((_iom - _c - 0.5 * _w) / (_iom - _c + 0.5 * _w)));
        }
        Gsample *= _inv_pi;
        delta_m_old = (Gsample - Gf[0]) / Sigma[0];
        for (uint32_t iw = 0; iw < Ngrid; iw++) {
            Gsample = 0.0;
            for (uint32_t ic = 0; ic < _conf.size(); ic++) {
                double _c = _conf[ic].c; double _w = _conf[ic].w; complex<double> _iom = _cmplx_i * grid[iw];
                Gsample += _conf[ic].h * (_w + _iom * log((_iom - _c - 0.5 * _w) / (_iom - _c + 0.5 * _w)));
            }
            Gsample *= _inv_pi;
            delta_m = (Gsample - Gf[iw]) / Sigma[iw];
            kappa += 0.5 * (1.0 - cos(arg(delta_m) - arg(delta_m_old)));
            delta_m_old = delta_m;
        }
        kappa /= Ngrid - 1.0;
    }
    else {  // fermion case
        complex<double> Gsample = 0.0;
        for (uint32_t ic = 0; ic < _conf.size(); ic++) {
                double _c = _conf[ic].c; double _w = _conf[ic].w; complex<double> _iom = _cmplx_i * grid[0];
                Gsample += _conf[ic].h * log((_iom - _c + 0.5 * _w) / (_iom - _c - 0.5 * _w));
        }
        delta_m_old = (Gsample - Gf[0]) / Sigma[0];
        for (uint32_t iw = 0; iw < Ngrid; iw++) {
            Gsample = 0.0;
            for (uint32_t ic = 0; ic < _conf.size(); ic++) {
                double _c = _conf[ic].c; double _w = _conf[ic].w; complex<double> _iom = _cmplx_i * grid[iw];
                Gsample += _conf[ic].h * log((_iom - _c + 0.5 * _w) / (_iom - _c - 0.5 * _w));
            }
            delta_m = (Gsample - Gf[iw]) / Sigma[iw];
            kappa += 0.5 * (1.0 - cos(arg(delta_m) - arg(delta_m_old)));
            delta_m_old = delta_m;
        }
        kappa /= Ngrid - 1.0;
    }
    if (kappa >= 0.25) kappa_count++;
}

void SOM::calc_dev_rec(const conf_t& _conf, const uint32_t& _ik, vector<complex<double>>& _elem_dev) {
    constexpr double _inv_pi = 1.0 / 3.141592653589793;
    constexpr complex<double> _cmplx_i(0.0, 1.0);
    complex<double> Gsample;

    switch(idx_corr_type) {
        case cf_auto : {  // autocorrelator for hermitians
            for (uint32_t ig = 0; ig < Ngrid; ig++) {
                uint32_t _idx = ig * Kmax + _ik;
                Gsample = _conf.h * (_conf.w + grid[ig] * (atan((_conf.c - 0.5 * _conf.w) / grid[ig]) - atan((_conf.c + 0.5 * _conf.w) / grid[ig])));
                Gsample *= 2.0 * _inv_pi;
                _elem_dev[_idx] = Gsample;
            }
            break;
        }
        case cf_fermion : {  // fermion case
            for (uint32_t ig = 0; ig < Ngrid; ig++) {
                uint32_t _idx = ig * Kmax + _ik;
                Gsample = _conf.h * log((_cmplx_i * grid[ig] - _conf.c + 0.5 * _conf.w) / (_cmplx_i * grid[ig] - _conf.c - 0.5 * _conf.w));
                _elem_dev[_idx] = Gsample;
            }
            break;
        }
        case cf_boson : {    // boson-like correlator
            for (uint32_t ig = 0; ig < Ngrid; ig++) {
                uint32_t _idx = ig * Kmax + _ik;
                Gsample = _conf.h * (_conf.w + _cmplx_i * grid[ig] * log((_cmplx_i * grid[ig] - _conf.c - 0.5 * _conf.w) / (_cmplx_i * grid[ig] - _conf.c + 0.5 * _conf.w)));
                Gsample *= _inv_pi;
                _elem_dev[_idx] = Gsample;
            }
            break;
        }
        case cf_imtime : {   // imaginary-time version for autocorrelator of hermitians
            for (uint32_t ig = 0; ig < Ngrid; ig++) {
                uint32_t _idx = ig * Kmax + _ik;
                Gsample = _conf.h * exp(-1 * grid[ig] * _conf.c) * sinh(0.5 * grid[ig] * _conf.w);
                Gsample *= 2.0 / grid[ig];
                _elem_dev[_idx] = Gsample;
            }
            break;
        }
    }
}

double SOM::calc_dev(const vector<complex<double>>& _elem_dev, const uint32_t& _nk) {
    double res = 0.0;
    double _begin;
    complex<double> _delta_m;

    for (uint32_t ig = 0; ig < Ngrid; ig++) {
        _delta_m = 0.0;
        _begin = ig * Kmax;
        for (uint32_t _ik = _begin; _ik < _begin + _nk; _ik++) {
            _delta_m += _elem_dev[_ik];
        }
        res += abs((_delta_m - Gf[ig]) / Sigma[ig]);
    }

    return res;
}

// This function generate dx in range [xmin, xmax] with probability density
// function as P(dx) = N exp(-gamma * |dx| / X), where X = max(|xmin|, |xmax|).
double SOM::Pdx(const double& xmin, const double& xmax, mt19937_64* rng) {
    uniform_real_distribution<double> uniform_dist(0, 1);
    double _X = fmax(fabs(xmin), fabs(xmax));
    double _lambda = gamma / _X;
    double _elx = exp(-1 * _lambda * fabs(xmin));
    double _N = _lambda / ((xmin / fabs(xmin)) * (exp(-1 * _lambda * fabs(xmin)) - 1)
        + (xmax / fabs(xmax)) * (1 - exp(-1 * _lambda * fabs(xmax))));
 
    double y = uniform_dist(*rng);
    double _lysn = _lambda * y / _N;
    if (xmin >= 0) {
        return -1 * log(_elx - _lysn) / _lambda;
    }
    else if (xmax <= 0) {
        return log(_lysn + _elx) / _lambda;
    }
    else {
        double _C1 = _N * (1 - _elx) / _lambda;
        if (y <= _C1) {
            return log(_lysn + _elx) / _lambda;
        }
        else {
            return -1 * log(1 - _lysn + _lambda * _C1 / _N) / _lambda;
        }
    }
}

double SOM::get_norm() {
    constexpr double _pi = 3.141592653589793;
    constexpr double _small = 1.0e-6;
    double _norm;
    if (corr_type == "fermion") {
        _norm = 1.0;
    }
    else if (corr_type == "imtime") {
        for (uint32_t ig = 0; ig < Ngrid; ig++) {
            if (grid[ig] < _small) {
                _norm = abs(Gf[ig]);
                break;
            }
        }
    }
    else if (corr_type == "auto") {
        for (uint32_t ig = 0; ig < Ngrid; ig++) {
            if (grid[ig] < _small) {
                _norm = abs(Gf[ig]);
                break;
            }
        }
        _norm *= 0.5 * _pi;
    }
    else {  // boson case
        for (uint32_t ig = 0; ig < Ngrid; ig++) {
            if (grid[ig] < _small) {
                _norm = abs(Gf[ig]);
                break;
            }
        }
        _norm *= _pi;
    }
    return _norm;
}

void SOM::get_input() {
    constexpr double _boltz = 8.617333e-2;    // Boltzmann constant in meV/K
    std::string _key, _val, _line;

    // read input parameters from parmas.in file
    std::ifstream _infile("params.in", std::ios::in);
    while (getline(_infile, _line)) {
        std::istringstream iss(_line);
        iss >> _key >> _val;
        if (_key == "Lmax") { Lmax = static_cast<uint32_t>(stoul(_val)); }
        else if (_key == "Ngrid") { Ngrid = static_cast<uint32_t>(stoul(_val)); }
        else if (_key == "Nf") { Nf = static_cast<uint32_t>(stoul(_val)); }
        else if (_key == "Tmax") { Tmax = static_cast<uint32_t>(stoul(_val)); }
        else if (_key == "Kmax") { Kmax = static_cast<uint32_t>(stoul(_val)); }
        else if (_key == "nwout") { nwout = static_cast<uint32_t>(stoul(_val)); }
        else if (_key == "Smin") { Smin = stod(_val); }
        else if (_key == "wmin") { wmin = stod(_val); }
        else if (_key == "gamma") { gamma = stod(_val); }
        else if (_key == "dmax") { dmax = stod(_val); }
        else if (_key == "ommax") { ommax = stod(_val); }
        else if (_key == "ommin") { ommin = stod(_val); }
        else if (_key == "alpha_good") { alpha_good = stod(_val); }
        else if (_key == "temp") { beta = 1.0 / (_boltz * stod(_val)); }
        else if (_key == "Norm") { norm_spectrum = stod(_val); }
        else if (_key == "corr_type") { corr_type = _val; }
        else if (_key == "monitor_fit_quality") { std::istringstream(_val) >> std::boolalpha >> monitor_fit_quality; }
        else { std::cout << "Unknown input parameter in params.in file!" << std::endl; exit(1); }
    }
    _infile.close();
}
