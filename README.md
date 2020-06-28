TETRISOM
=============================================================

This program is C++ implementation of an analytic continuation
method proposed by Andrey S. Mishchenko. See
<http://www.cond-mat.de/events/correl12/manuscripts/mishchenko.pdf>
For more technical details, you can see Igor Krivenko's CPC paper
<https://doi.org/10.1016/j.cpc.2019.01.021>


Installation
------------

This project is written in C++11 without any additional dependencies.
You can easily compile it with a compiler supporting *C++11* and *MPI*.

First, you can obtain this code by
```bash
git clone https://github.com/xfzhang-phys/Tetrisom.git ${path_of_tetrisom}
```
Where `${path_of_tetrisom}` is the path on which you want tetrisom code to be located.

For compiling, first you should go to `${path_of_tetrisom}` directory and then change the `CC` variable in `Makefile`.
`CC` should be set to the MPI-C++ compiler in your system. The optimization tags `OPT` should also be changed correspondingly.

After this, you can type `make` in your terminal, and then the code will be installed in `${path_of_tetrisom}` directory.

Usage
-----
Three input files are needed for this code: *params.in*, *Gf.dat* and *Sigma.dat*.

*Gf.dat* is the averaged correlation function $\text{G}(i\omega_n)$ on Matsubara frequency axis.
This file contains three columes: $\omega_n$, $\text{Re}[\text{G}(i\omega_n)]$ and $\text{Im}[\text{G}(i\omega_n)]$.

*Sigma.dat* stores the corresponding standard errors with the same structure as *Gf.dat*.

*params.in* is the setting file. You can find an example in `script` directory.

When running the code, all these three files should be provided in the *running* directory. Then the calculation can be started by
```bash
mpirun -np ${nprocs} ${path_of_tetrisom}/som.x
```

License
-------

TETRISOM is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

TETRISOM is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
TETRISOM (in the file LICENSE.txt in this directory). If not, see
<http://www.gnu.org/licenses/>.
