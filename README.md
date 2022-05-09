# pyALS-RF
Approximate Logic Synthesis of Random-Forest classifiers.

The tool allows applying either or both the approximation methods from  the following papers.

> [Barbareschi, M., Barone, S. & Mazzocca, N. Advancing synthesis of decision tree-based multiple classifier systems: an approximate computing case study. Knowl Inf Syst 63, 1577–1596 (2021) DOI: 0.1007/s10115-021-01565-5](https://doi.org/10.1007/s10115-021-01565-5)


> [M. Barbareschi, S. Barone, N. Mazzocca and A. Moriconi, "A Catalog-based AIG-Rewriting Approach to the Design of Approximate Components" in IEEE Transactions on Emerging Topics in Computing, DOI: 10.1109/TETC.2022.3170502](https://doi.ieeecomputersociety.org/10.1109/TETC.2022.3170502)


Please, cite us!
```
@article{barbareschi2021advancing,
  title={Advancing synthesis of decision tree-based multiple classifier systems: an approximate computing case study},
  author={Barbareschi, Mario and Barone, Salvatore and Mazzocca, Nicola},
  journal={Knowledge and Information Systems},
  volume={63},
  number={6},
  pages={1577--1596},
  year={2021},
  publisher={Springer}
}

@article{barbareschi2022catalog,
  title={A Catalog-based AIG-Rewriting Approach to the Design of Approximate Components},
  author={Barbareschi, Mario and Barone, Salvatore and Mazzocca, Nicola and Moriconi, Alberto},
  journal={IEEE Transactions on Emerging Topics in Computing},
  year={2022},
  publisher={IEEE}
}
```
## Installation
pyALS-rf has quite a lot of dependencies. You need to install Yosys (and its dependencies), GHDL (and, again, its dependencies), and so forth.
Before you get a headache, ***you can use the Docker image I have made available to you [here](https://hub.docker.com/r/salvatorebarone/pyals-docker-image).***  

If, on the other hand, you really feel the need to install everything by hand, follow this guide step by step. 
I'm sure it will be very helpful.
The guide has been tested on Debian 11.

### Preliminaries
You need to install some basic dependencies. So, run
```
# apt-get install --fix-missing -y git bison clang cmake curl flex fzf g++ gnat gawk libffi-dev libreadline-dev libsqlite3-dev  libssl-dev make p7zip-full pkg-config python3 python3-dev python3-pip tcl-dev vim-nox wget xdot zlib1g-dev zlib1g-dev zsh libboost-dev libboost-filesystem-dev libboost-graph-dev libboost-iostreams-dev libboost-program-options-dev libboost-python-dev libboost-serialization-dev libboost-system-dev libboost-thread-dev
```

You also need to create some symbolic links.
```
# ln -s /usr/lib/x86_64-linux-gnu/libtinfo.so /usr/lib/x86_64-linux-gnu/libtinfo.so.5
# ln -fs /usr/lib/x86_64-linux-gnu/libboost_python39.a /usr/lib/x86_64-linux-gnu/libboost_python.a
# ln -fs /usr/lib/x86_64-linux-gnu/libboost_python39.so /usr/lib/x86_64-linux-gnu/libboost_python.so
```
Please, kindly note you are required to amend any differences concerning the python version. I'm using python 3.9 here.


### Installing Yosys
First, you need to clone Yosys from its public repository
```
$ git clone https://github.com/YosysHQ/yosys
```
This will create a ```yosys``` sub-directory inside your current directory. Now move into the ```yosys``` directory, and create a ```Makefile.conf``` file.
```
$ cd yosys
$ touch Makefile.conf
```
Paste the following into the ```Makefile.conf``` file.
```
CONFIG := clang
CXXFLAGS += -I/usr/include/python3.9/ -fPIC
ENABLE_LIBYOSYS=1
ENABLE_PYOSYS=1
PYTHON_EXECUTABLE=/usr/bin/python3 
PYTHON_VERSION=3.9 
PYTHON_CONFIG=python3-config 
PYTHON_DESTDIR=/usr/local/lib/python3.9/dist-packages
BOOST_PYTHON_LIB=/usr/lib/x86_64-linux-gnu/libboost_python.so -lpython3.9
```
Please, kindly note you are required to amend any differences concerning the python version. I'm using python 3.9 here.
Now you need to a little quick fix to yosys: edit the ```kernel/yosys.cc``` file, searching for the definition of the 
```run_pass``` function. Comment the call to the ```log``` function as follows.
```
void run_pass(std::string command, RTLIL::Design *design)
{
	if (design == nullptr)
		design = yosys_design;

	//log("\n-- Running command `%s' --\n", command.c_str());

	Pass::call(design, command);
}
```
This will remove redundant logs while running the optimizer.
Ok, now you are ready.
```
$ make -j `nproc`
# make install
# ln -s `realpath yosys` /usr/bin
# ln -s `realpath yosys-abc` /usr/bin
```

### Installing GHDL
GHDL and its Yosys plugin are required to process VHDL-encoded designs. 
Please, kindly note that you will be able to successfully install the GHDL Yosys plugin only if you successfully installed Yosys. 
Let's install GHDL first. As always, you need to clone GHDL from ist public repository and compile it.
```
$ git clone https://github.com/ghdl/ghdl.git
$ cd ghdl
$ ./configure --prefix=/usr/local
$ make
# make install
```
The same applies to its Yosys plugin. 
```
$ git clone https://github.com/ghdl/ghdl-yosys-plugin.git
$ cd ghdl-yosys-plugin
$ make
# make install
```

### Installing python dependencies
You're almost done, the last step is to install python dependencies. It's quite simple, and you just need to issue the following command from within the pyALS directory.
```
pip3 install -r requirements.txt 
```

## Running pyALS-RF
pyALS-rf provides several approximation flows, through a unified command line interface. You can select between the following commands:
```
  bitwidth      Performs precision-scaling approximation
  als-onestep   Performs one-step ALS approximation
  als-twostep   Performs two-steps ALS approximation
  full-onestep  Performs one-step full approximation (both ps and als)
  full-twostep  Performs two-steps full approximation (both ps and als)
  dump          just dumps the classifier and exit
```

Please kindly note you have to specify the path of the file where synthesized Boolean functions are stored.
You can find a ready to use cache at ```git@github.com:SalvatoreBarone/pyALS-lut-catalog```.
If you do not want to use the one I mentioned, pyALS-rf will perform the exact synthesis as needed.

### The ```dump``` command

This command just dumps the classifier, then exit. It is for debugging purpose.

Example:
```
./pyals-rf dump --pmml example/pmml/random_forest.pmml
```

### The ``` bitwidth ``` command
It allows performing the approximation flow from

> [Barbareschi, M., Barone, S. & Mazzocca, N. Advancing synthesis of decision tree-based multiple classifier systems: an approximate computing case study. Knowl Inf Syst 63, 1577–1596 (2021)](https://doi.org/10.1007/s10115-021-01565-5)

Usage:
```
pyals-rf bitwidth [OPTIONS]
```
Options:
```
  -c, --config TEXT   path of the configuration file  [required]
  -p, --pmml TEXT     specify the input PMML file  [required]
  -d, --dataset TEXT  specify the file name for the input dataset  [required]
  -o, --output TEXT   Output directory. Everything will be placed there. [required]
  -i, --improve TEXT  Run again the workflow  using previous Pareto set as initial archive
```
Esample:
```
./pyals-rf bitwidth --pmml example/pmml/random_forest.pmml --dataset example/test_dataset/random_forest.txt --config example/configs/config_ps.ini --output output_rf/
```

### The ``` als-onestep ``` command

One-step approximation strategy, using approximate logic synthesis on assertion functions.
All the decision-trees involved in the classification are approximated simultaneously. 
The set of decision variables of the optimization problem consists of the union of the sets of decision variables that
allow each of the individual trees to be optimized. Hence, it is easy for the solution space to explode quickly, but 
(theoretically) this strategy should allow converging towards the actual optimum between classification-accuracy loss
and hardware resource savings.

Usage: 
```
pyals-rf als-onestep [OPTIONS]
```
Options:
```
  -c, --config TEXT   path of the configuration file  [required]
  -p, --pmml TEXT     specify the input PMML file  [required]
  -d, --dataset TEXT  specify the file name for the input dataset  [required]
  -o, --output TEXT   Output directory. Everything will be placed there. [required]
  -i, --improve TEXT  Run again the workflow  using previous Pareto set as initial archive
  --help              Show this message and exit.
```
Esample:
```
./pyals-rf als-onestep --pmml example/pmml/random_forest.pmml --dataset example/test_dataset/random_forest.txt --config example/configs/config_ps.ini --output output_rf/
```

### The ``` full-onestep ``` command

One-step approximation strategy, using both the precision scaling of features, and the approximate logic synthesis on assertion functions.
All the decision-trees involved in the classification are approximated simultaneously. 
The set of decision variables of the optimization problem consists of the union of the sets of decision variables that
allow each of the individual trees to be optimized. Hence, it is easy for the solution space to explode quickly, but 
(theoretically) this strategy should allow converging towards the actual optimum between classification-accuracy loss
and hardware resource savings.

Usage: 
```
pyals-rf full-onestep [OPTIONS]
```
Options:
```
  -c, --config TEXT   path of the configuration file  [required]
  -p, --pmml TEXT     specify the input PMML file  [required]
  -d, --dataset TEXT  specify the file name for the input dataset  [required]
  -o, --output TEXT   Output directory. Everything will be placed there. [required]
  -i, --improve TEXT  Run again the workflow  using previous Pareto set as initial archive
  --help              Show this message and exit.
```
Esample:
```
./pyals-rf full-onestep --pmml example/pmml/random_forest.pmml --dataset example/test_dataset/random_forest.txt --config example/configs/config_ps.ini --output output_rf/
```

### The ``` als-twostep ``` command

Two-step approximation strategy, using approximate logic synthesis on assertion functions.
Each tree is independently approximated; then, the classifier as a whole is approximated. 
This strategy allows to drastically reduce the size of the solution space, but, on the other hand, it may result in
sub-optimum points between classification-accuracy loss and hardware resource savings.

Usage: 
```
pyals-rf als-twostep [OPTIONS]
```
Options:
```
  -c, --config TEXT   path of the configuration file  [required]
  -p, --pmml TEXT     specify the input PMML file  [required]
  -d, --dataset TEXT  specify the file name for the input dataset  [required]
  -o, --output TEXT   Output directory. Everything will be placed there. [required]
  -i, --improve TEXT  Run again the workflow  using previous Pareto set as initial archive
  --help              Show this message and exit.
```
Esample:
```
./pyals-rf als-twostep --pmml example/pmml/random_forest.pmml --dataset example/test_dataset/random_forest.txt --config example/configs/config_ps.ini --output output_rf/
```

### The ``` full-twostep ``` command
Two-step approximation strategy, using both the precision scaling of features, and the approximate logic synthesis on assertion functions.
Each tree is independently approximated; then, the classifier as a whole is approximated. 
This strategy allows to drastically reduce the size of the solution space, but, on the other hand, it may result in
sub-optimum points between classification-accuracy loss and hardware resource savings.

Usage: 
```
pyals-rf full-twostep [OPTIONS]
```
Options:
```
  -c, --config TEXT   path of the configuration file  [required]
  -p, --pmml TEXT     specify the input PMML file  [required]
  -d, --dataset TEXT  specify the file name for the input dataset  [required]
  -o, --output TEXT   Output directory. Everything will be placed there. [required]
  -i, --improve TEXT  Run again the workflow  using previous Pareto set as initial archive
  --help              Show this message and exit.
```
Esample:
```
./pyals-rf full-twostep --pmml example/pmml/random_forest.pmml --dataset example/test_dataset/random_forest.txt --config example/configs/config_ps.ini --output output_rf/
```

### The configuration file
Here, I report the basic structure of a configuration file. You will find it within the pyALS root directory.


#### Configuration file for the ```bitwidth``` command
```
[error]
max_loss = 5                    ; the error threshold, in terms of classification accuracy loss

[amosa]
archive_hard_limit = 30         ; Archive hard limit for the AMOSA optimization heuristic, see [1]
archive_soft_limit = 50         ; Archive soft limit for the AMOSA optimization heuristic, see [1]
archive_gamma = 2               ; Gamma parameter for the AMOSA optimization heuristic, see [1]
hill_climbing_iterations = 250  ; the number of iterations performed during the initial hill-climbing refinement, see [1];
initial_temperature = 500       ; Initial temperature of the matter for the AMOSA optimization heuristic, see [1]
final_temperature = 0.000001    ; Final temperature of the matter for the AMOSA optimization heuristic, see [1]
cooling_factor =  0.9           ; It governs how quickly the temperature of the matter decreases during the annealing process, see [1]
annealing_iterations = 600      ; The amount of refinement iterations performed during the main-loop of the AMOSA heuristic, see [1]
annealing_strength = 1          ; Governs the strength of random perturbations during the annealing phase; specifically, the number of variables whose value is affected by perturbation.
early_termination = 20          ; Early termination window. See [2]. Set it to zero in order to disable early-termination. Default is 20.
```

#### Configuration file for the ```als-onestep``` and ```full-onestep``` commands

```
[als]
cut_size = 4                    ; specifies the "k" for AIG-cuts, or, alternatively, the k-LUTs for LUT-mapping during cut-enumeration
catalog = lut_catalog.db        ; This is the path of the file where synthesized Boolean functions are stored. You can find a ready to use cache at git@github.com:SalvatoreBarone/LUTCatalog.git
solver = btor                   ; SAT-solver to be used. It can be either btor (Boolector) or z3 (Z3-solver)
timeout = 60000                 ; Timeout (in ms) for the Exact synthesis process. You don't need to change its default value.

[error]
max_loss = 5                    ; the error threshold, in terms of classification accuracy loss

[amosa]
archive_hard_limit = 30         ; Archive hard limit for the AMOSA optimization heuristic, see [1]
archive_soft_limit = 50         ; Archive soft limit for the AMOSA optimization heuristic, see [1]
archive_gamma = 2               ; Gamma parameter for the AMOSA optimization heuristic, see [1]
hill_climbing_iterations = 250  ; the number of iterations performed during the initial hill-climbing refinement, see [1];
initial_temperature = 500       ; Initial temperature of the matter for the AMOSA optimization heuristic, see [1]
final_temperature = 0.000001    ; Final temperature of the matter for the AMOSA optimization heuristic, see [1]
cooling_factor =  0.9           ; It governs how quickly the temperature of the matter decreases during the annealing process, see [1]
annealing_iterations = 600      ; The amount of refinement iterations performed during the main-loop of the AMOSA heuristic, see [1]
annealing_strength = 1          ; Governs the strength of random perturbations during the annealing phase; specifically, the number of variables whose value is affected by perturbation.
early_termination = 20          ; Early termination window. See [2]. Set it to zero in order to disable early-termination. Default is 20.
```

#### Configuration file for the ```als-twostep``` and ```full-twostep``` commands

```
[als]
cut_size = 4                    ; specifies the "k" for AIG-cuts, or, alternatively, the k-LUTs for LUT-mapping during cut-enumeration
catalog = lut_catalog.db        ; This is the path of the file where synthesized Boolean functions are stored. You can find a ready to use cache at git@github.com:SalvatoreBarone/LUTCatalog.git
solver = btor                   ; SAT-solver to be used. It can be either btor (Boolector) or z3 (Z3-solver)
timeout = 60000                 ; Timeout (in ms) for the Exact synthesis process. You don't need to change its default value.

[error]
max_freq = 10                   ; the maximum allowed error frequency for approximate assertion functions
max_loss = 5                    ; the error threshold, in terms of classification accuracy loss

;; options for the first stage optimizer
[amosa1]                        
archive_hard_limit = 30         ; Archive hard limit for the AMOSA optimization heuristic, see [1]
archive_soft_limit = 50         ; Archive soft limit for the AMOSA optimization heuristic, see [1]
archive_gamma = 2               ; Gamma parameter for the AMOSA optimization heuristic, see [1]
hill_climbing_iterations = 250  ; the number of iterations performed during the initial hill-climbing refinement, see [1];
initial_temperature = 500       ; Initial temperature of the matter for the AMOSA optimization heuristic, see [1]
final_temperature = 0.000001    ; Final temperature of the matter for the AMOSA optimization heuristic, see [1]
cooling_factor =  0.9           ; It governs how quickly the temperature of the matter decreases during the annealing process, see [1]
annealing_iterations = 600      ; The amount of refinement iterations performed during the main-loop of the AMOSA heuristic, see [1]
annealing_strength = 1          ; Governs the strength of random perturbations during the annealing phase; specifically, the number of variables whose value is affected by perturbation.
early_termination = 20          ; Early termination window. See [2]. Set it to zero in order to disable early-termination. Default is 20.

;; options for the second stage optimizer
[amosa2]                        
archive_hard_limit = 30         ; Archive hard limit for the AMOSA optimization heuristic, see [1]
archive_soft_limit = 50         ; Archive soft limit for the AMOSA optimization heuristic, see [1]
archive_gamma = 2               ; Gamma parameter for the AMOSA optimization heuristic, see [1]
hill_climbing_iterations = 250  ; the number of iterations performed during the initial hill-climbing refinement, see [1];
initial_temperature = 500       ; Initial temperature of the matter for the AMOSA optimization heuristic, see [1]
final_temperature = 0.000001    ; Final temperature of the matter for the AMOSA optimization heuristic, see [1]
cooling_factor =  0.9           ; It governs how quickly the temperature of the matter decreases during the annealing process, see [1]
annealing_iterations = 600      ; The amount of refinement iterations performed during the main-loop of the AMOSA heuristic, see [1]
annealing_strength = 1          ; Governs the strength of random perturbations during the annealing phase; specifically, the number of variables whose value is affected by perturbation.
early_termination = 20          ; Early termination window. See [2]. Set it to zero in order to disable early-termination. Default is 20.

```

## References
1. Bandyopadhyay, S., Saha, S., Maulik, U., & Deb, K. (2008). A simulated annealing-based multiobjective optimization algorithm: AMOSA. IEEE transactions on evolutionary computation, 12(3), 269-283.
2. Blank, Julian, and Kalyanmoy Deb. "A running performance metric and termination criterion for evaluating evolutionary multi-and many-objective optimization algorithms." 2020 IEEE Congress on Evolutionary Computation (CEC). IEEE, 2020.
