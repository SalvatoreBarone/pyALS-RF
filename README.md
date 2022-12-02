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

# Using the ready-to-use docker container
pyALS has quite a lot of dependencies. You need to install Yosys (and its dependencies), GHDL (and, again, its dependencies), and so forth.
Before you get a headache, ***you can use the Docker image I have made available to you [here](https://hub.docker.com/r/salvatorebarone/pyals-docker-image).***  

Please, use the following script to run the container, that allows specifying which catalog and which folder to share with the container.
```bash
#!/bin/bash

usage() {
  echo "Usage: $0 -c catalog -s path_to_shared_folder";
  exit 1;
}

while getopts "c:s:" o; do
    case "${o}" in
        c)
            catalog=${OPTARG}
            ;;
        s)
            shared=${OPTARG}
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

if [ -z "${catalog}" ] || [ -z "${shared}" ] ; then
    usage
fi

catalog=`realpath ${catalog}`
shared=`realpath ${shared}`
[ ! -d $shared ] && mkdir -p $shared
xhost local:docker
docker run --rm -e DISPLAY=unix$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix -v ${catalog}:/root/lut_catalog.db -v ${shared}:/root/shared -w /root --privileged -it salvatorebarone/pyals-docker-image /bin/zsh
```

If, on the other hand, you really feel the need to install everything by hand, follow the guide below step by step. 
I'm sure it will be very helpful.
# Running pyALS-RF
pyALS-rf provides several approximation flows, through a unified command line interface. You can select between the following commands:
```
  ps        Performs precision-scaling approximation
  als       Performs ALS approximation
  full      Performs ps and als combined
```

see
```
./pyals-rf --help
```
and
```
./pyals-rf COMMAND --help
```
for more details.

Please kindly note you have to specify the path of the file where synthesized Boolean functions are stored.
You can find a ready to use cache at ```git@github.com:SalvatoreBarone/pyALS-lut-catalog```.
If you do not want to use the one I mentioned, pyALS-rf will perform the exact synthesis as needed.

## The configuration file
Here, I report the basic structure of a configuration file. You will find it within the pyALS root directory.


### Configuration file for the ```ps``` command
```

```

### Configuration file for the ```als``` and ```full``` commands

```

```

#### Configuration file for the ```als-twostep``` and ```full-twostep``` commands

```

```


# Manual Installation
pyALS-rf has quite a lot of dependencies. You need to install Yosys (and its dependencies), GHDL (and, again, its dependencies), and so forth.
Before you get a headache, ***you can use the Docker image I have made available to you [here](https://hub.docker.com/r/salvatorebarone/pyals-docker-image).***  

If, on the other hand, you really feel the need to install everything by hand, follow this guide step by step. 
I'm sure it will be very helpful.
The guide has been tested on Debian 11.

## Preliminaries
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


## Installing Yosys
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

## Installing GHDL
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

## Installing python dependencies
You're almost done. The last step is to install python dependencies. Some of them can be installed automatically, some others must be installed manually.Let's start with the latter ones. 

You must install the [pyAMOSA](https://github.com/SalvatoreBarone/pyAMOSA) module
```
$ git clone https://github.com/SalvatoreBarone/pyAMOSA.git
$ cd pyAMOSA
# python3 setup.py install
$ cd ..
```
and the [pyALSlib](https://github.com/SalvatoreBarone/pyALSlib) module
```
$ git clone https://github.com/SalvatoreBarone/pyALSlib.git
$ cd pyALSlib
# python3 setup.py install
$ cd ..
```

Pertaining to other dependencies, installing them is quite simple, and you just need to issue the following command from within the pyALS directory.
```bash
pip3 install -r requirements.txt 
```


## References
1. Bandyopadhyay, S., Saha, S., Maulik, U., & Deb, K. (2008). A simulated annealing-based multiobjective optimization algorithm: AMOSA. IEEE transactions on evolutionary computation, 12(3), 269-283.
2. Blank, Julian, and Kalyanmoy Deb. "A running performance metric and termination criterion for evaluating evolutionary multi-and many-objective optimization algorithms." 2020 IEEE Congress on Evolutionary Computation (CEC). IEEE, 2020.
