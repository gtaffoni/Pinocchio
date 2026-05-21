# *****************************************************************
# *                        PINOCCHIO  V5.1                        *
# *  (PINpointing Orbit-Crossing Collapsed HIerarchical Objects)  *
# *****************************************************************
#
# This code was written by
# Pierluigi Monaco, Tom Theuns, Giuliano Taffoni, Marius Lepinzan,
# Chiara Moretti, Luca Tornatore, David Goz, Tiago Castro
# Copyright (C) 2025
#
# github: https://github.com/pigimonaco/Pinocchio
# web page: http://adlibitum.oats.inaf.it/monaco/pinocchio.html
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA


# Directory settings #

RUNDIR = ./

# Executable name #

EXEC = pinocchio.x

# -----------------------------------------------------------------------
# Build toggles (override from command line, e.g. make OMP=YES GPU=YES)
# -----------------------------------------------------------------------

DEBUG    ?= NO
OMP      ?= NO
# USE_HEFFTE: set to YES to use HeFFTe backend instead of pfft
USE_HEFFTE ?= NO
# GPU: enable OpenMP GPU offload (requires USE_HEFFTE=YES and nvc compiler)
GPU      ?= NO
# FULL_GPU: all FFT on GPU (GPU_OMP_FULL), implies USE_HEFFTE=YES
FULL_GPU ?= NO

########## Preprocessor Definitions ##########

# Displacement LPT order #

OPTIONS += -DTWO_LPT
OPTIONS += -DTHREE_LPT

# PLC reconstruction

# OPTIONS += -DPLC

# Dynamics of triaxial collapse #

OPTIONS += -DELL_CLASSIC
# OPTIONS += -DELL_SNG
# OPTIONS += -DTABULATED_CT

# Building groups and fragmentation #

# OPTIONS += -DCLASSIC_FRAGMENTATION

# Output #

# OPTIONS += -DSNAPSHOT
# OPTIONS += -DLIGHT_OUTPUT
# OPTIONS += -DLONGIDS
# OPTIONS += -DDOUBLE_PRECISION_PRODUCTS

# Beyond LambdaCDM models

# these are for neutrino cosmology
# OPTIONS += -DSCALE_DEPENDENT
# OPTIONS += -DREAD_PK_TABLE
# OPTIONS += -DONLY_MATTER_POWER
# OPTIONS += -DRECOMPUTE_DISPLACEMENTS

# add also these for f(R) gravity
# OPTIONS += -DMOD_GRAV_FR
# OPTIONS += -DFR0=1.e-8

# Other options #

# OPTIONS += -DWHITENOISE
OPTIONS += -DNORADIATION


########## Preprocessor Definitions end ##########

# -----------------------------------------------------------------------
# FFT Backend Selection
# USE_HEFFTE=YES selects HeFFTe; default keeps pfft.
# FULL_GPU=YES forces USE_HEFFTE=YES (HeFFTe is the only GPU-capable backend).
# -----------------------------------------------------------------------

ifeq ($(FULL_GPU), YES)
USE_HEFFTE = YES
endif

ifeq ($(GPU), YES)
USE_HEFFTE = YES
endif

# -----------------------------------------------------------------------
# GPU OpenMP offload flags
# -----------------------------------------------------------------------

ifeq ($(GPU), YES)
OPTIONS  += -DGPU_OMP
GPU_FLAG  = -gpu=ccnative -target=gpu
endif

ifeq ($(FULL_GPU), YES)
OPTIONS  += -DGPU_OMP_FULL
GPU_FLAG  = -gpu=ccnative -target=gpu
endif

# -----------------------------------------------------------------------
# HeFFTe preprocessor flag
# -----------------------------------------------------------------------

ifeq ($(USE_HEFFTE), YES)
OPTIONS += -DUSE_HEFFTE
endif

# -----------------------------------------------------------------------
# OpenMP Configuration
# -----------------------------------------------------------------------

ifeq ($(OMP), YES)
OPTIONS  += -DUSE_FFT_THREADS
OMP_FFTW  = -lfftw3_omp -lfftw3_threads
ifeq ($(GPU), YES)
OMP_FLAG  = -mp=multicore,gpu
else ifeq ($(FULL_GPU), YES)
OMP_FLAG  = -mp=multicore,gpu
else
OMP_FLAG  = -fopenmp
endif
else
OMP_FLAG  =
OMP_FFTW  =
endif

########## System-specific Configuration ############

SYSTYPE ?= "docker"

###################### DOCKER  ######################

ifeq ($(SYSTYPE),"docker")
CC          =  mpicc
CDEBUG      = -ggdb3 -Wall $(OMP_FLAG)
COPTIMIZED  = -O3 -Wno-unused-result $(OMP_FLAG)
FFTW_LIBR   = -L/usr/local/fftw3/lib -lfftw3_mpi -lfftw3 $(OMP_FFTW)
FFTW_INCL   = -I/usr/local/fftw3/include
PFFT_LIBR   = -L/usr/local/pfft/lib -lpfft
PFFT_INCL   = -I/usr/local/pfft/include

MPI_LIBR    =
MPI_INCL    =
GSL_LIBR    = -L$(HOME)/lib -lgsl -lgslcblas -lm
GSL_INCL    = -I$(HOME)/include/gsl
endif




###################### BASE ######################

ifeq ($(SYSTYPE),"base")
CC          =  mpicc
CDEBUG      = -ggdb3 -Wall $(OMP_FLAG)
COPTIMIZED  = -O3 -Wno-unused-result $(OMP_FLAG)
FFTW_LIBR   = -L$(FFTW_LIB) -lfftw3_mpi -lfftw3 $(OMP_FFTW)
FFTW_INCL   = -I$(HOME)/include
PFFT_LIBR   = -L$(HOME)/lib -lpfft
PFFT_INCL   = -I$(HOME)/include

MPI_LIBR    =
MPI_INCL    =
GSL_LIBR    = -L$(HOME)/lib -lgsl -lgslcblas -lm
GSL_INCL    = -I$(HOME)/include/gsl
endif

###################### MARCONI ######################

ifeq ($(SYSTYPE),"Marconi100")
CC          =  mpicc
CDEBUG      = -ggdb3 -Wall $(OMP_FLAG)
COPTIMIZED  = -O2 -Wno-unused-result  $(OMP_FLAG)
FFTW_LIBR   = -L$(HOME)/lib -lfftw3_mpi -lfftw3 $(OMP_FFTW)
FFTW_INCL   = -I$(HOME)/include
PFFT_LIBR   = -lpfft
PFFT_INCL   =
MPI_LIBR    = -L$(SMPI_ROOT)/lib -lmpi_ibm
MPI_INCL    = -I$(SMPI_ROOT)/include
GSL_LIBR    = -L$(GSL_LIB) -lgsl -lgslcblas -lm
GSL_INCL    = -I$(GSL_INC)
endif

###################### PLEIADI ######################

ifeq ($(SYSTYPE),"pleiadi")
CC          =  mpicc
CDEBUG      = -g -Wall $(OMP_FLAG)
COPTIMIZED  = -O3 -Wno-unused-result $(OMP_FLAG)
FFTW_LIBR   = -L/opt/cluster/spack/opt/spack/linux-centos7-broadwell/gcc-11.2.0/fftw-3.3.10-bap3ootbcmiypnmo7rouk57jbdhfjuty/lib -lfftw3_mpi -lfftw3 $(OMP_FFTW)
FFTW_INCL   = -I/opt/cluster/spack/opt/spack/linux-centos7-broadwell/gcc-11.2.0/fftw-3.3.10-bap3ootbcmiypnmo7rouk57jbdhfjuty/include
PFFT_LIBR   = -L/opt/cluster/spack/opt/spack/linux-centos7-broadwell/gcc-11.2.0/pfft-1.0.8-alpha-qy6lqerctfrhjzaoamzjhbpga2ofejqf/lib -lpfft
PFFT_INCL   = -I/opt/cluster/spack/opt/spack/linux-centos7-broadwell/gcc-11.2.0/pfft-1.0.8-alpha-qy6lqerctfrhjzaoamzjhbpga2ofejqf/include
MPI_LIBR    =
MPI_INCL    =
GSL_LIBR    = -L/opt/cluster/spack/opt/spack/linux-centos7-broadwell/gcc-11.2.0/gsl-2.7.1-uehbowahpvubxundtx3cz4a4a5sdwzun/lib -lgsl -lgslcblas -lm
GSL_INCL    = -I/opt/cluster/spack/opt/spack/linux-centos7-broadwell/gcc-11.2.0/gsl-2.7.1-uehbowahpvubxundtx3cz4a4a5sdwzun/include
endif


###################### LEONARDO (HeFFTe + NVHPC) ######################
# SYSTYPE=LeonardoBoost  USE_HEFFTE=YES  GPU=YES  or  FULL_GPU=YES
# Compiler: nvc (NVIDIA HPC SDK)
# Libraries: HeFFTe, FFTW, GSL, OpenMPI 4.1.6

ifeq ($(SYSTYPE), "LeonardoBoost")

CC          =  nvc
CDEBUG      = -g -O0 -Minfo=all -v -Mneginfo $(GPU_FLAG) $(OMP_FLAG)
COPTIMIZED  = -O3 -fast -Minfo=all -v -Mneginfo $(GPU_FLAG) $(OMP_FLAG)

MPI         = /leonardo/prod/opt/libraries/openmpi/4.1.6/nvhpc--24.3/
MPI_LIBR    = -L$(MPI)/lib -lmpi
MPI_INCL    = -I$(MPI)/include

LIBDIR      = /leonardo_scratch/fast/CNHPC_1498509/lib/nvhpc-23.11
FFTW        = $(LIBDIR)/fftw/fftw-3.3.10
FFTW_LIBR   = -L$(FFTW)/lib -lfftw3_mpi -lfftw3 $(OMP_FFTW)
FFTW_INCL   = -I$(FFTW)/include

PFFT        = $(LIBDIR)/pfft/pfft
PFFT_LIBR   = -L$(PFFT)/lib -lpfft
PFFT_INCL   = -I$(PFFT)/include

HEFFTE_DIR  = /leonardo/pub/userexternal/dgoz0000/libraries
HEFF_LIBR   = -L$(HEFFTE_DIR)/heffte_nvidia/lib64 -lheffte
HEFF_INCL   = -I$(HEFFTE_DIR)/heffte_nvidia/include

GSL         = $(LIBDIR)/gsl/gsl-2.7.1
GSL_LIBR    = -L$(GSL)/lib -lgsl -lgslcblas -lm
GSL_INCL    = -I$(GSL)/include

PMT         = /leonardo_scratch/fast/CNHPC_1498509/lib/pmt/local
PMT_LIBR    = -L$(PMT)/lib64 -lpmt
PMT_INCL    = -I$(PMT)/include

endif


########## System-specific Configuration end ##########

# -----------------------------------------------------------------------
# Aggregate include and library paths.
# - For the pfft (default CPU) path: use PFFT + FFTW.
# - For the heffte path: use HEFF + FFTW (still needed for thread init).
# -----------------------------------------------------------------------

ifeq ($(USE_HEFFTE), YES)
INC = $(HEFF_INCL) $(FFTW_INCL) $(MPI_INCL) $(GSL_INCL)
LIB = -lm $(HEFF_LIBR) $(FFTW_LIBR) $(MPI_LIBR) $(GSL_LIBR)
else
INC = $(PFFT_INCL) $(FFTW_INCL) $(MPI_INCL) $(GSL_INCL)
LIB = -lm $(PFFT_LIBR) $(FFTW_LIBR) $(MPI_LIBR) $(GSL_LIBR)
endif

# Compiler options : choose from CDEBUG or COPTIMIZED #

ifeq ($(DEBUG), YES)
COPTS = $(CDEBUG)
else
COPTS = $(COPTIMIZED)
endif

# -----------------------------------------------------------------------
# Source files
#
# Default (CPU pfft path): fmax-pfft.o + parallel_sort.o
# HeFFTe path (USE_HEFFTE=YES): fmax-heffte.o  (no pfft, keep parallel_sort)
# GPU path: additionally includes collapse_times_GPU.o,
#           cubic_spline_interpolation.o, cubic_spline_interpolation_full_GPU.o
# -----------------------------------------------------------------------

# Base objects (always compiled)
OBJECTS_BASE = fmax.o variables.o initialization.o collapse_times.o GenIC.o \
	ReadParamfile.o allocations.o LPT.o distribute.o \
	fragment.o build_groups.o parallel_sort.o write_halos.o write_snapshot.o cosmo.o

# FFT backend object: heffte or pfft
ifeq ($(USE_HEFFTE), YES)
FFT_OBJ = fmax-heffte.o
else
FFT_OBJ = fmax-pfft.o
endif

# GPU-specific objects (compiled only when GPU or FULL_GPU)
GPU_OBJECTS =
ifeq ($(GPU), YES)
GPU_OBJECTS += collapse_times_GPU.o cubic_spline_interpolation.o
endif
ifeq ($(FULL_GPU), YES)
GPU_OBJECTS += collapse_times_GPU.o cubic_spline_interpolation.o \
               cubic_spline_interpolation_full_GPU.o
endif

OBJECTS = $(OBJECTS_BASE) $(FFT_OBJ) $(GPU_OBJECTS)

# Main targets and rules

pinocchio: $(OBJECTS) pinocchio.o Makefile
	$(CC) $(INC) $(COPTS) $(OPTIONS) -o $(RUNDIR)$(EXEC) pinocchio.o $(OBJECTS) $(LIB)

run_planner: $(OBJECTS) run_planner.o Makefile
	$(CC) $(INC) $(COPTS) -o run_planner run_planner.o $(OBJECTS) $(LIB)

# Generic rule for compiling source files #

%.o: %.c Makefile pinocchio.h def_splines.h
	$(CC) $(INC) $(COPTS) $(OPTIONS) -c $<

# Clean target #

clean:
	\rm -f *.o *~ $(EXEC) run_planner

###############################################################
# Documentation
###############################################################
# Generates HTML docs from in-source Doxygen comments.
#
# Usage:
#   make docs            # builds HTML docs under ../docs/build/html
#   make docs-clean      # removes generated docs
#
# Requirements:
#   - doxygen installed and available on PATH
#   - config: ../docs/Doxyfile (provided in repo)

.PHONY: docs docs-clean

DOXYGEN ?= doxygen
DOCDIR  ?= ../docs
DOXYCFG ?= $(DOCDIR)/Doxyfile

docs:
	@command -v $(DOXYGEN) >/dev/null 2>&1 || { \
	  echo "Error: doxygen not found. Install doxygen and re-run 'make docs'."; \
	  exit 1; \
	}
	@echo "Generating documentation with $$( $(DOXYGEN) --version )..."
	@cd $(DOCDIR) && $(DOXYGEN) Doxyfile
	@echo "Docs available at $(DOCDIR)/build/html/index.html"

docs-clean:
	@rm -rf $(DOCDIR)/build
