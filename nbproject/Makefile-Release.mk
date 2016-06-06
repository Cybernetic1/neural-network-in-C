#
# Generated Makefile - do not edit!
#
# Edit the Makefile in the project folder instead (../Makefile). Each target
# has a -pre and a -post target defined where you can add customized code.
#
# This makefile implements configuration specific macros and targets.


# Environment
MKDIR=mkdir
CP=cp
GREP=grep
NM=nm
CCADMIN=CCadmin
RANLIB=ranlib
CC=gcc
CCC=g++
CXX=g++
FC=gfortran
AS=as

# Macros
CND_PLATFORM=GNU-Linux-x86
CND_DLIB_EXT=so
CND_CONF=Release
CND_DISTDIR=dist
CND_BUILDDIR=build

# Include project Makefile
include Makefile

# Object Directory
OBJECTDIR=${CND_BUILDDIR}/${CND_CONF}/${CND_PLATFORM}

# Object Files
OBJECTFILES= \
	${OBJECTDIR}/Chinese-test.o \
	${OBJECTDIR}/Q-learning.o \
	${OBJECTDIR}/arithmetic-test.o \
	${OBJECTDIR}/back-prop.o \
	${OBJECTDIR}/backprop-through-time.o \
	${OBJECTDIR}/basic-tests.o \
	${OBJECTDIR}/experiments.o \
	${OBJECTDIR}/genetic-NN.o \
	${OBJECTDIR}/main.o \
	${OBJECTDIR}/maze.o \
	${OBJECTDIR}/real-time-recurrent-learning.o \
	${OBJECTDIR}/stochastic-forward-backward.o \
	${OBJECTDIR}/tic-tac-toe2.o \
	${OBJECTDIR}/visualization.o


# C Compiler Flags
CFLAGS=

# CC Compiler Flags
CCFLAGS=
CXXFLAGS=

# Fortran Compiler Flags
FFLAGS=

# Assembler Flags
ASFLAGS=

# Link Libraries and Options
LDLIBSOPTIONS=

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	"${MAKE}"  -f nbproject/Makefile-${CND_CONF}.mk ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/genifer5c

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/genifer5c: ${OBJECTFILES}
	${MKDIR} -p ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}
	${LINK.cc} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/genifer5c ${OBJECTFILES} ${LDLIBSOPTIONS} -lm -lSDL2 -lgsl -lgslcblas -lsfml-graphics -lsfml-system -lsfml-window

${OBJECTDIR}/Chinese-test.o: Chinese-test.c 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.c) -O2 -std=c99 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/Chinese-test.o Chinese-test.c

${OBJECTDIR}/Q-learning.o: Q-learning.c 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.c) -O2 -std=c99 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/Q-learning.o Q-learning.c

${OBJECTDIR}/arithmetic-test.o: arithmetic-test.c 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.c) -O2 -std=c99 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/arithmetic-test.o arithmetic-test.c

${OBJECTDIR}/back-prop.o: back-prop.c 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.c) -O2 -std=c99 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/back-prop.o back-prop.c

${OBJECTDIR}/backprop-through-time.o: backprop-through-time.c 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.c) -O2 -std=c99 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/backprop-through-time.o backprop-through-time.c

${OBJECTDIR}/basic-tests.o: basic-tests.c 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.c) -O2 -std=c99 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/basic-tests.o basic-tests.c

${OBJECTDIR}/experiments.o: experiments.c 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.c) -O2 -std=c99 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/experiments.o experiments.c

${OBJECTDIR}/genetic-NN.o: genetic-NN.c 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.c) -O2 -std=c99 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/genetic-NN.o genetic-NN.c

${OBJECTDIR}/main.o: main.c 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.c) -O2 -std=c99 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/main.o main.c

${OBJECTDIR}/maze.o: maze.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/maze.o maze.cpp

${OBJECTDIR}/real-time-recurrent-learning.o: real-time-recurrent-learning.c 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.c) -O2 -std=c99 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/real-time-recurrent-learning.o real-time-recurrent-learning.c

${OBJECTDIR}/stochastic-forward-backward.o: stochastic-forward-backward.c 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.c) -O2 -std=c99 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/stochastic-forward-backward.o stochastic-forward-backward.c

${OBJECTDIR}/tic-tac-toe2.o: tic-tac-toe2.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/tic-tac-toe2.o tic-tac-toe2.cpp

${OBJECTDIR}/visualization.o: visualization.c 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.c) -O2 -std=c99 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/visualization.o visualization.c

# Subprojects
.build-subprojects:

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r ${CND_BUILDDIR}/${CND_CONF}
	${RM} ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/genifer5c

# Subprojects
.clean-subprojects:

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
