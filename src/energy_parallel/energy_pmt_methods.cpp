#include <pmt.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <sched.h>
#include <cstring>
#include <string>
#include <map>
#include <cmath>


/******************************************************/
#include "energy_pmt_methods.hpp"
/******************************************************/

struct EnergyState
{
  pmt::State   start;
  pmt::State   stop;
  double       joules;
  double       watts;
  double       seconds;
  unsigned int count;
};

static std::vector<bool> PMT_ERROR;



/************************************************************
DEFINE THE MPI NAMESPACE
************************************************************/

namespace rapl // CPU PROFILING
{
  static std::vector<std::unique_ptr<pmt::PMT>> sensor_cpu;
  static std::vector<std::map<std::string, EnergyState>> state_cpu;
}

namespace gpu // GPU PROFILING
{
  static std::vector<std::map<int, std::unique_ptr<pmt::PMT>>> sensor_gpu;
  static std::vector<std::map<int, std::map<std::string, EnergyState>>> state_gpu;
}

//namespace mpi
//{
#include <mpi.h>
void PMT_err(const int task)
{
  std::cout << "\n\t Task: " << task << ": PMT Error \n" << std::endl;
  
  return;
}

#define CPU_ID_ENTRY_IN_PROCSTAT 39

int read_proc__self_stat( int, int * );

int get_cpu_id( void )
{

 #if defined(_GNU_SOURCE)                              // GNU SOURCE ------------

  return  sched_getcpu( );

 #else

 #ifdef SYS_getcpu                                     //     direct sys call ---

  int cpuid;
  if ( syscall( SYS_getcpu, &cpuid, NULL, NULL ) == -1 )
    return -1;
  else
    return cpuid;

 #else

  int val;
  if ( read_proc__self_stat( CPU_ID_ENTRY_IN_PROCSTAT, &val ) == -1 )
    return -1;

  return (int)val;

 #endif                                                // -----------------------
 #endif

}


int get_socket_id( int cpuid )
{

  FILE *file = fopen( "/proc/cpuinfo", "r" );
  if (file == NULL )
    return -1;

  char *buffer = NULL;
  int socket = -1;
  int get_socket = 0;

  while ( socket < 0 )
    {
      size_t  len;
      ssize_t read;
      read = getline( &buffer, &len, file );
      if( read > 0 ) {
	switch( get_socket ) {
	case 0: {       if( strstr( buffer, "processor" ) != NULL ) {
	      char *separator = strstr( buffer, ":" );
	      int   proc_num  = atoi( separator + 1 );
	      get_socket = (proc_num == cpuid ); } } break;
	case 1: { if( strstr( buffer, "physical id" ) != NULL ) {
	      char *separator = strstr( buffer, ":" );
	      socket    = atoi( separator + 1 ); } } break;
	}} else if ( read == -1 ) break;
    }

  fclose(file);
  free(buffer);

  return socket;
}


MPI_Comm get_my_socket_communicator(MPI_Comm my_node_comm, int my_node_rank, int my_node_ntasks)
{

  MPI_Comm my_socket_comm = MPI_COMM_NULL;
  /* Get my socket ID and then construct the socket communicator */

 #if defined(_MPICH_) // Get socket ID in order to create socket communicator with MPICH
  /* Get cpu ID and socket ID */
  int my_cpu_id                      = get_cpu_id();
  int my_socket_id[my_node_ntasks];

  /* Assign a socket to the MPI ranks in each node */
  my_socket_id[my_node_rank]         = get_socket_id(my_cpu_id);

  my_socket_comm = MPI_COMM_NULL;

  const int ret = MPI_Comm_split(my_node_comm, my_socket_id[my_node_rank],
				 my_node_rank, &my_socket_comm);

  /* Check that return is MPI_SUCCESS */
  if (ret != MPI_SUCCESS)
    {
      char estring[MPI_MAX_ERROR_STRING];
      int resultlen;
      MPI_Error_string(ret, estring, &resultlen);
    }
  
 #else // OpenMPI case
  
  const int ret = MPI_Comm_split_type(my_node_comm, OMPI_COMM_TYPE_SOCKET, 0,
				      MPI_INFO_NULL, &my_socket_comm);

  /* Check that return is MPI_SUCCESS */
  if (ret != MPI_SUCCESS)
    {
      char estring[MPI_MAX_ERROR_STRING];
      int resultlen;
      MPI_Error_string(ret, estring, &resultlen);
    }
  
 #endif // defined(_MPICH_)

  return my_socket_comm;
  
}


/* ######################################################### */
numa my_numa;           // NUMA CLASS
measures my_measures;   // MEASURE CLASS
gpus my_gpus;           // GPU CLASS
//std::vector<int> devID; // DEVICE ID
int* devID; // DEVICE ID
/* ######################################################### */
  
void numa::create_communicators()
{
  my_node_comm = MPI_COMM_NULL;
  /* Create the intra-node communicator from original ex: MPI_COMM_WORLD */
  MPI_Comm_split_type(world_comm, MPI_COMM_TYPE_SHARED, 0,
		      MPI_INFO_NULL, &my_node_comm);
  MPI_Comm_rank(my_node_comm, &my_node_rank); // Needed below
  MPI_Comm_rank(my_node_comm, &my_node_ntasks); // Needed below
    
  /* Create the intra-socket communicator from the intra-node one 
     Ex: the communicator which groups all tasks in the single 
     sockets on each node. If the node is single socket the 
     intra-socket communicator and the intra-node communicator
     are equal */


  /* OpenMPI vs MPICH case with _MPICH_ compilation flag */
  my_socket_comm = get_my_socket_communicator(my_node_comm, my_node_rank, my_node_ntasks);
  MPI_Comm_rank(my_socket_comm, &my_socket_rank); // Needed below
    
    
  /* Create the communicator which groups the master of each socket
     in the original communicator ex: MPI_COMM_WORLD
     This communicator is fundamental to relate the results from
     all sockets */
  world_socket_masters_comm = MPI_COMM_NULL;
  int Im_world_socket_master = ( (my_socket_rank == 0) ? 1 : MPI_UNDEFINED); // The same below, it's just for keeping communicators splitting separated
  MPI_Comm_split(world_comm, Im_world_socket_master, world_rank, &world_socket_masters_comm);
    
  /* Create the communicator which groups the master of each socket
     in each node. Rank 0 in this communicator will be the 
     node master */
  socket_masters_comm = MPI_COMM_NULL;
  int Im_socket_master = ( (my_socket_rank == 0) ? 1 : MPI_UNDEFINED);
  MPI_Comm_split(my_node_comm, Im_socket_master, my_node_rank, &socket_masters_comm );
    
      
  /* Create the communicator which groups the master of each node
     Rank 0 in this communicator will be the master of masters */
  node_masters_comm = MPI_COMM_NULL;
  int Im_node_master = ( (my_node_rank == 0) ? 1 : MPI_UNDEFINED);
  MPI_Comm_split(world_comm, Im_node_master, world_rank, &node_masters_comm );
    
}

void numa::free_communicators()
{
  MPI_Comm_free(&my_node_comm);

  /* OpenMPI vs MPICH case with _MPICH_ compilation flag */
  MPI_Comm_free(&my_socket_comm);
     
  /* Create the communicator which groups the master of each socket
     in the original communicator ex: MPI_COMM_WORLD
     This communicator is fundamental to relate the results from
     all sockets */

  MPI_Comm_free(&world_socket_masters_comm);
    
  /* Create the communicator which groups the master of each socket
     in each node. Rank 0 in this communicator will be the 
     node master */
  MPI_Comm_free(&socket_masters_comm);

  /* Create the communicator which groups the master of each node
     Rank 0 in this communicator will be the master of masters */
  MPI_Comm_free(&node_masters_comm);
}


int numa::get_my_world_rank()          // Get my rank in the original communicator
{
  MPI_Comm_rank(world_comm, &world_rank);
  return world_rank;
}

int numa::get_my_node_rank()          // Get my rank in the intra-node communicator
{
  MPI_Comm_rank(my_node_comm, &my_node_rank);
  return my_node_rank;
}

int numa::get_my_socket_rank()       // Get my rank in the intra-socket communicator
{
  MPI_Comm_rank(my_socket_comm, &my_socket_rank);
  return my_socket_rank;
}

int numa::get_my_socket_masters_rank() // Get my rank in the socket masters' communicator
{
  MPI_Comm_rank(socket_masters_comm, &socket_masters_rank);
  return socket_masters_rank;
}

int numa::get_my_world_socket_masters_rank() // Get my rank in the world socket masters' communicator
{
  MPI_Comm_rank(world_socket_masters_comm, &my_world_socket_masters_rank);
  return my_world_socket_masters_rank;
}
    
int numa::get_my_node_masters_rank() // Get my rank in the node masters' communicator
{
  MPI_Comm_rank(node_masters_comm, &node_masters_rank);
  return node_masters_rank;
}

int numa::get_world_ntasks()         // Get ntasks in the original communicator
{
  MPI_Comm_size(world_comm, &world_ntasks);
  return world_ntasks;
}

int numa::get_node_ntasks()          // Get ntasks in the intra-node communicator
{
  MPI_Comm_size(my_node_comm, &my_node_ntasks);
  return my_node_ntasks;
}

int numa::get_socket_ntasks()        // Get ntasks in the intra-socket communicator
{
  MPI_Comm_size(my_socket_comm, &my_socket_ntasks);
  return my_socket_ntasks;
}

int numa::get_socket_masters_ntasks()  // Get ntasks in the socket masters' communicator
{
  MPI_Comm_size(socket_masters_comm, &socket_masters_ntasks);
  return socket_masters_ntasks;
}

int numa::get_world_socket_masters_ntasks()  // Get ntasks in the world socket masters' communicator
{
  MPI_Comm_size(world_socket_masters_comm, &world_socket_masters_ntasks);
  return world_socket_masters_ntasks;
}
  
  
int numa::get_node_masters_ntasks()  // Get ntasks in the node masters' communicator
{
  MPI_Comm_size(node_masters_comm, &node_masters_ntasks);
  return node_masters_ntasks;
}

  

  
/******************************************************/

std::vector<double> & measures::get_my_socket_maxima(std::string tag) // Get MPI_MAX among all tasks in the socket
{
  double measure[3];
    
  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].seconds, &measure[0], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);
  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].joules, &measure[1], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);
  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].watts, &measure[2], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);

  for (int index=0; index<3; index++)
    my_socket_maxima.push_back(measure[index]);

  my_socket_maxima.at(2) = my_socket_maxima.at(2) / rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].count; //Do not sum the power for more counts of the same tag
    
  return my_socket_maxima;
}

std::vector<double> & measures::get_my_node_summation(std::string tag) // Get MPI_SUM among all sockets in the node or MPI_MAX among all the tasks in the node if nsockets_per_node = 1
{
  double measure[3];
  double nsockets_per_node;

    
  if (my_numa.get_my_socket_rank() == 0)
    nsockets_per_node = my_numa.get_socket_masters_ntasks();
  else
    nsockets_per_node = 0;

  MPI_Bcast(&nsockets_per_node, 1, MPI_INT, 0, my_numa.my_socket_comm); // Broadcast the socket number to all the task in the socket
    
    
  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].seconds, &measure[0], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);
  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].joules, &measure[1], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);
  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].watts, &measure[2], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);
    
  MPI_Barrier(my_numa.my_socket_comm);
    
  if (my_numa.get_my_socket_rank() == 0) // Only the socket masters in the node participate to the summation 
    {
      MPI_Allreduce(MPI_IN_PLACE, &measure[0], 1, MPI_DOUBLE, MPI_MAX, my_numa.socket_masters_comm); // MPI_MAX for time!
      MPI_Allreduce(MPI_IN_PLACE, &measure[1], 1, MPI_DOUBLE, MPI_SUM, my_numa.socket_masters_comm);
      MPI_Allreduce(MPI_IN_PLACE, &measure[2], 1, MPI_DOUBLE, MPI_SUM, my_numa.socket_masters_comm);
    }
  else if (my_numa.get_my_socket_rank() != 0)
    {
      //Do nothing...
    }
  else // Find the MPI_MAX among all the tasks in the node
    {
      MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].seconds, &measure[0], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_node_comm);
      MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].joules, &measure[1], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_node_comm);
      MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].watts, &measure[2], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_node_comm);
    }

  for (int index=0; index<3; index++)
    my_node_summation.push_back(measure[index]);
    
  my_node_summation.at(2) = my_node_summation.at(2) / rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].count; //Do not sum the power for more counts of the same tag
    
  return my_node_summation;
}

std::vector<double> & measures::get_my_node_socket_maxima(std::string tag) // Get MPI_MAX among all tasks in the socket masters' in the node
{
  double measure[3];
  /*
    double nsockets_per_node;
    
    if (my_numa.get_my_socket_rank() == 0)
    nsockets_per_node = my_numa.get_socket_masters_ntasks();
    else
    nsockets_per_node = 0;

    MPI_Bcast(&nsockets_per_node, 1, MPI_INT, 0, my_numa.my_socket_comm); // Broadcast the socket number to all the task in the socket
  */
  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].seconds, &measure[0], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);
  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].joules, &measure[1], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);
  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].watts, &measure[2], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);

  MPI_Barrier(my_numa.my_socket_comm);
    
  if (my_numa.get_my_socket_rank() == 0) // Only the socket masters in the node participate to the MPI_MAX search
    {
      MPI_Allreduce(MPI_IN_PLACE, &measure[0], 1, MPI_DOUBLE, MPI_MAX, my_numa.socket_masters_comm);
      MPI_Allreduce(MPI_IN_PLACE, &measure[1], 1, MPI_DOUBLE, MPI_MAX, my_numa.socket_masters_comm);
      MPI_Allreduce(MPI_IN_PLACE, &measure[2], 1, MPI_DOUBLE, MPI_MAX, my_numa.socket_masters_comm);
    }
    
  for (int index=0; index<3; index++)
    my_node_socket_maxima.push_back(measure[index]);

  my_node_socket_maxima.at(2) = my_node_socket_maxima.at(2) / rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].count; //Do not sum the power for more counts of the same tag
    
  return my_node_socket_maxima;
}

std::vector<double> & measures::get_my_node_socket_minima(std::string tag) // Get MPI_MIN among all tasks in the socket masters' in the node
{
  double measure[3];
    
  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].seconds, &measure[0], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);
  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].joules, &measure[1], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);
  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].watts, &measure[2], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);

  MPI_Barrier(my_numa.my_node_comm);
    
  if (my_numa.get_my_socket_rank() == 0) // Only the socket masters in the node participate to the MPI_MAX search
    {
      MPI_Allreduce(MPI_IN_PLACE, &measure[0], 1, MPI_DOUBLE, MPI_MIN, my_numa.socket_masters_comm);
      MPI_Allreduce(MPI_IN_PLACE, &measure[1], 1, MPI_DOUBLE, MPI_MIN, my_numa.socket_masters_comm);
      MPI_Allreduce(MPI_IN_PLACE, &measure[2], 1, MPI_DOUBLE, MPI_MIN, my_numa.socket_masters_comm);
    }

  for (int index=0; index<3; index++)
    my_node_socket_minima.push_back(measure[index]);

  my_node_socket_minima.at(2) = my_node_socket_minima.at(2) / rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].count; //Do not sum the power for more counts of the same tag
    
  return my_node_socket_minima;
}

std::vector<double> & measures::get_my_node_socket_average(std::string tag) // Get the average among all tasks in the socket masters' in the node
{
  double measure[3];
    
  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].seconds, &measure[0], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);
  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].joules, &measure[1], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);
  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].watts, &measure[2], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);

  MPI_Barrier(my_numa.my_node_comm);
    
  if (my_numa.get_my_socket_rank() == 0) // Only the socket masters in the node participate to the MPI_SUM
    {
      MPI_Allreduce(MPI_IN_PLACE, &measure[0], 1, MPI_DOUBLE, MPI_SUM, my_numa.socket_masters_comm);
      MPI_Allreduce(MPI_IN_PLACE, &measure[1], 1, MPI_DOUBLE, MPI_SUM, my_numa.socket_masters_comm);
      MPI_Allreduce(MPI_IN_PLACE, &measure[2], 1, MPI_DOUBLE, MPI_SUM, my_numa.socket_masters_comm);

      for (int index=0; index<3; index++)
	measure[index] = measure[index] / my_numa.get_socket_masters_ntasks();
    }
     
  for (int index=0; index<3; index++)
    my_node_socket_average.push_back(measure[index]);

  my_node_socket_average.at(2) = my_node_socket_average.at(2) / rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].count; //Do not sum the power for more counts of the same tag
    
  return my_node_socket_average;
}

std::vector<double> & measures::get_my_node_socket_std(std::string tag) // Get the standard deviation among all tasks in the socket masters' in the node
{
  double measure[3];
  double dev[3]; // Array of deviations in time, energy and power, respectively
    
  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].seconds, &measure[0], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);
  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].joules, &measure[1], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);
  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].watts, &measure[2], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);

  MPI_Barrier(my_numa.my_node_comm);
    
  if (my_numa.get_my_socket_rank() == 0) // Only the socket masters in the node participate to the MPI_SUM 
    {
      MPI_Allreduce(MPI_IN_PLACE, &measure[0], 1, MPI_DOUBLE, MPI_SUM, my_numa.socket_masters_comm);
      MPI_Allreduce(MPI_IN_PLACE, &measure[1], 1, MPI_DOUBLE, MPI_SUM, my_numa.socket_masters_comm);
      MPI_Allreduce(MPI_IN_PLACE, &measure[2], 1, MPI_DOUBLE, MPI_SUM, my_numa.socket_masters_comm);

      for (int index=0; index<3; index++)
	measure[index] = measure[index] / my_numa.get_socket_masters_ntasks();

      // Compute deviations of single sockets from average
      double my_time_dev = (rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].seconds - measure[0]) * (rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].seconds - measure[0]);
      double my_energy_dev = (rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].joules - measure[1]) * (rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].joules - measure[1]);
      double my_power_dev = (rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].watts - measure[2]) * (rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].watts - measure[2]);

      // Sum all deviations
      MPI_Allreduce(&my_time_dev, &dev[0], 1, MPI_DOUBLE, MPI_SUM, my_numa.socket_masters_comm);
      MPI_Allreduce(&my_energy_dev, &dev[1], 1, MPI_DOUBLE, MPI_SUM, my_numa.socket_masters_comm);
      MPI_Allreduce(&my_power_dev, &dev[2], 1, MPI_DOUBLE, MPI_SUM, my_numa.socket_masters_comm);

      // Get standard deviations
      for (int index=0; index<3; index++)
	dev[index] = sqrt(dev[index] / my_numa.get_socket_masters_ntasks());
    }
    
  for (int index=0; index<3; index++)
    my_node_socket_std.push_back(dev[index]);

  my_node_socket_std.at(2) = my_node_socket_std.at(2) / rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].count; //Do not sum the power for more counts of the same tag
    
  return my_node_socket_std;
}

std::vector<double> & measures::get_my_world_socket_maxima(std::string tag) // Get MPI_MAX among all tasks in the socket masters' in the world comm
{
  double measure[3];

  // Compute the maxima per socket
  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].seconds, &measure[0], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);
  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].joules, &measure[1], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);
  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].watts, &measure[2], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);

  MPI_Barrier(my_numa.my_socket_comm);
    
    
  if (my_numa.get_my_socket_rank() == 0) // Only the socket masters in the world participate to the MPI_MAX search
    {
      MPI_Allreduce(MPI_IN_PLACE, &measure[0], 1, MPI_DOUBLE, MPI_MAX, my_numa.world_socket_masters_comm);
      MPI_Allreduce(MPI_IN_PLACE, &measure[1], 1, MPI_DOUBLE, MPI_MAX, my_numa.world_socket_masters_comm);
      MPI_Allreduce(MPI_IN_PLACE, &measure[2], 1, MPI_DOUBLE, MPI_MAX, my_numa.world_socket_masters_comm);
    }
  else
    {
      // The other tasks are simply waiting...
    }
    
  for (int index=0; index<3; index++)
    my_world_socket_maxima.push_back(measure[index]);

  my_world_socket_maxima.at(2) = my_world_socket_maxima.at(2) / rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].count; //Do not sum the power for more counts of the same tag
    
  return my_world_socket_maxima;
}

std::vector<double> & measures::get_my_world_socket_minima(std::string tag) // Get MPI_MIN among all tasks in the socket masters' in the world comm
{
  double measure[3];

    
  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].seconds, &measure[0], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);
  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].joules, &measure[1], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);
  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].watts, &measure[2], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);

  MPI_Barrier(my_numa.my_socket_comm);
    

  if (my_numa.get_my_socket_rank() == 0) // Only the socket masters in the world participate to the MPI_MIN search
    {
      MPI_Allreduce(MPI_IN_PLACE, &measure[0], 1, MPI_DOUBLE, MPI_MIN, my_numa.world_socket_masters_comm);
      MPI_Allreduce(MPI_IN_PLACE, &measure[1], 1, MPI_DOUBLE, MPI_MIN, my_numa.world_socket_masters_comm);
      MPI_Allreduce(MPI_IN_PLACE, &measure[2], 1, MPI_DOUBLE, MPI_MIN, my_numa.world_socket_masters_comm);
    }
  else
    {
      // The other tasks are simply waiting...
    }
    
  for (int index=0; index<3; index++)
    my_world_socket_minima.push_back(measure[index]);

  my_world_socket_minima.at(2) = my_world_socket_minima.at(2) / rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].count; //Do not sum the power for more counts of the same tag
    
  return my_world_socket_minima;
}

std::vector<double> & measures::get_my_world_socket_average(std::string tag) // Get the average among all tasks in the socket masters' in the world comm
{
  double measure[3];
    
  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].seconds, &measure[0], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);
  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].joules, &measure[1], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);
  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].watts, &measure[2], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);

  MPI_Barrier(my_numa.my_socket_comm);
        

  if (my_numa.get_my_socket_rank() == 0) // Only the socket masters in the world participate to the MPI_SUM
    {
      MPI_Allreduce(MPI_IN_PLACE, &measure[0], 1, MPI_DOUBLE, MPI_SUM, my_numa.world_socket_masters_comm);
      MPI_Allreduce(MPI_IN_PLACE, &measure[1], 1, MPI_DOUBLE, MPI_SUM, my_numa.world_socket_masters_comm);
      MPI_Allreduce(MPI_IN_PLACE, &measure[2], 1, MPI_DOUBLE, MPI_SUM, my_numa.world_socket_masters_comm);

      for (int index=0; index<3; index++)
	measure[index] = measure[index] / my_numa.get_world_socket_masters_ntasks();
    }
  else
    {
      // The other tasks are simply waiting...
    }
    
  for (int index=0; index<3; index++)
    my_world_socket_average.push_back(measure[index]);

  my_world_socket_average.at(2) = my_world_socket_average.at(2) / rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].count; //Do not sum the power for more counts of the same tag
    
  return my_world_socket_average;
}

std::vector<double> & measures::get_my_world_socket_std(std::string tag) // Get the standard deviation among all tasks in the socket masters' in the world comm
{
  double measure[3];
  double dev[3]; // Array of deviations in time, energy and power, respectively

  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].seconds, &measure[0], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);
  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].joules, &measure[1], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);
  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].watts, &measure[2], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);

  MPI_Barrier(my_numa.my_socket_comm);
        
  if (my_numa.get_my_socket_rank() == 0) // Only the socket masters in the world participate to the MPI_SUM 
    {
      MPI_Allreduce(MPI_IN_PLACE, &measure[0], 1, MPI_DOUBLE, MPI_SUM, my_numa.world_socket_masters_comm);
      MPI_Allreduce(MPI_IN_PLACE, &measure[1], 1, MPI_DOUBLE, MPI_SUM, my_numa.world_socket_masters_comm);
      MPI_Allreduce(MPI_IN_PLACE, &measure[2], 1, MPI_DOUBLE, MPI_SUM, my_numa.world_socket_masters_comm);

      for (int index=0; index<3; index++)
	measure[index] = measure[index] / my_numa.get_world_socket_masters_ntasks();

      // Compute deviations of single sockets from average
      double my_time_dev = (rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].seconds - measure[0]) * (rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].seconds - measure[0]);
      double my_energy_dev = (rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].joules - measure[1]) * (rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].joules - measure[1]);
      double my_power_dev = (rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].watts - measure[2]) * (rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].watts - measure[2]);

      // Sum all deviations
      MPI_Allreduce(&my_time_dev, &dev[0], 1, MPI_DOUBLE, MPI_SUM, my_numa.world_socket_masters_comm);
      MPI_Allreduce(&my_energy_dev, &dev[1], 1, MPI_DOUBLE, MPI_SUM, my_numa.world_socket_masters_comm);
      MPI_Allreduce(&my_power_dev, &dev[2], 1, MPI_DOUBLE, MPI_SUM, my_numa.world_socket_masters_comm);

      // Get standard deviations
      for (int index=0; index<3; index++)
	dev[index] = sqrt(dev[index] / my_numa.get_world_socket_masters_ntasks());
    }
  else
    {
      // The other tasks are simply waiting...
    }
    
  for (int index=0; index<3; index++)
    my_world_socket_std.push_back(dev[index]);

  my_world_socket_std.at(2) = my_world_socket_std.at(2) / rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].count; //Do not sum the power for more counts of the same tag
    
  return my_world_socket_std;
}

std::vector<double> & measures::get_my_world_node_maxima(std::string tag) // Get MPI_MAX among all tasks in the node masters' in the world comm
{
  double measure[3];

  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].seconds, &measure[0], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);
  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].joules, &measure[1], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);
  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].watts, &measure[2], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);
    
  MPI_Barrier(my_numa.my_socket_comm);
    
  if (my_numa.get_my_socket_rank() == 0) // Only the socket masters in the node participate to the summation 
    {
      MPI_Allreduce(MPI_IN_PLACE, &measure[0], 1, MPI_DOUBLE, MPI_MAX, my_numa.socket_masters_comm); // MPI_MAX for time!
      MPI_Allreduce(MPI_IN_PLACE, &measure[1], 1, MPI_DOUBLE, MPI_SUM, my_numa.socket_masters_comm);
      MPI_Allreduce(MPI_IN_PLACE, &measure[2], 1, MPI_DOUBLE, MPI_SUM, my_numa.socket_masters_comm);
    }
  else if (my_numa.get_my_socket_rank() != 0)
    {
      //Do nothing...
    }
  else
    {
      // Do nothing...
    }
    
        
  if (my_numa.get_my_node_rank() == 0) // Only the node masters in the world participate to the MPI_MAX search
    {
      MPI_Allreduce(MPI_IN_PLACE, &measure[0], 1, MPI_DOUBLE, MPI_MAX, my_numa.node_masters_comm);
      MPI_Allreduce(MPI_IN_PLACE, &measure[1], 1, MPI_DOUBLE, MPI_MAX, my_numa.node_masters_comm);
      MPI_Allreduce(MPI_IN_PLACE, &measure[2], 1, MPI_DOUBLE, MPI_MAX, my_numa.node_masters_comm);
    }
  else if (my_numa.get_my_node_rank() != 0)
    {
      // Do nothing...
    }
  else
    {
      // The other tasks are simply waiting...
    }
    
  for (int index=0; index<3; index++)
    my_world_node_maxima.push_back(measure[index]);

  my_world_node_maxima.at(2) = my_world_node_maxima.at(2) / rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].count; //Do not sum the power for more counts of the same tag
    
  return my_world_node_maxima;
}

std::vector<double> & measures::get_my_world_node_minima(std::string tag) // Get MPI_MIN among all tasks in the node masters' in the world comm
{
  double measure[3];

  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].seconds, &measure[0], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);
  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].joules, &measure[1], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);
  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].watts, &measure[2], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);
    
  MPI_Barrier(my_numa.my_socket_comm);
    
  if (my_numa.get_my_socket_rank() == 0) // Only the socket masters in the node participate to the summation 
    {
      MPI_Allreduce(MPI_IN_PLACE, &measure[0], 1, MPI_DOUBLE, MPI_MAX, my_numa.socket_masters_comm); // MPI_MAX for time!
      MPI_Allreduce(MPI_IN_PLACE, &measure[1], 1, MPI_DOUBLE, MPI_SUM, my_numa.socket_masters_comm);
      MPI_Allreduce(MPI_IN_PLACE, &measure[2], 1, MPI_DOUBLE, MPI_SUM, my_numa.socket_masters_comm);
    }
  else if (my_numa.get_my_socket_rank() != 0)
    {
      //Do nothing...
    }
  else
    {
      // Do nothing...
    }
    
  if (my_numa.get_my_node_rank() == 0) // Only the node masters in the world participate to the MPI_MIN search
    {
      MPI_Allreduce(MPI_IN_PLACE, &measure[0], 1, MPI_DOUBLE, MPI_MIN, my_numa.node_masters_comm);
      MPI_Allreduce(MPI_IN_PLACE, &measure[1], 1, MPI_DOUBLE, MPI_MIN, my_numa.node_masters_comm);
      MPI_Allreduce(MPI_IN_PLACE, &measure[2], 1, MPI_DOUBLE, MPI_MIN, my_numa.node_masters_comm);
    }
  else if (my_numa.get_my_node_rank() != 0)
    {
      //Do nothing...
    }    
  else
    {
      // The other tasks are simply waiting...
    }
    
  for (int index=0; index<3; index++)
    my_world_node_minima.push_back(measure[index]);

  my_world_node_minima.at(2) = my_world_node_minima.at(2) / rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].count; //Do not sum the power for more counts of the same tag
    
  return my_world_node_minima;
}

std::vector<double> & measures::get_my_world_node_average(std::string tag) // Get the average among all tasks in the node masters' in the world comm
{
  double measure[3];

  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].seconds, &measure[0], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);
  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].joules, &measure[1], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);
  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].watts, &measure[2], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);
    
  MPI_Barrier(my_numa.my_socket_comm);
    
  if (my_numa.get_my_socket_rank() == 0) // Only the socket masters in the node participate to the summation 
    {
      MPI_Allreduce(MPI_IN_PLACE, &measure[0], 1, MPI_DOUBLE, MPI_MAX, my_numa.socket_masters_comm); // MPI_MAX for time!
      MPI_Allreduce(MPI_IN_PLACE, &measure[1], 1, MPI_DOUBLE, MPI_SUM, my_numa.socket_masters_comm);
      MPI_Allreduce(MPI_IN_PLACE, &measure[2], 1, MPI_DOUBLE, MPI_SUM, my_numa.socket_masters_comm);
    }
  else if (my_numa.get_my_socket_rank() != 0)
    {
      //Do nothing...
    }
  else
    {
      // Do nothing...
    }
    
  if (my_numa.get_my_node_rank() == 0) // Only the node masters in the world participate to the MPI_SUM
    {
      MPI_Allreduce(MPI_IN_PLACE, &measure[0], 1, MPI_DOUBLE, MPI_SUM, my_numa.node_masters_comm);
      MPI_Allreduce(MPI_IN_PLACE, &measure[1], 1, MPI_DOUBLE, MPI_SUM, my_numa.node_masters_comm);
      MPI_Allreduce(MPI_IN_PLACE, &measure[2], 1, MPI_DOUBLE, MPI_SUM, my_numa.node_masters_comm);

      for (int index=0; index<3; index++)
	measure[index] = measure[index] / my_numa.get_node_masters_ntasks();
    }
  else if (my_numa.get_my_node_rank() != 0)
    {
      //Do nothing...
    }
  else
    {
      // The other tasks are simply waiting...
    }
    
  for (int index=0; index<3; index++)
    my_world_node_average.push_back(measure[index]);

  my_world_node_average.at(2) = my_world_node_average.at(2) / rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].count; //Do not sum the power for more counts of the same tag
    
  return my_world_node_average;
}

std::vector<double> & measures::get_my_world_node_std(std::string tag) // Get the standard deviation among all tasks in the node masters' in the world comm
{
  double measure[3];
  double node_measure[3];
  double dev[3]; // Array of deviations in time, energy and power, respectively

  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].seconds, &measure[0], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);
  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].joules, &measure[1], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);
  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].watts, &measure[2], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);
    
  MPI_Barrier(my_numa.my_socket_comm);
    
  if (my_numa.get_my_socket_rank() == 0) // Only the socket masters in the node participate to the summation 
    {
      // This step is required for the correct standard deviation calculation
      MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].seconds, &node_measure[0], 1, MPI_DOUBLE, MPI_MAX, my_numa.socket_masters_comm);
      MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].joules, &node_measure[1], 1, MPI_DOUBLE, MPI_SUM, my_numa.socket_masters_comm);
      MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].watts, &node_measure[2], 1, MPI_DOUBLE, MPI_SUM, my_numa.socket_masters_comm);
	
      MPI_Allreduce(MPI_IN_PLACE, &measure[0], 1, MPI_DOUBLE, MPI_MAX, my_numa.socket_masters_comm); // MPI_MAX for time!
      MPI_Allreduce(MPI_IN_PLACE, &measure[1], 1, MPI_DOUBLE, MPI_SUM, my_numa.socket_masters_comm);
      MPI_Allreduce(MPI_IN_PLACE, &measure[2], 1, MPI_DOUBLE, MPI_SUM, my_numa.socket_masters_comm);
    }
  else if (my_numa.get_my_socket_rank() != 0)
    {
      //Do nothing...
    }
  else
    {
      // Do nothing...
    }

    
  if (my_numa.get_my_node_rank() == 0) // Only the node masters in the world participate to the MPI_SUM 
    {
	
      MPI_Allreduce(MPI_IN_PLACE, &measure[0], 1, MPI_DOUBLE, MPI_SUM, my_numa.node_masters_comm);
      MPI_Allreduce(MPI_IN_PLACE, &measure[1], 1, MPI_DOUBLE, MPI_SUM, my_numa.node_masters_comm);
      MPI_Allreduce(MPI_IN_PLACE, &measure[2], 1, MPI_DOUBLE, MPI_SUM, my_numa.node_masters_comm);
	
      for (int index=0; index<3; index++)
	measure[index] = measure[index] / my_numa.get_node_masters_ntasks();

      // Compute deviations of single nodes from average
      double my_time_dev = (node_measure[0] - measure[0]) * (node_measure[0] - measure[0]);
      double my_energy_dev = (node_measure[1] - measure[1]) * (node_measure[1] - measure[1]);
      double my_power_dev = (node_measure[2] - measure[2]) * (node_measure[2] - measure[2]);

      /*
      // Compute deviations of single nodes from average
      double my_time_dev = (rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].seconds - measure[0]) * (rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].seconds - measure[0]);
      double my_energy_dev = (rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].joules - measure[1]) * (rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].joules - measure[1]);
      double my_power_dev = (rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].watts - measure[2]) * (rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].watts - measure[2]);
      */
	
	
      // Sum all deviations
      MPI_Allreduce(&my_time_dev, &dev[0], 1, MPI_DOUBLE, MPI_SUM, my_numa.node_masters_comm);
      MPI_Allreduce(&my_energy_dev, &dev[1], 1, MPI_DOUBLE, MPI_SUM, my_numa.node_masters_comm);
      MPI_Allreduce(&my_power_dev, &dev[2], 1, MPI_DOUBLE, MPI_SUM, my_numa.node_masters_comm);
	
	
      // Get standard deviations
      for (int index=0; index<3; index++)
	dev[index] = sqrt(dev[index] / my_numa.get_node_masters_ntasks());
    }
  else if (my_numa.get_my_node_rank() != 0)
    {
      //Do nothing...
    }
  else
    {
      // The other tasks are simply waiting...
    }
    
  for (int index=0; index<3; index++)
    my_world_node_std.push_back(dev[index]);

  my_world_node_std.at(2) = my_world_node_std.at(2) / rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].count; //Do not sum the power for more counts of the same tag
    
  return my_world_node_std;
}

std::vector<double> & measures::get_my_world_summation(std::string tag) // Get MPI_SUM among all sockets in the world or MPI_MAX among all the tasks in the world if nsockets_per_world = 1
{
  double measure[3];

  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].seconds, &measure[0], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);
  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].joules, &measure[1], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);
  MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].watts, &measure[2], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_socket_comm);
    
  MPI_Barrier(my_numa.my_socket_comm);
    
  if (my_numa.get_my_socket_rank() == 0) // Only the socket masters in the node participate to the summation 
    {
      MPI_Allreduce(MPI_IN_PLACE, &measure[0], 1, MPI_DOUBLE, MPI_MAX, my_numa.socket_masters_comm); // MPI_MAX for time!
      MPI_Allreduce(MPI_IN_PLACE, &measure[1], 1, MPI_DOUBLE, MPI_SUM, my_numa.socket_masters_comm);
      MPI_Allreduce(MPI_IN_PLACE, &measure[2], 1, MPI_DOUBLE, MPI_SUM, my_numa.socket_masters_comm);
    }
  else if (my_numa.get_my_socket_rank() != 0)
    {
      //Do nothing...
    }
  else
    {
      // Do nothing...
    }
    
  if (my_numa.get_my_node_rank() == 0) // Only the socket masters in the world participate to the summation 
    {
      MPI_Allreduce(MPI_IN_PLACE, &measure[0], 1, MPI_DOUBLE, MPI_MAX, my_numa.node_masters_comm); // MPI_MAX for time!
      MPI_Allreduce(MPI_IN_PLACE, &measure[1], 1, MPI_DOUBLE, MPI_SUM, my_numa.node_masters_comm);
      MPI_Allreduce(MPI_IN_PLACE, &measure[2], 1, MPI_DOUBLE, MPI_SUM, my_numa.node_masters_comm);
	
    }
  else if (my_numa.get_my_node_rank() != 0)
    {
      //Do nothing...
    }
  else // Find the MPI_MAX among all the tasks in the node
    {
      MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].seconds, &measure[0], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_node_comm);
      MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].joules, &measure[1], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_node_comm);
      MPI_Allreduce(&rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].watts, &measure[2], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_node_comm);
    }
    
  for (int index=0; index<3; index++)
    my_world_summation.push_back(measure[index]);
    
  my_world_summation.at(2) = my_world_summation.at(2) / rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].count; //Do not sum the power for more counts of the same tag
    
  return my_world_summation;
}
  

/* ######################################################################### 



   CPU FUNCTIONS ARE OVER
   NOW GPU FUNCTIONS ARE BEING WRITTEN



   ######################################################################### */
  
#if defined(_NVIDIA_) || defined(_AMD_) // FUNCTIONS FOR GPUS SAME ALSO FOR SETONIX

std::vector<double> & gpus::get_my_node_gpu_maxima(std::string tag, const int devID) // Get MPI_MAX among all GPUs in the node
{
  double measure[3];
    
  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].seconds, &measure[0], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_node_comm);
  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].joules, &measure[1], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_node_comm);
  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].watts, &measure[2], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_node_comm);

  for (int index=0; index<3; index++)
    my_node_gpu_maxima.push_back(measure[index]);

  my_node_gpu_maxima.at(2) = my_node_gpu_maxima.at(2) / gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].count; //Do not sum the power for more counts of the same tag
    
  return my_node_gpu_maxima;
}

  
std::vector<double> & gpus::get_my_node_gpu_minima(std::string tag, const int devID) // Get MPI_MIN among all GPUs in the node
{
  double measure[3];

 #if defined(_SETONIX_)
    
  if (my_numa.get_my_node_rank() % 2 == 0) // Only even ranks read non-negligible GPU values for energy and power
    {
      MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].seconds, &measure[0], 1, MPI_DOUBLE, MPI_MIN, my_numa.my_node_comm);
      MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].joules, &measure[1], 1, MPI_DOUBLE, MPI_MIN, my_numa.my_node_comm);
      MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].watts, &measure[2], 1, MPI_DOUBLE, MPI_MIN, my_numa.my_node_comm);
    }
  else if (my_numa.get_my_node_rank() % 2 != 0)
    {
      // Do nothing...
    }
  else
    {
      // The other tasks are simply waiting...
    }
    
 #else
    
  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].seconds, &measure[0], 1, MPI_DOUBLE, MPI_MIN, my_numa.my_node_comm);
  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].joules, &measure[1], 1, MPI_DOUBLE, MPI_MIN, my_numa.my_node_comm);
  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].watts, &measure[2], 1, MPI_DOUBLE, MPI_MIN, my_numa.my_node_comm);

 #endif
    
  for (int index=0; index<3; index++)
    my_node_gpu_minima.push_back(measure[index]);
    
  my_node_gpu_minima.at(2) = my_node_gpu_minima.at(2) / gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].count; //Do not sum the power for more counts of the same tag
    
  return my_node_gpu_minima;
}

std::vector<double> & gpus::get_my_node_gpu_average(std::string tag, const int devID) // Get the average among all GPUs in the node
{
  double measure[3];
    
  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].seconds, &measure[0], 1, MPI_DOUBLE, MPI_SUM, my_numa.my_node_comm);
  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].joules, &measure[1], 1, MPI_DOUBLE, MPI_SUM, my_numa.my_node_comm);
  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].watts, &measure[2], 1, MPI_DOUBLE, MPI_SUM, my_numa.my_node_comm);

 #if defined(_SETONIX_) // Be careful on when you have odd number of tasks
  int relevant_ntasks = my_numa.get_node_ntasks() % 2 == 0 ? my_numa.get_node_ntasks() / 2 : (my_numa.get_node_ntasks() / 2) + 1;
  for (int index=0; index<3; index++)
    {
      measure[index] = measure[index] / relevant_ntasks; 
      my_node_gpu_average.push_back(measure[index]);
    }
 #else
  for (int index=0; index<3; index++)
    {
      measure[index] = measure[index] / my_numa.get_node_ntasks();
      my_node_gpu_average.push_back(measure[index]);
    } 
 #endif // defined(_SETONIX_)
    
  my_node_gpu_average.at(2) = my_node_gpu_average.at(2) / gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].count; //Do not sum the power for more counts of the same tag
    
  return my_node_gpu_average;
}

std::vector<double> & gpus::get_my_node_gpu_std(std::string tag, const int devID) // Get the standard deviation among all GPUs in the node
{
  double measure[3];
  double dev[3]; // Array of deviations in time, energy and power, respectively
        

  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].seconds, &measure[0], 1, MPI_DOUBLE, MPI_SUM, my_numa.my_node_comm);
  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].joules, &measure[1], 1, MPI_DOUBLE, MPI_SUM, my_numa.my_node_comm);
  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].watts, &measure[2], 1, MPI_DOUBLE, MPI_SUM, my_numa.my_node_comm);

 #if defined(_SETONIX_) // Be careful on when you have odd number of tasks
  int relevant_ntasks = my_numa.get_node_ntasks() % 2 == 0 ? my_numa.get_node_ntasks() / 2 : (my_numa.get_node_ntasks() / 2) + 1;
  for (int index=0; index<3; index++)
    {
      measure[index] = measure[index] / relevant_ntasks; 
      my_node_gpu_average.push_back(measure[index]);
    }

  if (my_numa.get_my_node_rank() % 2 == 0)
    {
      // Compute deviations of single sockets from average
      double my_time_dev = (gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].seconds - measure[0]) * (gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].seconds - measure[0]);
      double my_energy_dev = (gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].joules - measure[1]) * (gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].joules - measure[1]);
      double my_power_dev = (gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].watts - measure[2]) * (gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].watts - measure[2]);

      // Sum all deviations
      MPI_Allreduce(&my_time_dev, &dev[0], 1, MPI_DOUBLE, MPI_SUM, my_numa.my_node_comm);
      MPI_Allreduce(&my_energy_dev, &dev[1], 1, MPI_DOUBLE, MPI_SUM, my_numa.my_node_comm);
      MPI_Allreduce(&my_power_dev, &dev[2], 1, MPI_DOUBLE, MPI_SUM, my_numa.my_node_comm);
    }
  else if (my_numa.get_my_node_rank() % 2 != 0)
    {
      // Do nothing...
    }
  else
    {
      // The other tasks are simply waiting...
    }
    
    
  // Get standard deviations
  for (int index=0; index<3; index++)
    {
      dev[index] = sqrt(dev[index] / relevant_ntasks);
      my_node_gpu_std.push_back(dev[index]);
    }
        
 #else
  for (int index=0; index<3; index++)
    measure[index] = measure[index] / my_numa.get_node_ntasks();
       
  // Compute deviations of single sockets from average
  double my_time_dev = (gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].seconds - measure[0]) * (gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].seconds - measure[0]);
  double my_energy_dev = (gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].joules - measure[1]) * (gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].joules - measure[1]);
  double my_power_dev = (gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].watts - measure[2]) * (gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].watts - measure[2]);

  // Sum all deviations
  MPI_Allreduce(&my_time_dev, &dev[0], 1, MPI_DOUBLE, MPI_SUM, my_numa.my_node_comm);
  MPI_Allreduce(&my_energy_dev, &dev[1], 1, MPI_DOUBLE, MPI_SUM, my_numa.my_node_comm);
  MPI_Allreduce(&my_power_dev, &dev[2], 1, MPI_DOUBLE, MPI_SUM, my_numa.my_node_comm);

  // Get standard deviations
  for (int index=0; index<3; index++)
    {
      dev[index] = sqrt(dev[index] / my_numa.get_node_ntasks());
      my_node_gpu_std.push_back(dev[index]);
    }
 #endif // defined(_SETONIX_)
    
  my_node_gpu_std.at(2) = my_node_gpu_std.at(2) / gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].count; //Do not sum the power for more counts of the same tag
    
  return my_node_gpu_std;
}

std::vector<double> & gpus::get_my_world_gpu_maxima(std::string tag, const int devID) // Get MPI_MAX among all GPUs in the world comm
{
  double measure[3];
    
  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].seconds, &measure[0], 1, MPI_DOUBLE, MPI_MAX, my_numa.world_comm);
  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].joules, &measure[1], 1, MPI_DOUBLE, MPI_MAX, my_numa.world_comm);
  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].watts, &measure[2], 1, MPI_DOUBLE, MPI_MAX, my_numa.world_comm);

  for (int index=0; index<3; index++)
    my_world_gpu_maxima.push_back(measure[index]);

  my_world_gpu_maxima.at(2) = my_world_gpu_maxima.at(2) / gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].count; //Do not sum the power for more counts of the same tag
    
  return my_world_gpu_maxima;
}

std::vector<double> & gpus::get_my_world_gpu_minima(std::string tag, const int devID) // Get MPI_MIN among all GPUs in the world comm
{
  double measure[3];

 #if defined(_SETONIX_)
    
  if (my_numa.get_my_world_rank() % 2 == 0) // Only even ranks read non-negligible GPU values for energy and power
    {
      MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].seconds, &measure[0], 1, MPI_DOUBLE, MPI_MIN, my_numa.world_comm);
      MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].joules, &measure[1], 1, MPI_DOUBLE, MPI_MIN, my_numa.world_comm);
      MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].watts, &measure[2], 1, MPI_DOUBLE, MPI_MIN, my_numa.world_comm);
    }
  else if (my_numa.get_my_world_rank() % 2 != 0)
    {
      // Do nothing...
    }
  else
    {
      // The other tasks are simply waiting...
    }
    
    
 #else

  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].seconds, &measure[0], 1, MPI_DOUBLE, MPI_MIN, my_numa.world_comm);
  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].joules, &measure[1], 1, MPI_DOUBLE, MPI_MIN, my_numa.world_comm);
  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].watts, &measure[2], 1, MPI_DOUBLE, MPI_MIN, my_numa.world_comm);

 #endif // defined(_SETONIX_)
    
  for (int index=0; index<3; index++)
    my_world_gpu_minima.push_back(measure[index]);
    
  my_world_gpu_minima.at(2) = my_world_gpu_minima.at(2) / gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].count; //Do not sum the power for more counts of the same tag
    
  return my_world_gpu_minima;
}

std::vector<double> & gpus::get_my_world_gpu_average(std::string tag, const int devID) // Get the average among all GPUs in the world comm
{
  double measure[3];
    
  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].seconds, &measure[0], 1, MPI_DOUBLE, MPI_SUM, my_numa.world_comm);
  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].joules, &measure[1], 1, MPI_DOUBLE, MPI_SUM, my_numa.world_comm);
  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].watts, &measure[2], 1, MPI_DOUBLE, MPI_SUM, my_numa.world_comm);

 #if defined(_SETONIX_) // Be careful on when you have odd number of tasks

  int relevant_ntasks = my_numa.get_world_ntasks() % 2 == 0 ? my_numa.get_world_ntasks() / 2 : (my_numa.get_world_ntasks() / 2) + 1;
  for (int index=0; index<3; index++)
    {
      measure[index] = measure[index] / relevant_ntasks; 
      my_node_gpu_average.push_back(measure[index]);
    }
 #else
       
  for (int index=0; index<3; index++)
    {
      measure[index] = measure[index] / my_numa.get_world_ntasks(); 
      my_world_gpu_average.push_back(measure[index]);
    }

 #endif // defined(_SETONIX_)
    
  my_world_gpu_average.at(2) = my_world_gpu_average.at(2) / gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].count; //Do not sum the power for more counts of the same tag
    
  return my_world_gpu_average;
}

std::vector<double> & gpus::get_my_world_gpu_std(std::string tag, const int devID) // Get the standard deviation among all GPUs in the world comm
{
  double measure[3];
  double dev[3]; // Array of deviations in time, energy and power, respectively
        

  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].seconds, &measure[0], 1, MPI_DOUBLE, MPI_SUM, my_numa.world_comm);
  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].joules, &measure[1], 1, MPI_DOUBLE, MPI_SUM, my_numa.world_comm);
  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].watts, &measure[2], 1, MPI_DOUBLE, MPI_SUM, my_numa.world_comm);

 #if defined(_SETONIX_) // Be careful on when you have odd number of tasks

  int relevant_ntasks = my_numa.get_world_ntasks() % 2 == 0 ? my_numa.get_world_ntasks() / 2 : (my_numa.get_world_ntasks() / 2) + 1;
  for (int index=0; index<3; index++)
    {
      measure[index] = measure[index] / relevant_ntasks; 
      my_node_gpu_average.push_back(measure[index]);
    }

  if (my_numa.get_my_world_rank() % 2 == 0)
    {
      // Compute deviations of single sockets from average
      double my_time_dev = (gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].seconds - measure[0]) * (gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].seconds - measure[0]);
      double my_energy_dev = (gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].joules - measure[1]) * (gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].joules - measure[1]);
      double my_power_dev = (gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].watts - measure[2]) * (gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].watts - measure[2]);

      // Sum all deviations
      MPI_Allreduce(&my_time_dev, &dev[0], 1, MPI_DOUBLE, MPI_SUM, my_numa.world_comm);
      MPI_Allreduce(&my_energy_dev, &dev[1], 1, MPI_DOUBLE, MPI_SUM, my_numa.world_comm);
      MPI_Allreduce(&my_power_dev, &dev[2], 1, MPI_DOUBLE, MPI_SUM, my_numa.world_comm);
    }
  else if (my_numa.get_my_world_rank() % 2 != 0)
    {
      // Do nothing...
    }
  else
    {
      // The other tasks are simply waiting...
    }
    
    
  // Get standard deviations
  for (int index=0; index<3; index++)
    {
      dev[index] = sqrt(dev[index] / relevant_ntasks);
      my_node_gpu_std.push_back(dev[index]);
    }
        
 #else
       
  for (int index=0; index<3; index++)
    measure[index] = measure[index] / my_numa.get_world_ntasks();

  // Compute deviations of single sockets from average
  double my_time_dev = (gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].seconds - measure[0]) * (gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].seconds - measure[0]);
  double my_energy_dev = (gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].joules - measure[1]) * (gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].joules - measure[1]);
  double my_power_dev = (gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].watts - measure[2]) * (gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].watts - measure[2]);

  // Sum all deviations
  MPI_Allreduce(&my_time_dev, &dev[0], 1, MPI_DOUBLE, MPI_SUM, my_numa.world_comm);
  MPI_Allreduce(&my_energy_dev, &dev[1], 1, MPI_DOUBLE, MPI_SUM, my_numa.world_comm);
  MPI_Allreduce(&my_power_dev, &dev[2], 1, MPI_DOUBLE, MPI_SUM, my_numa.world_comm);

  // Get standard deviations
  for (int index=0; index<3; index++)
    {
      dev[index] = sqrt(dev[index] / my_numa.get_world_ntasks());
      my_world_gpu_std.push_back(dev[index]);
    }

 #endif // defined(_SETONIX_)
    
  my_world_gpu_std.at(2) = my_world_gpu_std.at(2) / gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].count; //Do not sum the power for more counts of the same tag
    
  return my_world_gpu_std;
}

std::vector<double> & gpus::get_my_node_gpu_summation(std::string tag, const int devID) // Get the summation over all GPUs in the node
{
  double measure[3];
    
  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].seconds, &measure[0], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_node_comm); // MPI_MAX for time
  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].joules, &measure[1], 1, MPI_DOUBLE, MPI_SUM, my_numa.my_node_comm);
  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].watts, &measure[2], 1, MPI_DOUBLE, MPI_SUM, my_numa.my_node_comm);

  for (int index=0; index<3; index++)
    my_node_gpu_summation.push_back(measure[index]);

  my_node_gpu_summation.at(2) = my_node_gpu_summation.at(2) / gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].count; //Do not sum the power for more counts of the same tag
    
  return my_node_gpu_summation;
}

std::vector<double> & gpus::get_my_world_gpu_summation(std::string tag, const int devID) // Get the summation over all GPUs in the world comm
{
  double measure[3];
    
  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].seconds, &measure[0], 1, MPI_DOUBLE, MPI_MAX, my_numa.world_comm); // MPI_MAX for time
  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].joules, &measure[1], 1, MPI_DOUBLE, MPI_SUM, my_numa.world_comm);
  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].watts, &measure[2], 1, MPI_DOUBLE, MPI_SUM, my_numa.world_comm);

  for (int index=0; index<3; index++)
    my_world_gpu_summation.push_back(measure[index]);

  my_world_gpu_summation.at(2) = my_world_gpu_summation.at(2) / gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].count; //Do not sum the power for more counts of the same tag
    
  return my_world_gpu_summation;
}

std::vector<double> & gpus::get_my_node_world_maxima(std::string tag, const int devID) // Get MPI_MAX among all nodes in the world comm
{
  double measure[3];
    
  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].seconds, &measure[0], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_node_comm); //MPI_MAX for time!!!
  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].joules, &measure[1], 1, MPI_DOUBLE, MPI_SUM, my_numa.my_node_comm);
  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].watts, &measure[2], 1, MPI_DOUBLE, MPI_SUM, my_numa.my_node_comm);

  if (my_numa.get_my_node_rank() == 0) // Only the node masters in the world participate to the MPI_MAX search
    {
      MPI_Allreduce(MPI_IN_PLACE, &measure[0], 1, MPI_DOUBLE, MPI_MAX, my_numa.node_masters_comm);
      MPI_Allreduce(MPI_IN_PLACE, &measure[1], 1, MPI_DOUBLE, MPI_MAX, my_numa.node_masters_comm);
      MPI_Allreduce(MPI_IN_PLACE, &measure[2], 1, MPI_DOUBLE, MPI_MAX, my_numa.node_masters_comm);
    }
  else if (my_numa.get_my_node_rank() != 0)
    {
      // Do nothing...
    }
  else
    {
      // The other tasks are simply waiting...
    }
    
    
  for (int index=0; index<3; index++)
    my_node_world_maxima.push_back(measure[index]);

  my_node_world_maxima.at(2) = my_node_world_maxima.at(2) / gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].count; //Do not sum the power for more counts of the same tag
    
  return my_node_world_maxima;
}

std::vector<double> & gpus::get_my_node_world_minima(std::string tag, const int devID) // Get MPI_MIN among all nodes in the world comm
{
  double measure[3];
    
  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].seconds, &measure[0], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_node_comm); //MPI_MAX for time!!!
  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].joules, &measure[1], 1, MPI_DOUBLE, MPI_SUM, my_numa.my_node_comm);
  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].watts, &measure[2], 1, MPI_DOUBLE, MPI_SUM, my_numa.my_node_comm);

  if (my_numa.get_my_node_rank() == 0) // Only the node masters in the world participate to the MPI_MIN search
    {
      MPI_Allreduce(MPI_IN_PLACE, &measure[0], 1, MPI_DOUBLE, MPI_MIN, my_numa.node_masters_comm);
      MPI_Allreduce(MPI_IN_PLACE, &measure[1], 1, MPI_DOUBLE, MPI_MIN, my_numa.node_masters_comm);
      MPI_Allreduce(MPI_IN_PLACE, &measure[2], 1, MPI_DOUBLE, MPI_MIN, my_numa.node_masters_comm);
    }
  else if (my_numa.get_my_node_rank() != 0)
    {
      // Do nothing...
    }
  else
    {
      // The other tasks are simply waiting...
    }
    
    
  for (int index=0; index<3; index++)
    my_node_world_minima.push_back(measure[index]);

  my_node_world_minima.at(2) = my_node_world_minima.at(2) / gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].count; //Do not sum the power for more counts of the same tag
    
  return my_node_world_minima;
}

std::vector<double> & gpus::get_my_node_world_average(std::string tag, const int devID) // Get the average among all nodes in the world comm
{
  double measure[3];
    
  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].seconds, &measure[0], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_node_comm); //MPI_MAX for time!!!
  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].joules, &measure[1], 1, MPI_DOUBLE, MPI_SUM, my_numa.my_node_comm);
  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].watts, &measure[2], 1, MPI_DOUBLE, MPI_SUM, my_numa.my_node_comm);

  if (my_numa.get_my_node_rank() == 0) // Only the node masters in the world participate to the MPI_MIN search
    {
      MPI_Allreduce(MPI_IN_PLACE, &measure[0], 1, MPI_DOUBLE, MPI_SUM, my_numa.node_masters_comm);
      MPI_Allreduce(MPI_IN_PLACE, &measure[1], 1, MPI_DOUBLE, MPI_SUM, my_numa.node_masters_comm);
      MPI_Allreduce(MPI_IN_PLACE, &measure[2], 1, MPI_DOUBLE, MPI_SUM, my_numa.node_masters_comm);

      for (int index=0; index<3; index++)
	measure[index] = measure[index] / my_numa.get_node_masters_ntasks();
    }
  else if (my_numa.get_my_node_rank() != 0)
    {
      // Do nothing...
    }
  else
    {
      // The other tasks are simply waiting...
    }
    
    
  for (int index=0; index<3; index++)
    my_node_world_average.push_back(measure[index]);

  my_node_world_average.at(2) = my_node_world_average.at(2) / gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].count; //Do not sum the power for more counts of the same tag
    
  return my_node_world_average;
}

std::vector<double> & gpus::get_my_node_world_std(std::string tag, const int devID) // Get the standard deviation among all nodes in the world comm
{
  double measure[3];
  double dev[3];
  double node_measure[3];
    
  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].seconds, &measure[0], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_node_comm); //MPI_MAX for time!!!
  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].joules, &measure[1], 1, MPI_DOUBLE, MPI_SUM, my_numa.my_node_comm);
  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].watts, &measure[2], 1, MPI_DOUBLE, MPI_SUM, my_numa.my_node_comm);

  // This step is required for the correct standard deviation calculation
  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].seconds, &node_measure[0], 1, MPI_DOUBLE, MPI_MAX, my_numa.my_node_comm);
  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].joules, &node_measure[1], 1, MPI_DOUBLE, MPI_SUM, my_numa.my_node_comm);
  MPI_Allreduce(&gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].watts, &node_measure[2], 1, MPI_DOUBLE, MPI_SUM, my_numa.my_node_comm);
    
  if (my_numa.get_my_node_rank() == 0) // Only the node masters in the world participate to the MPI_MIN search
    {
      MPI_Allreduce(MPI_IN_PLACE, &measure[0], 1, MPI_DOUBLE, MPI_SUM, my_numa.node_masters_comm);
      MPI_Allreduce(MPI_IN_PLACE, &measure[1], 1, MPI_DOUBLE, MPI_SUM, my_numa.node_masters_comm);
      MPI_Allreduce(MPI_IN_PLACE, &measure[2], 1, MPI_DOUBLE, MPI_SUM, my_numa.node_masters_comm);

      for (int index=0; index<3; index++)
	measure[index] = measure[index] / my_numa.get_node_masters_ntasks();

      // Compute deviations of single nodes from average
      double my_time_dev = (node_measure[0] - measure[0]) * (node_measure[0] - measure[0]);
      double my_energy_dev = (node_measure[1] - measure[1]) * (node_measure[1] - measure[1]);
      double my_power_dev = (node_measure[2] - measure[2]) * (node_measure[2] - measure[2]);

	
	
      // Sum all deviations
      MPI_Allreduce(&my_time_dev, &dev[0], 1, MPI_DOUBLE, MPI_SUM, my_numa.node_masters_comm);
      MPI_Allreduce(&my_energy_dev, &dev[1], 1, MPI_DOUBLE, MPI_SUM, my_numa.node_masters_comm);
      MPI_Allreduce(&my_power_dev, &dev[2], 1, MPI_DOUBLE, MPI_SUM, my_numa.node_masters_comm);
	
	
      // Get standard deviations
      for (int index=0; index<3; index++)
	dev[index] = sqrt(dev[index] / my_numa.get_node_masters_ntasks());

    }
  else if (my_numa.get_my_node_rank() != 0)
    {
      // Do nothing...
    }
  else
    {
      // The other tasks are simply waiting...
    }
    
    
  for (int index=0; index<3; index++)
    my_node_world_std.push_back(dev[index]);

  my_node_world_std.at(2) = my_node_world_std.at(2) / gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].count; //Do not sum the power for more counts of the same tag
    
  return my_node_world_std;
}

  
#endif // defined(_NVIDIA_) || defined(_AMD_)
    
void Create_PMT(const int * const devID, const int numGPUs)
{
  /* Duplicate the original communicator in order to keep the library
     separated from the rest of the program */
  //MPI_Comm_dup(comm, &my_numa.world_comm);

    
    
 #if defined(_AMD_)

  int original_rank, original_size;

  MPI_Comm_rank(MPI_COMM_WORLD, &original_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &original_size);
    
  int Im_even_rank = ( (original_rank % 2 == 0) ? 1 : MPI_UNDEFINED);
  MPI_Comm_split(MPI_COMM_WORLD, Im_even_rank, original_rank, &my_numa.world_comm);

 #else
    
  MPI_Comm_dup(MPI_COMM_WORLD, &my_numa.world_comm);

 #endif // defined(_AMD_)

  /* Only tasks in the communicator must participate */
  if (my_numa.world_comm != MPI_COMM_NULL)
    {
      
      PMT_ERROR.resize(my_numa.get_world_ntasks());
      PMT_ERROR.at(my_numa.get_my_world_rank()) = ((my_numa.get_world_ntasks() > 0) ? false : true);

        
      if (PMT_ERROR.at(my_numa.get_my_world_rank()))
	{
	  PMT_err(my_numa.get_my_world_rank());
	  return;
	}
    
      // Create all the communicators needed by the library
      my_numa.create_communicators();

      if ( my_numa.get_my_world_rank() == 0)
	{
	  if ( my_numa.get_socket_masters_ntasks() < 1)
	    {
	      int error_nsockets = 444;
	      std::cout << "Error! The number of sockets must be at least one! " << std::endl;
	      MPI_Abort(my_numa.world_comm, error_nsockets);
	    }

	  if ( my_numa.get_node_masters_ntasks() < 1)
	    {
	      int error_nnodes = 555;
	      std::cout << "Error! The number of nodes must be at least one! " << std::endl;
	      MPI_Abort(my_numa.world_comm, error_nnodes);
	    }
	}
    
    
      if (my_numa.get_my_world_rank() == 0)
	{
	  std::cout << "\t\t\t[PMT MPI NUMA REPORT]" << std::endl;
	  std::cout << "\t\t\tTotal number of processes " << my_numa.get_world_ntasks() << std::endl;
	  std::cout << "\t\t\tTotal number of nodes " << my_numa.get_node_masters_ntasks() << std::endl;
	  std::cout << "\t\t\tNumber of sockets per node " << my_numa.get_socket_masters_ntasks() << std::endl;
	}
    
      //MPI_Barrier(my_numa.world_comm);

      // Initialize rapl counters for the CPU
      rapl::sensor_cpu.resize(my_numa.get_world_ntasks());
      rapl::sensor_cpu.at(my_numa.get_my_world_rank()) = pmt::rapl::Rapl::Create();
      rapl::state_cpu.resize(my_numa.get_world_ntasks());

      /* Do the same for the GPUs... */
    
      if ((numGPUs > 0) && (devID != nullptr))
	{
	  gpu::sensor_gpu.resize(my_numa.get_world_ntasks());
	  gpu::state_gpu.resize(my_numa.get_world_ntasks());

	  for (int dev=0 ; dev<numGPUs ; dev++)
	    {
	     #if defined(_NVIDIA_) // Sensor creation for NVIDIA GPUs
	      gpu::sensor_gpu.at(my_numa.get_my_world_rank()).insert(std::pair<int, std::unique_ptr<pmt::PMT>>(devID[dev],
													       pmt::nvml::NVML::Create(devID[dev])));
	     #elif defined(_AMD_) // Sensor creation for AMD GPUs
	      gpu::sensor_gpu.at(my_numa.get_my_world_rank()).insert(std::pair<int, std::unique_ptr<pmt::PMT>>(devID[dev],
													       pmt::rocm::ROCM::Create(devID[dev])));
	     #endif // _NVIDIA_ or _AMD_
	    }
	}
    }
  else
    {
      //Do nothing...
    }
    
  return; 
}
  
void Start_PMT_CPU(const char *label)
{

  if ( my_numa.world_comm != MPI_COMM_NULL )
    {
      
      if (PMT_ERROR.at(my_numa.get_my_world_rank()))
	{
	  PMT_err(my_numa.get_my_world_rank());
	  return;
	}

      const std::string tag{std::string{label}};

      // check if the label already exists
      if (rapl::state_cpu.at(my_numa.get_my_world_rank()).count(tag))
	{
	  rapl::state_cpu.at(my_numa.get_my_world_rank())[tag].start = rapl::sensor_cpu.at(my_numa.get_my_world_rank())->Read();
	}
      else
	{
	  // create new EnergyState
	  const EnergyState newState{rapl::sensor_cpu.at(my_numa.get_my_world_rank())->Read(),
				     static_cast<pmt::State>(0),
				     static_cast<double>(0),
				     static_cast<double>(0),
				     static_cast<double>(0),
				     static_cast<unsigned int>(0)};

	  // insert the key and initialize the counters
	  rapl::state_cpu.at(my_numa.get_my_world_rank()).insert(std::pair<std::string, EnergyState>(tag, newState));
	}

    }

  else
    {
      //Do nothing...
    }
    
  return;
}  
  
void Stop_PMT_CPU(const char *label)
{

  if (my_numa.world_comm != MPI_COMM_NULL)
    {
      
      if (PMT_ERROR.at(my_numa.get_my_world_rank()))
	{
	  PMT_err(my_numa.get_my_world_rank());
	  return;
	}

      const std::string tag{std::string{label}};

      // check if the label already exists
      // if not error
      if (!rapl::state_cpu.at(my_numa.get_my_world_rank()).count(tag))
	{
	  PMT_ERROR.at(my_numa.get_my_world_rank()) = true;
	  PMT_err(my_numa.get_my_world_rank());
	  return;
	}
      else
	{
	  // get the energy state
	  EnergyState &State = rapl::state_cpu.at(my_numa.get_my_world_rank())[tag];
      
	  // read the counter
	  State.stop = rapl::sensor_cpu.at(my_numa.get_my_world_rank())->Read();

	  // update quantities
	  State.seconds += rapl::sensor_cpu.at(my_numa.get_my_world_rank())->seconds(State.start, State.stop);
      
	  State.joules  += rapl::sensor_cpu.at(my_numa.get_my_world_rank())->joules(State.start, State.stop);

	  State.watts   += rapl::sensor_cpu.at(my_numa.get_my_world_rank())->watts(State.start, State.stop);

	  State.count++;

	}
    }

  else
    {
      //Do nothing...
    }
    
  return;
}

void Show_PMT_CPU(const char *label)
{
  if (my_numa.world_comm != MPI_COMM_NULL)
    {
    
      const std::string tag{std::string{label}};
    
    
      if (PMT_ERROR.at(my_numa.get_my_world_rank()))
	{
	  PMT_err(my_numa.get_my_world_rank());
	  return;
	}


      else if (rapl::state_cpu.at(my_numa.get_my_world_rank()).count(tag))
	{
	  /* Broadcast to all the tasks in the socket the number of sockets 
	     and the number of nodes IMPORTANT!!!*/

	  int nsockets_per_node, nnodes, nsockets_tot;

	  // SOCKETS
	  if (my_numa.get_my_socket_rank() == 0)
	    {
	      nsockets_per_node = my_numa.get_socket_masters_ntasks();
	      nsockets_tot      = my_numa.get_world_socket_masters_ntasks();
	    }
	  else
	    {
	      nsockets_per_node = 0;
	      nsockets_tot      = 0;
	    }

	  // NODES
	  if (my_numa.get_my_node_rank() == 0)
	    {
	      nnodes   = my_numa.get_node_masters_ntasks();
	    }
	  else
	    {
	      nnodes   = 0;
	    }
	
	  MPI_Bcast(&nsockets_per_node, 1, MPI_INT, 0, my_numa.my_socket_comm);
	  MPI_Bcast(&nsockets_tot, 1, MPI_INT, 0, my_numa.my_socket_comm);
	  MPI_Bcast(&nnodes, 1, MPI_INT, 0, my_numa.my_node_comm);

		
	  MPI_Barrier(my_numa.world_comm);


	  std::vector<double> &my_socket_maxima        = my_measures.get_my_socket_maxima(tag);   // Get the maxima of each socket

	
	  std::vector<double> &my_node_summation       = my_measures.get_my_node_summation(tag); // Get the total measures of each node
	

	
	  std::vector<double> &my_node_socket_maxima   = my_measures.get_my_node_socket_maxima(tag); // Get the maxima of each node
	

	
	  std::vector<double> &my_node_socket_minima   = my_measures.get_my_node_socket_minima(tag); // Get the minima of each node 
	
	  std::vector<double> &my_node_socket_average  = my_measures.get_my_node_socket_average(tag);// Get the average of each node
	  std::vector<double> &my_node_socket_std      = my_measures.get_my_node_socket_std(tag);    // Get the standard deviation of each node
	
	
	  std::vector<double> &my_world_socket_maxima  = my_measures.get_my_world_socket_maxima(tag); // Get the maxima of all sockets in world comm
	  std::vector<double> &my_world_socket_minima  = my_measures.get_my_world_socket_minima(tag); // Get the minima of sockets in world comm
	  std::vector<double> &my_world_socket_average = my_measures.get_my_world_socket_average(tag);// Get the average of all sockets in world comm
	  std::vector<double> &my_world_socket_std     = my_measures.get_my_world_socket_std(tag);    // Get the standard deviation of all sockets in world comm
	  std::vector<double> &my_world_node_maxima    = my_measures.get_my_world_node_maxima(tag); // Get the maxima of all nodes in world 
	  std::vector<double> &my_world_node_minima    = my_measures.get_my_world_node_minima(tag); // Get the minima of nodes in world 
	  std::vector<double> &my_world_node_average   = my_measures.get_my_world_node_average(tag);// Get the average of all nodes in world 
	  std::vector<double> &my_world_node_std       = my_measures.get_my_world_node_std(tag);    // Get the standard deviation of all nodes in world 
	  std::vector<double> &my_world_summation      = my_measures.get_my_world_summation(tag); // Get the total measures of all nodes in world 

	  // Gather relevant quantities to the master in world communicator ex: MPI_COMM_WORLD

	  std::vector<double> single_nodes_summations(nnodes*my_node_summation.size());           // Array of measures per node


	
	  std::vector<double> node_maxima(nnodes*my_node_summation.size());               // Array of maxima per socket (in each node)
	
	  std::vector<double> node_minima(nnodes*my_node_summation.size());               // Array of minima per socket (in each node)
	  std::vector<double> node_average(nnodes*my_node_summation.size());              // Array of averages per socket (in each node)
	  std::vector<double> node_std(nnodes*my_node_summation.size());                  // Array of standard deviations per socket (in each node)

	  
	  int node_buffer_dim = my_node_summation.size();

		
	  if (my_numa.get_my_node_rank() == 0) // Only the node masters perform Allgathers
	    {
	      MPI_Allgather(my_node_summation.data(), node_buffer_dim, MPI_DOUBLE, single_nodes_summations.data(),
			    node_buffer_dim, MPI_DOUBLE, my_numa.node_masters_comm);
	    
	      MPI_Allgather(my_node_socket_maxima.data(), node_buffer_dim, MPI_DOUBLE, node_maxima.data(),
			    node_buffer_dim, MPI_DOUBLE, my_numa.node_masters_comm);
	      MPI_Allgather(my_node_socket_minima.data(), node_buffer_dim, MPI_DOUBLE, node_minima.data(),
			    node_buffer_dim, MPI_DOUBLE, my_numa.node_masters_comm);
	      MPI_Allgather(my_node_socket_average.data(), node_buffer_dim, MPI_DOUBLE, node_average.data(),
			    node_buffer_dim, MPI_DOUBLE, my_numa.node_masters_comm);
	      MPI_Allgather(my_node_socket_std.data(), node_buffer_dim, MPI_DOUBLE, node_std.data(),
			    node_buffer_dim, MPI_DOUBLE, my_numa.node_masters_comm);
	    }
	  else
	    {
	    }

	  if (my_numa.get_my_world_rank() == 0)
	    {
	      // Open the output File
	      std::ofstream OutputFile("CPU_report_"+tag+".txt");

	      OutputFile << "######################################################################" << std::endl;
	      OutputFile << "\t\t\t[PMT MPI NUMA REPORT]" << std::endl;
	      OutputFile << "\tTOTAL NUMBER OF PROCESSES: " << my_numa.get_world_ntasks() << std::endl;
	      OutputFile << "\t\tNUMBER OF NODES: " << my_numa.get_node_masters_ntasks() << std::endl;
	      OutputFile << "\t\t\tNUMBER OF PROCESSES PER NODE: " << my_numa.get_node_ntasks() << std::endl;
	      OutputFile << "\t\t\t\tNUMBER OF SOCKETS PER NODE: " << my_numa.get_socket_masters_ntasks() << std::endl;
	      OutputFile << "\t\t\t\t\tNUMBER OF PROCESSES PER SOCKET: " << my_numa.get_socket_ntasks() << std::endl;
	      OutputFile << "######################################################################" << std::endl;
	    
	      for (int stride=0,node_ID=0; stride<nnodes*my_node_summation.size(), node_ID<nnodes; stride+=my_node_summation.size(), node_ID++)
		{
		  OutputFile << "\n\n" << std::endl;
		  OutputFile << "\t\t\t\t[NODE " << node_ID << "]" << std::endl;
		  OutputFile << "[TOTAL PER NODE]:       " << single_nodes_summations.at(stride) << " [S];   " << single_nodes_summations.at(stride+1) << " [J];   " << single_nodes_summations.at(stride+2) << " [W];" << std::endl;
		

		  if (nsockets_per_node > 1)
		    {
		    
		      // Compute relative errors to enforce the standard deviation calculation
		      double err_time   = node_std.at(stride) * 100 / node_average.at(stride);
		      double err_energy = node_std.at(stride+1) * 100 / node_average.at(stride+1);
		      double err_power  = node_std.at(stride+2) * 100 / node_average.at(stride+2);
		    
		    
		      OutputFile << "\t\t[MAXIMA]:                     " << node_maxima.at(stride) << " [S];   " << node_maxima.at(stride+1) << " [J];   " << node_maxima.at(stride+2) << " [W];" << std::endl;
		      OutputFile << "\t\t[MINIMA]:                     " << node_minima.at(stride) << " [S];   " << node_minima.at(stride+1) << " [J];   " << node_minima.at(stride+2) << " [W];" << std::endl;
		      OutputFile << "\t\t[AVERAGE TIME PER SOCKET]:    " << node_average.at(stride) << " +- " << node_std.at(stride) << " (" << err_time << " %) " << " [S];" << std::endl;
		      OutputFile << "\t\t[AVERAGE ENERGY PER SOCKET]:  " << node_average.at(stride+1) << " +- " << node_std.at(stride+1) << " (" << err_energy << " %) " << " [J];" << std::endl;
		      OutputFile << "\t\t[AVERAGE POWER PER SOCKET]:   " << node_average.at(stride+2) << " +- " << node_std.at(stride+2) << " (" << err_power << " %) " << " [W];" << std::endl;
		    
		    }
		  OutputFile << "\n\n" << std::endl;
		
		}
	    	    	      
	      if (nnodes > 1)
		{
		  double err_time_nodes_total   = my_world_node_std.at(0) * 100 / my_world_node_average.at(0);
		  double err_energy_nodes_total = my_world_node_std.at(1) * 100 / my_world_node_average.at(1);
		  double err_power_nodes_total  = my_world_node_std.at(2) * 100 / my_world_node_average.at(2);
			
		  OutputFile << "######################################################################" << std::endl;
		  OutputFile << "\t\t\t\t\t[PRINT SUMMARY]" << std::endl;
		  OutputFile << "[TOTAL RESULTS FROM ALL NODES]:       " << my_world_summation.at(0) << " [S];   " << my_world_summation.at(1) << " [J];   " << my_world_summation.at(2) << " [W];" << std::endl;

		  OutputFile << "\n\n" << std::endl;
		  OutputFile << "\t\t[ALL NODES MAXIMA]:           " << my_world_node_maxima.at(0) << " [S];   " << my_world_node_maxima.at(1) << " [J];   " << my_world_node_maxima.at(2) << " [W];" << std::endl;
		  OutputFile << "\t\t[ALL NODES MINIMA]:           " << my_world_node_minima.at(0) << " [S];   " << my_world_node_minima.at(1) << " [J];   " << my_world_node_minima.at(2) << " [W];" << std::endl;
		  OutputFile << "\t\t[AVERAGE TIME PER NODE]:      " << my_world_node_average.at(0) << " +- " << my_world_node_std.at(0) << " (" << err_time_nodes_total << " %) " << " [S];" << std::endl;
		  OutputFile << "\t\t[AVERAGE ENERGY PER NODE]:    " << my_world_node_average.at(1) << " +- " << my_world_node_std.at(1) << " (" << err_energy_nodes_total << " %) " << " [J];" << std::endl;
		  OutputFile << "\t\t[AVERAGE POWER PER NODE]:     " << my_world_node_average.at(2) << " +- " << my_world_node_std.at(2) << " (" << err_power_nodes_total << " %) " << " [W];" << std::endl;
		  OutputFile << "\n\n" << std::endl;
		
		  if (nsockets_per_node > 1)
		    {
		      // Compute relative errors to enforce the standard deviation calculation
		      double err_time_sockets_total   = my_world_socket_std.at(0) * 100 / my_world_socket_average.at(0);
		      double err_energy_sockets_total = my_world_socket_std.at(1) * 100 / my_world_socket_average.at(1);
		      double err_power_sockets_total  = my_world_socket_std.at(2) * 100 / my_world_socket_average.at(2);

		      OutputFile << "\t\t[ALL SOCKETS MAXIMA]:         " << my_world_socket_maxima.at(0) << " [S];   " << my_world_socket_maxima.at(1) << " [J];   " << my_world_socket_maxima.at(2) << " [W];" << std::endl;
		      OutputFile << "\t\t[ALL SOCKETS MINIMA]:         " << my_world_socket_minima.at(0) << " [S];   " << my_world_socket_minima.at(1) << " [J];   " << my_world_socket_minima.at(2) << " [W];" << std::endl;
		      OutputFile << "\t\t[AVERAGE TIME PER SOCKET]:    " << my_world_socket_average.at(0) << " +- " << my_world_socket_std.at(0) << " (" << err_time_sockets_total << " %) " << " [S];" << std::endl;
		      OutputFile << "\t\t[AVERAGE ENERGY PER SOCKET]:  " << my_world_socket_average.at(1) << " +- " << my_world_socket_std.at(1) << " (" << err_energy_sockets_total << " %) " << " [J];" << std::endl;
		      OutputFile << "\t\t[AVERAGE POWER PER SOCKET]:   " << my_world_socket_average.at(2) << " +- " << my_world_socket_std.at(2) << " (" << err_power_sockets_total << " %) " << " [W];" << std::endl;
		      OutputFile << "######################################################################" << std::endl;
		    }
	
		}
	    
	      // Close the output File
	      OutputFile.close();
	    
	    }

	
	  /* Free the std::vectors in the allocation order */
	  /*
	  delete[] my_socket_maxima;
	  delete[] my_node_summation;
	  delete[] my_world_socket_maxima;
	  delete[] my_world_socket_minima;
	  delete[] my_world_socket_average;
	  delete[] my_world_socket_std;
	  delete[] my_world_node_maxima;
	  delete[] my_world_node_minima;
	  delete[] my_world_node_average;
	  delete[] my_world_node_std;
	  delete[] my_world_summation;
	  delete[] single_nodes_summations;
	  delete[] node_maxima;
	  delete[] node_minima;
	  delete[] node_average;
	  delete[] node_std;
	  */	
	}
    }
  else
    {
      //Do nothing...
    }
    
  return;
}
  

#if defined(_NVIDIA_) || defined(_AMD_)
void Start_PMT_GPU(const char *label, const int devID)
		     
{

  if (my_numa.world_comm != MPI_COMM_NULL)
    {
      
      if (PMT_ERROR.at(my_numa.get_my_world_rank()) || !gpu::sensor_gpu.at(my_numa.get_my_world_rank()).count(devID))
	{
	  PMT_err(my_numa.get_my_world_rank());
	  return;
	}

      const std::string tag{std::string{label}};

    
      // check if the devID already exists
      if (!gpu::state_gpu.at(my_numa.get_my_world_rank()).count(devID))
	{
	  // initialize the counters
	  const EnergyState newState{gpu::sensor_gpu.at(my_numa.get_my_world_rank())[devID]->Read(),
				     static_cast<pmt::State>(0),
				     static_cast<double>(0),
				     static_cast<double>(0),
				     static_cast<double>(0),
				     static_cast<unsigned int>(0)};
	
	
	  // insert my device ID and the associated pair< label, EnergyState >
	  std::map<std::string, EnergyState> tag_EnergyState;
	  tag_EnergyState.insert(std::pair<std::string, EnergyState>(tag, newState));
	  gpu::state_gpu.at(my_numa.get_my_world_rank()).insert(std::pair<int, std::map<std::string, EnergyState>>(devID, tag_EnergyState));
	
	}
      else
	{
	  // the tag exists
	  if (gpu::state_gpu.at(my_numa.get_my_world_rank())[devID].count(tag))
	    {
	      // read the sensor
	      gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag].start = gpu::sensor_gpu.at(my_numa.get_my_world_rank())[devID]->Read();
	    }
	  else
	    {
	      // insert the label and initialize the counters
	      const EnergyState newState{gpu::sensor_gpu.at(my_numa.get_my_world_rank())[devID]->Read(),
					 static_cast<pmt::State>(0),
					 static_cast<double>(0),
					 static_cast<double>(0),
					 static_cast<double>(0),
					 static_cast<unsigned int>(0)};

	      gpu::state_gpu.at(my_numa.get_my_world_rank())[devID].insert(std::pair<std::string, EnergyState>(tag, newState));
	    }
	}
    }
  else
    {
      //Do nothing...
    }
    
  return;
}

void Stop_PMT_GPU(const char *label, const int devID)
{

  if (my_numa.world_comm != MPI_COMM_NULL)
    {

      // check if the devID already exists
      // if not error
      if (!gpu::state_gpu.at(my_numa.get_my_world_rank()).count(devID) || PMT_ERROR.at(my_numa.get_my_world_rank()) || !gpu::sensor_gpu.at(my_numa.get_my_world_rank()).count(devID))
	{
	  PMT_ERROR.at(my_numa.get_my_world_rank()) = true;
	  PMT_err(my_numa.get_my_world_rank());
	  return;
	}
      else
	{
	  const std::string tag{std::string{label}};
      
	  // check if the label already exists
	  // if not error
	  if (!gpu::state_gpu.at(my_numa.get_my_world_rank())[devID].count(tag))
	    {
	      PMT_ERROR.at(my_numa.get_my_world_rank()) = true;
	      PMT_err(my_numa.get_my_world_rank());
	      return;
	    }
	  else
	    {
	      EnergyState &State = gpu::state_gpu.at(my_numa.get_my_world_rank())[devID][tag];
	  
	      // read the counter
	      State.stop = gpu::sensor_gpu.at(my_numa.get_my_world_rank())[devID]->Read();

	      // update quantities
	      State.seconds +=
		gpu::sensor_gpu.at(my_numa.get_my_world_rank())[devID]->seconds(State.start,
										State.stop);
      
	      State.joules +=
		gpu::sensor_gpu.at(my_numa.get_my_world_rank())[devID]->joules(State.start,
									       State.stop);

	      State.watts +=
		gpu::sensor_gpu.at(my_numa.get_my_world_rank())[devID]->watts(State.start,
									      State.stop);
      
	      State.count++;
	    }
	}

    }
  else
    {
      //Do nothing...
    }
    
  return;
}

void Show_PMT_GPU(const char *label, const int devID)
{
    
  if (my_numa.world_comm != MPI_COMM_NULL)
    {

      if (PMT_ERROR.at(my_numa.get_my_world_rank()))
	{
	  PMT_err(my_numa.get_my_world_rank());
	  return;
	}
      else
	{
	  const std::string tag{std::string{label}};

	

	  int nnodes;
	
	  // NODES (Broadcast again to all tasks)
	  if (my_numa.get_my_node_rank() == 0)
	    {
	      nnodes   = my_numa.get_node_masters_ntasks();
	    }
	  else
	    {
	      nnodes   = 0;
	    }
	
	  MPI_Bcast(&nnodes, 1, MPI_INT, 0, my_numa.my_node_comm);

		
	  MPI_Barrier(my_numa.world_comm);

	
	  std::vector<double> &my_node_gpu_maxima        = my_gpus.get_my_node_gpu_maxima(tag, devID);   // Get the maxima of all node's GPUs
	
	  std::vector<double> &my_node_gpu_minima        = my_gpus.get_my_node_gpu_minima(tag, devID);   // Get the minima of all node's GPUs
	  std::vector<double> &my_node_gpu_average       = my_gpus.get_my_node_gpu_average(tag, devID);  // Get the averages of all node's GPUs
	  std::vector<double> &my_node_gpu_std           = my_gpus.get_my_node_gpu_std(tag, devID);      // Get the standard deviations of all node's GPUs
	  std::vector<double> &my_world_gpu_maxima       = my_gpus.get_my_world_gpu_maxima(tag, devID);   // Get the maxima of all world's GPUs
	  std::vector<double> &my_world_gpu_minima       = my_gpus.get_my_world_gpu_minima(tag, devID);   // Get the minima of all world's GPUs
	  std::vector<double> &my_world_gpu_average      = my_gpus.get_my_world_gpu_average(tag, devID);  // Get the averages of all world's GPUs
	  std::vector<double> &my_world_gpu_std          = my_gpus.get_my_world_gpu_std(tag, devID);      // Get the standard deviations of all world's GPUs
	  std::vector<double> &my_node_gpu_summation     = my_gpus.get_my_node_gpu_summation(tag, devID); // Get the summations of all node's GPUs
	  std::vector<double> &my_world_gpu_summation    = my_gpus.get_my_world_gpu_summation(tag, devID);// Get the summations of all world's GPUs
	  std::vector<double> &my_node_world_maxima      = my_gpus.get_my_node_world_maxima(tag, devID);   // Get the maxima of all nodes
	  std::vector<double> &my_node_world_minima      = my_gpus.get_my_node_world_minima(tag, devID);   // Get the minima of all nodes
	  std::vector<double> &my_node_world_average     = my_gpus.get_my_node_world_average(tag, devID);  // Get the averages of all nodes
	  std::vector<double> &my_node_world_std         = my_gpus.get_my_node_world_std(tag, devID);      // Get the standard deviations of all nodes
	
	
	  // Gather relevant quantities to the master in world communicator ex: MPI_COMM_WORLD

	  std::vector<double> single_nodes_summations(nnodes*my_node_gpu_summation.size());           // Array of measures per node


	
	  std::vector<double> node_maxima(nnodes*my_node_gpu_summation.size());               // Array of maxima per socket (in each node)
	
	  std::vector<double> node_minima(nnodes*my_node_gpu_summation.size());               // Array of minima per socket (in each node)
	  std::vector<double> node_average(nnodes*my_node_gpu_summation.size());              // Array of averages per socket (in each node)
	  std::vector<double> node_std(nnodes*my_node_gpu_summation.size());                  // Array of standard deviations per socket (in each node)

	  
	  int node_buffer_dim = my_node_gpu_summation.size();

		
	  if (my_numa.get_my_node_rank() == 0) // Only the node masters perform Allgathers
	    {
	      MPI_Allgather(my_node_gpu_summation.data(), node_buffer_dim, MPI_DOUBLE, single_nodes_summations.data(),
			    node_buffer_dim, MPI_DOUBLE, my_numa.node_masters_comm);
	    
	      MPI_Allgather(my_node_gpu_maxima.data(), node_buffer_dim, MPI_DOUBLE, node_maxima.data(),
			    node_buffer_dim, MPI_DOUBLE, my_numa.node_masters_comm);
	      MPI_Allgather(my_node_gpu_minima.data(), node_buffer_dim, MPI_DOUBLE, node_minima.data(),
			    node_buffer_dim, MPI_DOUBLE, my_numa.node_masters_comm);
	      MPI_Allgather(my_node_gpu_average.data(), node_buffer_dim, MPI_DOUBLE, node_average.data(),
			    node_buffer_dim, MPI_DOUBLE, my_numa.node_masters_comm);
	      MPI_Allgather(my_node_gpu_std.data(), node_buffer_dim, MPI_DOUBLE, node_std.data(),
			    node_buffer_dim, MPI_DOUBLE, my_numa.node_masters_comm);
	    }
	  else
	    {
	    }


	  if (my_numa.get_my_world_rank() == 0)
	    {
	      // Open the output File
	      std::ofstream OutputFile("GPU_report_"+tag+".txt");

	      /* BE AWARE!!! ONE GPU PER MPI TASK!!! */
	      int ngpus_per_node = my_numa.get_node_ntasks();
	    
	      OutputFile << "######################################################################" << std::endl;
	      OutputFile << "\t\t\t[PMT MPI NUMA REPORT]" << std::endl;
	      OutputFile << "\tTOTAL NUMBER OF PROCESSES: " << my_numa.get_world_ntasks() << std::endl;
	      OutputFile << "\t\tNUMBER OF NODES: " << my_numa.get_node_masters_ntasks() << std::endl;
	      OutputFile << "\t\t\tNUMBER OF GPUS PER NODE: " << ngpus_per_node << std::endl;
	      OutputFile << "######################################################################" << std::endl;	    

	      for (int stride=0,node_ID=0; stride<nnodes*my_node_gpu_summation.size(), node_ID<nnodes; stride+=my_node_gpu_summation.size(), node_ID++)
		{
		  OutputFile << "\n\n" << std::endl;
		  OutputFile << "\t\t\t\t[NODE " << node_ID << "]" << std::endl;
		  OutputFile << "[TOTAL PER NODE]:       " << single_nodes_summations.at(stride) << " [S];   " << single_nodes_summations.at(stride+1) << " [J];   " << single_nodes_summations.at(stride+2) << " [W];" << std::endl;
		

		  if (ngpus_per_node > 1)
		    {
		    
		      // Compute relative errors to enforce the standard deviation calculation
		      double err_time   = node_std.at(stride) * 100 / node_average.at(stride);
		      double err_energy = node_std.at(stride+1) * 100 / node_average.at(stride+1);
		      double err_power  = node_std.at(stride+2) * 100 / node_average.at(stride+2);
		   
		    
		      OutputFile << "\t\t[MAXIMA]:                     " << node_maxima.at(stride) << " [S];   " << node_maxima.at(stride+1) << " [J];   " << node_maxima.at(stride+2) << " [W];" << std::endl;
	
		      OutputFile << "\t\t[MINIMA]:                     " << node_minima.at(stride) << " [S];   " << node_minima.at(stride+1) << " [J];   " << node_minima.at(stride+2) << " [W];" << std::endl;
		      OutputFile << "\t\t[AVERAGE TIME PER GPU]:    " << node_average.at(stride) << " +- " << node_std.at(stride) << " (" << err_time << " %) " << " [S];" << std::endl;
		      OutputFile << "\t\t[AVERAGE ENERGY PER GPU]:  " << node_average.at(stride+1) << " +- " << node_std.at(stride+1) << " (" << err_energy << " %) " << " [J];" << std::endl;
		      OutputFile << "\t\t[AVERAGE POWER PER GPU]:   " << node_average.at(stride+2) << " +- " << node_std.at(stride+2) << " (" << err_power << " %) " << " [W];" << std::endl;
	
		    }
		  OutputFile << "\n\n" << std::endl;
		
		}

	      if (nnodes > 1)
		{
		  double err_time_nodes_total   = my_node_world_std.at(0) * 100 / my_node_world_average.at(0);
		  double err_energy_nodes_total = my_node_world_std.at(1) * 100 / my_node_world_average.at(1);
		  double err_power_nodes_total  = my_node_world_std.at(2) * 100 / my_node_world_average.at(2);
			
		  OutputFile << "######################################################################" << std::endl;
		  OutputFile << "\t\t\t\t\t[PRINT SUMMARY]" << std::endl;
		  OutputFile << "[TOTAL RESULTS FROM ALL NODES]:       " << my_world_gpu_summation.at(0) << " [S];   " << my_world_gpu_summation.at(1) << " [J];   " << my_world_gpu_summation.at(2) << " [W];" << std::endl;

		  OutputFile << "\n\n" << std::endl;
		  OutputFile << "\t\t[ALL NODES MAXIMA]:           " << my_node_world_maxima.at(0) << " [S];   " << my_node_world_maxima.at(1) << " [J];   " << my_node_world_maxima.at(2) << " [W];" << std::endl;
		  OutputFile << "\t\t[ALL NODES MINIMA]:           " << my_node_world_minima.at(0) << " [S];   " << my_node_world_minima.at(1) << " [J];   " << my_node_world_minima.at(2) << " [W];" << std::endl;
		  OutputFile << "\t\t[AVERAGE TIME PER NODE]:      " << my_node_world_average.at(0) << " +- " << my_node_world_std.at(0) << " (" << err_time_nodes_total << " %) " << " [S];" << std::endl;
		  OutputFile << "\t\t[AVERAGE ENERGY PER NODE]:    " << my_node_world_average.at(1) << " +- " << my_node_world_std.at(1) << " (" << err_energy_nodes_total << " %) " << " [J];" << std::endl;
		  OutputFile << "\t\t[AVERAGE POWER PER NODE]:     " << my_node_world_average.at(2) << " +- " << my_node_world_std.at(2) << " (" << err_power_nodes_total << " %) " << " [W];" << std::endl;
		  OutputFile << "\n\n" << std::endl;

		
		  if (ngpus_per_node > 1)
		    {
		      // Compute relative errors to enforce the standard deviation calculation
		      double err_time_gpus_total   = my_world_gpu_std.at(0) * 100 / my_world_gpu_average.at(0);
		      double err_energy_gpus_total = my_world_gpu_std.at(1) * 100 / my_world_gpu_average.at(1);
		      double err_power_gpus_total  = my_world_gpu_std.at(2) * 100 / my_world_gpu_average.at(2);
  
		      OutputFile << "\t\t[ALL GPUS MAXIMA]:            " << my_world_gpu_maxima.at(0) << " [S];   " << my_world_gpu_maxima.at(1) << " [J];   " << my_world_gpu_maxima.at(2) << " [W];" << std::endl;
		      OutputFile << "\t\t[ALL GPUS MINIMA]:            " << my_world_gpu_minima.at(0) << " [S];   " << my_world_gpu_minima.at(1) << " [J];   " << my_world_gpu_minima.at(2) << " [W];" << std::endl;
		      OutputFile << "\t\t[AVERAGE TIME PER GPU]:       " << my_world_gpu_average.at(0) << " +- " << my_world_gpu_std.at(0) << " (" << err_time_gpus_total << " %) " << " [S];" << std::endl;
		      OutputFile << "\t\t[AVERAGE ENERGY PER GPU]:     " << my_world_gpu_average.at(1) << " +- " << my_world_gpu_std.at(1) << " (" << err_energy_gpus_total << " %) " << " [J];" << std::endl;
		      OutputFile << "\t\t[AVERAGE POWER PER GPU]:      " << my_world_gpu_average.at(2) << " +- " << my_world_gpu_std.at(2) << " (" << err_power_gpus_total << " %) " << " [W];" << std::endl;
		      OutputFile << "######################################################################" << std::endl;
		    }

		}

	      // Close the output File
	      OutputFile.close();
	    
	    
	    
	    }
	
	
	  /* I'M NOW COMMENTING THIS CODE PORTION 
	     BUT I HAVE TO ASK DAVID */
	  /*
	    for (const auto &[key, value]: state_gpu.at(my_numa.get_my_world_rank()))
	    {
	    if (value.count(tag))
	    {
	    std::cout << "\n\t GPU [" << key << "] kernel:" << tag << ":" << std::endl;
	    std::cout << "\t\t" << value.at(tag).seconds << " [s]" << std::endl;
	    std::cout << "\t\t" << value.at(tag).joules  << " [J]" << std::endl;
	    std::cout << "\t\t" << value.at(tag).watts / value.at(tag).count  << " [W]" << "\n" << std::endl;
	    }
	    }
	  */

	  /* Free the std::vectors in the allocation order */                  

	  /*
	  delete[] my_node_gpu_summation;
	  	
	  delete[] my_node_gpu_maxima;
	  delete[] my_node_gpu_minima;
	  delete[] my_node_gpu_average;
	  delete[] my_node_gpu_std;
	  delete[] my_world_gpu_maxima;
	  delete[] my_world_gpu_minima;
	  delete[] my_world_gpu_average;
	  delete[] my_world_gpu_std;
	  delete[] my_world_gpu_summation;
	  delete[] single_nodes_summations;
	  delete[] node_maxima;
	  delete[] node_minima;
	  delete[] node_average;
	  delete[] node_std;
	  delete[] my_node_world_maxima;
	  delete[] my_node_world_minima;
	  delete[] my_node_world_average;
	  delete[] my_node_world_std;
	  */
	}
    }
  else
    {
      //Do nothing...
    }
    
    
  return;
}
#endif // defined(_NVIDIA_) || defined(_AMD_)

void Free_PMT()
{
  if (my_numa.get_my_world_rank() == 0)
    my_numa.free_communicators();
}

//}



  
