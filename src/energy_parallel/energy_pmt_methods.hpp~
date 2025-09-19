#pragma once
#include <mpi.h>

#include "functions.h"

//namespace mpi
//{
void PMT_err(const int task);

MPI_Comm get_my_socket_communicator(MPI_Comm, int, int);
  
class numa
{
private:
  int world_rank;                   // Rank in the original communicator ex: MPI_COMM_WORLD
  int world_ntasks;                 // Ntasks in the original communicator ex: MPI_COMM_WORLD
  int my_node_rank;                 // Rank in the intra-node communicator
  int my_node_ntasks;               // Ntasks in the intra-node communicator
  int my_socket_rank;               // Rank in the intra-socket communicator
  int my_socket_ntasks;             // Ntasks in the intra-socket communicator
  int socket_masters_rank;          // Rank in socket masters' communicator
  int socket_masters_ntasks;        // Ntasks in socket masters' communicator
  int node_masters_rank;            // Rank in node masters' communicator
  int node_masters_ntasks;          // Ntasks in node masters' communicator
  int my_world_socket_masters_rank; // Rank in world socket masters' communicator
  int world_socket_masters_ntasks;  // Ntasks in world socket masters' communicator

    
public: 
  MPI_Comm world_comm;         // Original communicator ex: MPI_COMM_WORLD    
  MPI_Comm my_node_comm;       // Intra-node communicator
  MPI_Comm my_socket_comm;     // Intra-socket communicator
  MPI_Comm socket_masters_comm;// Socket masters' communicator intra-node 
  MPI_Comm node_masters_comm;         // Node masters' communicator in the original communicator ex: MPI_COMM_WORLD
  MPI_Comm world_socket_masters_comm; // Socket masters' communicator in the original communicator ex: MPI_COMM_WORLD
    
  void create_communicators(void);
  int get_my_world_rank(void);          // Get my rank in the original communicator
  int get_my_node_rank(void);          // Get my rank in the intra-node communicator
  int get_my_socket_rank(void);       // Get my rank in the intra-socket communicator
  int get_my_socket_masters_rank(void); // Get my rank in the socket masters' communicator
  int get_my_world_socket_masters_rank(void); // Get my rank in the world socket masters' communicator
  int get_my_node_masters_rank(void); // Get my rank in the node masters' communicator
  int get_world_ntasks(void);         // Get ntasks in the original communicator
  int get_node_ntasks(void);          // Get ntasks in the intra-node communicator
  int get_socket_ntasks(void);        // Get ntasks in the intra-socket communicator
  int get_socket_masters_ntasks(void);  // Get ntasks in the socket masters' communicator
  int get_world_socket_masters_ntasks(void); // Get ntasks in the world socket masters' communicator
  int get_node_masters_ntasks(void);  // Get ntasks in the node masters' communicator
}; //class numa

/* MEASURES ARE TIME, ENERGY AND POWER, THIS EXPLAINS WHY WE NEED ARRAYS OF THREE DOUBLES */
class measures
{
private:
  std::vector<double> my_socket_maxima;       // Measures per socket: MPI_MAX among all tasks in the socket
  std::vector<double> my_node_socket_maxima;  // Measures per node's sockets: MPI_MAX among all the sockets in the node
  std::vector<double> my_node_socket_minima;  // Measures per node's sockets: MPI_MIN among all the sockets in the node
  std::vector<double> my_node_socket_average; // Measures per node's sockets: Average among all the sockets in the node
  std::vector<double> my_node_socket_std;     // Measures per node's sockets: Standard deviation among all the sockets in the node
  std::vector<double> my_world_socket_maxima;  // Measures per world's sockets: MPI_MAX among all the sockets in the world comm
  std::vector<double> my_world_socket_minima;  // Measures per world's sockets: MPI_MIN among all the sockets in the world comm
  std::vector<double> my_world_socket_average; // Measures per world's sockets: Average among all the sockets in the world comm
  std::vector<double> my_world_socket_std;     // Measures per world's sockets: Standard deviation among all the sockets in the world comm
  std::vector<double> my_node_summation;       // Measures per node: MPI_SUM among all the sockets in the node or MPI_MAX among alxol the tasks in the node
  std::vector<double> my_world_node_maxima;    // Measures per world's nodes: MPI_MAX among all the nodes in the world comm
  std::vector<double> my_world_node_minima;    // Measures per world's nodes: MPI_MIN among all the nodes in the world comm
  std::vector<double> my_world_node_average;   // Measures per world's nodes: Average among all the nodes in the world comm
  std::vector<double> my_world_node_std;       // Measures per world's nodes: Standard deviation among all the nodes in the world comm
  std::vector<double> my_world_summation;      //Measures per world's nodes: 


public:
  std::vector<double> &get_my_socket_maxima(std::string);        // Get MPI_MAX among all tasks in the socket
  std::vector<double> &get_my_node_summation(std::string);       // Get MPI_SUM among all sockets in the node or MPI_MAX among all the tasks in the node if nsockets_per_node = 1
  std::vector<double> &get_my_node_socket_maxima(std::string);   // Get MPI_MAX among all tasks in the socket masters' in the node
  std::vector<double> &get_my_node_socket_minima(std::string);   // Get MPI_MIN among all tasks in the socket masters' in the node
  std::vector<double> &get_my_node_socket_average(std::string);  // Get the average among all tasks in the socket masters' in the node
  std::vector<double> &get_my_node_socket_std(std::string);      // Get the standard deviation among all tasks in the socket masters' in the node
  std::vector<double> &get_my_world_socket_maxima(std::string);  // Get MPI_MAX among all tasks in the socket masters in the world comm
  std::vector<double> &get_my_world_socket_minima(std::string);  // Get MPI_MIN among all tasks in the socket masters in the world comm
  std::vector<double> &get_my_world_socket_average(std::string); // Get the average among all tasks in the socket masters in the world comm
  std::vector<double> &get_my_world_socket_std(std::string);     // Get the standard deviation among all tasks in the socket masters in the world comm
  std::vector<double> &get_my_world_node_maxima(std::string);    // Get MPI_MAX among all tasks in the node masters in the world comm
  std::vector<double> &get_my_world_node_minima(std::string);    // Get MPI_MIN among all tasks in the node masters in the world comm
  std::vector<double> &get_my_world_node_average(std::string);   // Get the average among all tasks in the node masters in the world comm
  std::vector<double> &get_my_world_node_std(std::string);       // Get the standard deviation among all tasks in the node masters in the world comm
  std::vector<double> &get_my_world_summation(std::string);      // Get MPI_SUM among all nodes in the world
};

  
/* MEASURES ARE TIME, ENERGY AND POWER, THIS EXPLAINS WHY WE NEED ARRAYS OF THREE DOUBLES */
/* THIS TIME THIS IT'S NEEDED FOR GPUS */
class gpus
{
private:
  int numDev;                                 // Get the number of GPUs (AGAIN: BE CAREFUL ON TURNING ON MPI OR NOT! IN THE CASE OF MPI IT IS NATURAL TO HAVE 1 MPI TASK PER GPU)
  std::vector<double> my_node_gpu_maxima;     // Measures per node: MPI_MAX among all GPUs in the node
  std::vector<double> my_node_gpu_minima;     // Measures per node: MPI_MIN among all GPUs in the node
  std::vector<double> my_node_gpu_average;    // Measures per node: Average among all GPUs in the node
  std::vector<double> my_node_gpu_std;        // Measures per node: Standard deviation among all GPUs in the node
  std::vector<double> my_world_gpu_maxima;    // Measures per world: MPI_MAX among all GPUs in the world comm
  std::vector<double> my_world_gpu_minima;    // Measures per world: MPI_MIN among all GPUs in the world comm
  std::vector<double> my_world_gpu_average;   // Measures per world: Average among all GPUs in the world comm
  std::vector<double> my_world_gpu_std;       // Measures per world: Standard deviation among all GPUs in the world comm
  std::vector<double> my_node_gpu_summation;  // Measures per node: Summation over all GPUs in the node
  std::vector<double> my_world_gpu_summation; // Measures per world: Summation over all GPUs in the world comm
  std::vector<double> my_node_world_maxima;   // Measures per world: MPI_MAX among all GPUs in the node
  std::vector<double> my_node_world_minima;   // Measures per world: MPI_MIN among all GPUs in the node
  std::vector<double> my_node_world_average;  // Measures per world: Average among all GPUs in the node
  std::vector<double> my_node_world_std;      // Measures per world: Standard deviation among all GPUs in the node
    
    
    
public:
  std::vector<double> &get_my_node_gpu_maxima(std::string, const int devID);   // Get MPI_MAX among all GPUs in the node
  std::vector<double> &get_my_node_gpu_minima(std::string, const int devID);   // Get MPI_MIN among all GPUs in the node
  std::vector<double> &get_my_node_gpu_average(std::string, const int devID);  // Get the average among all GPUs in the node
  std::vector<double> &get_my_node_gpu_std(std::string, const int devID);      // Get the standard deviation among all GPUs in the node
  std::vector<double> &get_my_world_gpu_maxima(std::string, const int devID);   // Get MPI_MAX among all GPUs in the world comm
  std::vector<double> &get_my_world_gpu_minima(std::string, const int devID);   // Get MPI_MIN among all GPUs in the world comm
  std::vector<double> &get_my_world_gpu_average(std::string, const int devID);  // Get the average among all GPUs in the world comm
  std::vector<double> &get_my_world_gpu_std(std::string, const int devID);      // Get the standard deviation among all GPUs in the world comm
  std::vector<double> &get_my_node_gpu_summation(std::string, const int devID); // Get the summation over all GPUs in the node
  std::vector<double> &get_my_world_gpu_summation(std::string, const int devID); // Get the summation over all GPUs in the world comm
  std::vector<double> &get_my_node_world_maxima(std::string, const int devID);   // Get MPI_MAX among all nodes in the world comm
  std::vector<double> &get_my_node_world_minima(std::string, const int devID);   // Get MPI_MIN among all nodes in the world comm
  std::vector<double> &get_my_node_world_average(std::string, const int devID);  // Get the average among all nodes in the world comm
  std::vector<double> &get_my_node_world_std(std::string, const int devID);      // Get the standard deviation among all nodes in the world comm
    
};
  

//} //namespace mpi
