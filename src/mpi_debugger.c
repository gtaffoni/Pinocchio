/**
 * @file mpi_debugger.c
 * @brief MPI debugging utility: attach debugger to running MPI process group
 *
 * @details
 * Provides a utility function to pause a running MPI application and allow
 * debuggers (gdb, lldb) to attach to specific ranks. Useful for debugging
 * parallel code issues that only manifest at runtime.
 *
 * **Usage:**
 *
 * Add `#define MPI_ATTACH_DEBUGGER` before including this file, then call:
 * ```c
 * mpi_attach_debugger(MPI_COMM_WORLD);
 * ```
 *
 * The function will:
 * 1. Print rank, PID, and hostname for each MPI rank
 * 2. Write a file `pid_list_for_debugger.txt` with all process IDs
 * 3. Spin in a loop waiting for the continue_run flag to be set
 *
 * Attach debugger to a process by PID:
 * ```bash
 * gdb -p <PID>
 * (gdb) set var continue_run = 1
 * (gdb) continue
 * ```
 *
 * **Compilation:**
 *
 * Define `MPI_ATTACH_DEBUGGER` at compile time to enable this module:
 * ```makefile
 * COPTS = -DMPI_ATTACH_DEBUGGER ...
 * ```
 *
 * @author Leonardo collaboration
 * @date 2025
 * @version 1.0
 *
 * @see mpi_attach_debugger()
 */

#ifdef MPI_ATTACH_DEBUGGER

#include <unistd.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MASTER 0

/**
 * @brief Pause MPI application and wait for debugger attachment
 *
 * @details
 * Synchronizes all MPI ranks, prints process information (PID, hostname),
 * collects all PIDs to a file, then spins until the continue_run flag
 * is set (typically by a debugger via `set var continue_run=1`).
 *
 * This is useful for debugging race conditions and parallel logic errors
 * that require stopping execution at a known synchronization point across
 * all ranks.
 *
 * @param[in] comm  MPI communicator (typically MPI_COMM_WORLD)
 *
 * @return void
 *
 * @par MPI
 * Collective operation. All ranks in comm must call this function.
 * Includes MPI_Barrier() calls for synchronization.
 *
 * @note Only compiled if MPI_ATTACH_DEBUGGER is defined at compile time.
 *
 * @par Example
 * ```c
 * #define MPI_ATTACH_DEBUGGER
 * #include "mpi_debugger.c"
 * ...
 * mpi_attach_debugger(MPI_COMM_WORLD);
 * // All ranks now spinning, waiting for debugger
 * ```
 *
 * @warning This function BLOCKS indefinitely until continue_run is set.
 *          Do not call in production code or in tight loops.
 * @warning Spinning in the while loop may consume CPU. Use in test runs only.
 */
void mpi_attach_debugger(MPI_Comm comm)
{
  // get MPI rank
  int rank = -1;
  MPI_Comm_rank(comm, &rank);

  // get MPI communicator size                                                                                                                                                           
  int Nranks = -1;
  MPI_Comm_size(comm, &Nranks);
  // get the hostname

  char hostname[MPI_MAX_PROCESSOR_NAME];
  int resultlen = -1;
  MPI_Get_processor_name(hostname, &resultlen);

  // get the pid of the current process
  const pid_t pid = getpid();

  for (int task=0 ; task<Nranks ; task++)
    {
      if (task == rank)
	{
          printf("\n\t Task: %d - pid: %i - hostname: %s",
                 rank, pid, hostname);
          fflush(stdout);
        }
      MPI_Barrier(comm);
    }

  // master rank collects all pids
  pid_t *all_pids = NULL;
  if (rank == MASTER)
    all_pids = (pid_t *)malloc(Nranks * sizeof(pid_t));

  MPI_Gather(&pid,     sizeof(pid_t), MPI_BYTE,
             all_pids, sizeof(pid_t), MPI_BYTE,
             MASTER, comm);

  if (rank == MASTER)
    {
      FILE *fd = fopen("pid_list_for_debugger.txt", "w");
      for (int task=0 ; task<Nranks ; task++)
        fprintf(fd, "%i \n", all_pids[task]);
      fclose(fd);

      free(all_pids);

      printf("\n\n\n\t pid_list_for_debugger.txt written \n\n");
      fflush(stdout);
    }

  MPI_Barrier(comm);

  volatile int continue_run = 0;
  while (continue_run == 0) /* continue_run needsto be set to 1 ("set var
                               continue_run = 1") by debugger to continue */
    {
      sleep(1);
    }

  return;
}

#endif // MPI_ATTACH_DEBUGGER
