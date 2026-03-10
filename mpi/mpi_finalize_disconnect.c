/******************************************************************************
 * This shared library overrides mpi_finalize() in the spawned VIC process.
 * Before calling pmpi_finalize(), it checks whether the process was started
 * with mpi_comm_spawn() and disconnects from the parent intercommunicator.
 *
 * The VIC image driver is spawned from the python coupling controller. If the
 * child ranks finalize while still attached to the parent communicator, the
 * mpi runtime can leave the parent-child relationship in a bad state. That can
 * lead to hangs, aborts, or inconsistent shutdown behavior across coupling
 * windows.
 *
 * This helper keeps the fix outside the VIC source tree. It is intended to be
 * preloaded into the spawned VIC executable so the disconnect happens at the
 * C MPI layer immediately before finalization.
 *
 * Build:
 * Compile this file into libvic_parent_disconnect.so using the shell script
 * compile_mpi_disconnect_c.sh in the same directory.
 ******************************************************************************/

#include <mpi.h>

int MPI_Finalize(void)
{
    MPI_Comm parent = MPI_COMM_NULL;
    MPI_Comm_get_parent(&parent);

    if (parent != MPI_COMM_NULL) {
        MPI_Comm_disconnect(&parent);
    }

    return PMPI_Finalize();
}
