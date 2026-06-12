/*****************************************************************
 *                        PINOCCHIO  V5.1                        *
 *  (PINpointing Orbit-Crossing Collapsed HIerarchical Objects)  *
 *****************************************************************
 
 This code was written by
 Pierluigi Monaco, Tom Theuns, Giuliano Taffoni, Marius Lepinzan, 
 Chiara Moretti, Luca Tornatore, David Goz, Tiago Castro
 Copyright (C) 2025
 
 github: https://github.com/pigimonaco/Pinocchio
 web page: http://adlibitum.oats.inaf.it/monaco/pinocchio.html
 
 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 2 of the License, or
 (at your option) any later version.
 
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/


#include "pinocchio.h"
#include "parallel_sort.h"
#include <sys/types.h>
#include <sys/stat.h>

//#define DEBUG

int fragment(void);
int count_peaks(int *);
void set_fragment_parameters(int);
void write_map(int);
int create_map(void);
void reorder(int *, int);
void reorder_nofrag(int *, int); 
void sort_and_organize(void);
#ifdef RECOMPUTE_DISPLACEMENTS
int shift_all_displacements(void);
int recompute_group_velocities(void);
#endif


void set_fragment_parameters(int order)
{

#ifndef TWO_LPT
  order=1;
#else
#ifndef THREE_LPT
  if (order>2)
    order=2;
#else
  if (order>3)
    order=3;
#endif
#endif

  /* Parameters for the simulation */

  f_200   = 0.171;
  switch(order)
    {
    case 1:
#ifdef USE_SIM_PARAMS
      f_m = f_a = 0.495;
      f_rm    = -0.075;
      espo    = 0.852;
      f_ra    = 0.500;
#else
      f_m = f_a = 0.505;
      f_rm    = 0.000;
      espo    = 0.820;
      f_ra    = 0.300;
#endif  
      sigmaD0 = 1.7;
      break;

    case 2:
#ifdef USE_SIM_PARAMS
      f_m = f_a = 0.475;
      f_rm    = -0.020;
      espo    = 0.780;
      f_ra    = 0.650;
#else
      f_m = f_a = 0.501;
      f_rm    = 0.052;
      espo    = 0.745;
      f_ra    = 0.334;
#endif
      sigmaD0 = 1.5;
      break;

    case 3:
#ifdef USE_SIM_PARAMS
      f_m = f_a = 0.455;
      f_rm    = 0.000;
      espo    = 0.755;
      f_ra    = 0.700;
#else
      f_m = f_a = 0.5024;
      f_rm    = 0.1475;
      espo    = 0.6852;
      f_ra    = 0.4584;
#endif
      sigmaD0 = 1.2;
      break;
 
    default:
      break;
    }

}


static int index_compare_F(const void * a,const void * b)
{
  if ( frag[*((int *)a)].Fmax == frag[*((int *)b)].Fmax )
    return 0;
  else
    if ( frag[*((int *)a)].Fmax > frag[*((int *)b)].Fmax )
      return -1;
    else
      return 1;
}


static int index_compare_P(const void * a,const void * b)
{
  if ( frag_pos[*((int *)a)] == frag_pos[*((int *)b)] )
    return 0;
  else
    if ( frag_pos[*((int *)a)] < frag_pos[*((int *)b)] )
      return -1;
    else
      return 1;
}


int fragment_driver()
{
  /* This routine can be edited to call fragmentation for many combinations 
     of fragment parameters */

  if (!ThisTask)
    printf("\n[%s] Second part: fragmentation of the collapsed medium\n",fdate());

  /* parameters are assigned in the standard way */
  set_fragment_parameters(ORDER_FOR_GROUPS);
  
  /* reallocates the memory */
  if (reallocate_memory_for_fragmentation())
    return 1;

  if (fragment())
    return 1;

  return 0;
}


int fragment()
{
  /* This function is the driver for fragmentation of collapsed medium
     and construction of halo catalogs */

  int Npeaks, Ngood;
#ifndef CLASSIC_FRAGMENTATION
  unsigned int nadd[2];
#endif
  double BestPredPeakFactor;
  unsigned long long mynadd[2], nadd_all[2];
  double tmp;


  /* timing */
  cputime.frag=MPI_Wtime();
  cputime.group=cputime.sort=cputime.distr=0.0;
#ifdef PLC
  cputime.plc=0.0;
#endif


  /* In the updated code, fragmentation is performed twice;
     in the first turn a quick fragmentation is performed to locate
     halos and determine the interesting particles; velocities are not updated;
     in the second turn the full fragmentation is performed, segmentation is applied
     and velocities are updated.
     For the classic fragmentation the first turn is skipped */

#ifdef CLASSIC_FRAGMENTATION
#define START_TURN 1
#else
#define START_TURN 0
  int all_pbc = (subbox.pbc[_x_] && subbox.pbc[_y_] && subbox.pbc[_z_]);
#endif


  for (int turn=START_TURN; turn<2; turn++)
    {

#ifndef CLASSIC_FRAGMENTATION
      if (!turn)
	{

	  /* it creates the map by setting to 1 the well-resolved region */
	  if (!ThisTask)
	    printf("[%s] Creating map of needed particles\n",fdate());

	  if (create_map())
	    return 1;

	  /* the rest of the first turn is skipped if there are PBCs in all directions */
	  if (all_pbc)
	    continue;
	}
      else if (!all_pbc)
	{

	  /* it updates the map by adding spheres around halos in the boundary layer */
	  if (!ThisTask)
	    printf("[%s] Updating map of needed particles\n",fdate());

	  update_map(nadd);

	  mynadd[0]=nadd[0];
	  mynadd[1]=nadd[1];
	  MPI_Reduce(mynadd, nadd_all, 2, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

	  if (!ThisTask)
	    {
	      printf("[%s] Requesting %Ld particles from the boundary layer\n",fdate(),nadd_all[0]);
	      if (nadd_all[1])
		printf("WARNING: %Ld requested particles lie beyond the boundary layer, some halos may be inaccurate\n",nadd_all[1]);
	    }
	}

      /* redistribution of products for queried particles */
      if (!ThisTask)
	printf("[%s] Starting %s re-distribution of products\n",fdate(),(turn?"second":"first"));
#else

      if (!ThisTask)
	printf("[%s] Starting re-distribution of products\n",fdate());

#endif

      tmp=MPI_Wtime();

      /* distribution of products from the fft space to the subbox space */
      map_to_be_used=0;
      if (!turn)
	subbox.Nneeded=0;
      if (distribute())
	return 1;

      tmp=MPI_Wtime()-tmp;
      cputime.distr += tmp;

      if (!ThisTask)
	printf("[%s] Re-distribution of products done, cputime = %14.6f\n", fdate(), tmp);

#ifndef CLASSIC_FRAGMENTATION
      if (subbox.Nneeded>subbox.Nalloc)
	{
	  if (turn)
	    {
	      printf("CRITICAL WARNING in Task %d: it allocated %d but received %d particles\n",
		     ThisTask, subbox.Nalloc, subbox.Nneeded);
	      printf("The required overhead is %f\n",(float)subbox.Nneeded/(float)MyGrids[0].ParticlesPerTask);
	      printf("Please increase MaxMemPerParticle by at least %d and start again\n",
		     (int)((float)(subbox.Nneeded-subbox.Nalloc)/(float)MyGrids[0].ParticlesPerTask * 
			   (sizeof(product_data) + FRAGFIELDS * sizeof(int)))+1+(int)params.MaxMemPerParticle);
	      if (params.ExitIfExtraParticles)
		return 1;
	    }
	  else
	    {
	      printf("ERROR in Task %d: the number of allocated particles (%d) is WAY too small! I need at least %d\n",
		     ThisTask, subbox.Nalloc, subbox.Nneeded);
	      printf("The required overhead is %f\n",(float)subbox.Nneeded/(float)MyGrids[0].ParticlesPerTask);
	      printf("Please increase MaxMemPerParticle to at least %d and start again\n",
		     (int)((float)(subbox.Nneeded-subbox.Nalloc)/(float)MyGrids[0].ParticlesPerTask * 
			   (sizeof(product_data) + FRAGFIELDS * sizeof(int)))+1+(int)params.MaxMemPerParticle);
	      return 1;
	    }
	}

      /* memory checks on the redistributed particles */
      mynadd[0]=subbox.Nstored;
      MPI_Reduce(mynadd, nadd_all, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

      if (!ThisTask)
	printf("[%s] %s re-distribution of Fmax done, %Ld particles stored by all tasks, average overhead: %f, cputime = %14.6f\n",
	       fdate(), (turn?"Second":"First"), nadd_all[0], 
	       (float)nadd_all[0]/(float)MyGrids[0].Ntotal, tmp);

      mynadd[0]=subbox.Nstored;
      MPI_Reduce(mynadd, nadd_all  , 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
      MPI_Reduce(mynadd, nadd_all+1, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

      if (!ThisTask)
	printf("[%s] Smallest and largest overhead: %f, %f\n",fdate(),
	       (float)nadd_all[0]/(float)MyGrids[0].ParticlesPerTask,
	       (float)nadd_all[1]/(float)MyGrids[0].ParticlesPerTask);

      /* sets or updates the map of potentially loaded particles */
      if (!turn)
	for (int i=0; i<subbox.maplength; i++)
	  frag_map[i]=frag_map_update[i];
      else
	for (int i=0; i<subbox.maplength; i++)
	  frag_map[i]|=frag_map_update[i];

#ifdef DEBUG
      write_map(turn);
#endif

      /* this part sorts particles and reorganizes index arrays */
      sort_and_organize();
#else

      /* sorting of particles according to their collapse time */
      tmp=MPI_Wtime();
      if (!ThisTask)
	printf("[%s] Starting sorting (parallel radix sort)\n",fdate());

      parallel_radix_sort_by_fmax_desc(frag, indices, indicesY,
				       (unsigned int *)sorted_pos, group_ID,
				       subbox.Npart);

      tmp=MPI_Wtime()-tmp;
      cputime.sort+=tmp;
      if (!ThisTask)
	printf("[%s] Sorting done, total cputime = %14.6f\n",fdate(),tmp);

#endif

      /* counts the number of peaks in the sub-volume to know the number of groups */
      Npeaks=count_peaks(&Ngood);

      mynadd[0]=Ngood;
      mynadd[1]=Npeaks;
      MPI_Reduce(mynadd, nadd_all, 2, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

      if (!ThisTask)
	{
	  printf("[%s] Task 0 found %d peaks, %d in the well resolved region. Total number of peaks: %Ld\n",
		 fdate(),Npeaks,Ngood,nadd_all[0]);
	}

       /* the number of peaks was supposed to be at most 1/10 of the number of particles
	 (we need space for Npeaks+2 groups) */
      if (Npeaks+2 > subbox.PredNpeaks)
	{
	  printf("ERROR on task %d: the number of peaks %d exceeds the predicted one (%d)\n",
		 ThisTask,Npeaks,subbox.PredNpeaks);
	  printf("      Please increase PredPeakFactor and restart\n");
	  fflush(stdout);
	  return 1;
	}

      if (turn)
	{
	  /* Gives the minimal PredPeakFactor */
	  BestPredPeakFactor=(double)nadd_all[1]/(double)MyGrids[0].ParticlesPerTask*6.0;
	}


      /* sets arrays to zero */
      memset(group_ID,0,subbox.Nalloc*sizeof(int));
      memset(linking_list,0,subbox.Nalloc*sizeof(int));
      memset(groups,0,subbox.PredNpeaks*sizeof(group_data));
      memset(wheretoplace_mycat,0,subbox.PredNpeaks*sizeof(histories_data));
#ifdef PLC
      memset(plcgroups,0,plc.Nmax*sizeof(plcgroup_data));
#endif


#ifndef CLASSIC_FRAGMENTATION
      if (!turn)
	{

	  /* quick fragmentation is done in a shot */
	  tmp=MPI_Wtime();

	  /* quickly generates the group catalogue */
	  if (quick_build_groups(Npeaks))
	  return 1;

	  tmp=MPI_Wtime()-tmp;
	  cputime.group+=tmp;
	  if (!ThisTask)
	    printf("[%s] Quick fragmentation done, cputime = %14.6f\n",fdate(),tmp);

	}
      else
#endif
	{
	  /* full fragmentation is done segmenting the redshift interval */

#ifdef USE_FASTFRAG
	  /* FastFrag 8-pass subvolume fragmentation */
	  tmp=MPI_Wtime();
	  if (!ThisTask)
	    printf("[%s] Starting FastFrag 8-pass subvolume fragmentation\n",fdate());

	  if (fragment_fastfrag())
	    return 1;

	  tmp=MPI_Wtime()-tmp;
	  cputime.group+=tmp;
	  if (!ThisTask)
	    printf("[%s] FastFrag fragmentation done, cputime = %14.6f\n",fdate(),tmp);

#else /* !USE_FASTFRAG */

	  /* Initialize GFLUT tables for thread-safe growth factor lookups.
	     The range covers z=0 to the maximum Fmax in the particle list. */
	  {
	    double gflut_zmin = 0.0;
	    double gflut_zmax = (double)frag[0].Fmax - 1.0;
	    if (gflut_zmax < 1.0) gflut_zmax = 1.0;  /* safety floor */
	    gflut_init(gflut_zmin, gflut_zmax);
	    gflut_validate();
	  }

	  for (int mysegment=0; mysegment<ScaleDep.nseg; mysegment++)
	    {
	      ScaleDep.myseg=mysegment;

#ifdef RECOMPUTE_DISPLACEMENTS
	      /* The first segment uses only the first computed velocity
		 The second segment has the two velocities already loaded */
	      if (mysegment>=1)
		{
		  /* computation of displacements for the next redshift */
		  if (!ThisTask)
		    printf("\n[%s] Computing displacements for redshift %f\n",fdate(),ScaleDep.z[mysegment]);
		  if (shift_all_displacements())
		    return 1;

		  if (compute_displacements(0,0,ScaleDep.z[mysegment]))
		    return 1;

		  tmp=MPI_Wtime();
		  if (!ThisTask)
		    printf("[%s] Starting re-distribution of products\n",fdate());

		  map_to_be_used=1;
		  subbox.Nneeded=0;
		  if (distribute())
		    return 1;
		  sort_and_organize();

		  tmp=MPI_Wtime()-tmp;
		  cputime.distr += tmp;
		  if (!ThisTask)
		    printf("[%s] Re-distribution of Fmax products done, cputime = %14.6f\n",fdate(),tmp);

		  if (recompute_group_velocities())
		    return 1;
		}
#endif

	      /* Pre-compute Eulerian positions and sigmaD for OpenMP parallel path */
	      precomp_allocate(subbox.Nstored);
	      precompute_particles();

	      tmp=MPI_Wtime();

	      if (build_groups(Npeaks,ScaleDep.z[mysegment],(mysegment==0)))
		return 1;

	      /* Free pre-computed data after each segment */
	      precomp_free();

	      tmp=MPI_Wtime()-tmp;
	      if (!ThisTask)
		printf("[%s] Fragmentation done to redshift %6.4f, cputime = %14.6f\n",
		       fdate(),ScaleDep.z[mysegment],tmp);
	      cputime.group+=tmp;
	    }
#endif /* USE_FASTFRAG */
	}
    }

  /* cpu time for fragmentation */
  cputime.frag = MPI_Wtime() - cputime.frag;

  if (!ThisTask)
    printf("[%s] Finishing fragment, total cputime = %14.6f\n",fdate(),cputime.frag);

#ifdef SNAPSHOT
  if (params.WriteTimelessSnapshot)
    {
      /* redistribution of products */
      if (!ThisTask)
	printf("[%s] Starting to distribute back accretion redshifts\n",fdate());
      tmp=MPI_Wtime();

      if (distribute_back())
	return 1;

      tmp=MPI_Wtime()-tmp;
      cputime.distr += tmp;

      if (!ThisTask)
	printf("[%s] back-distribution of zacc done, cputime = %14.6f\n", fdate(), tmp);

      if (write_timeless_snapshot())
	return 1;
    }
#endif

  if (!ThisTask)
    {
      printf("\n");
      printf("[%s] The PredPeakFactor parameter could have been %5.2f in place of %5.2f\n",
	     fdate(),BestPredPeakFactor,params.PredPeakFactor);
    }

  return 0;
}


void sort_and_organize(void)
{
  double tmp;
  int i;

  tmp=MPI_Wtime();
  if (!ThisTask)
    printf("[%s] Starting sorting (parallel radix sort)\n",fdate());

  /* Sort particles in order of descending Fmax.
     Uses group_ID as scratch buffer for indices (it will be zeroed later).
     Uses sorted_pos (cast to unsigned int) as scratch for keys.
     Uses indicesY as scratch for keys during double-buffering. */
  parallel_radix_sort_by_fmax_desc(frag, indices, indicesY,
				   (unsigned int *)sorted_pos, group_ID,
				   subbox.Nstored);

  /* inverse permutation needed by the in-place reorder */
  for (i=0; i<(int)subbox.Nstored; i++)
    indicesY[indices[i]]=i;

  /* reorder the frag data structure and frag_pos in order of descending Fmax */
  reorder(indicesY, subbox.Nstored);

  /* Build direct lookup table: sorted_pos[grid_position] = fmax_order_index.
     This replaces the previous Sort 2 + bsearch approach with O(1) access.
     After this, find_location() returns the fmax-order index directly,
     eliminating the need for both the position sort and the indices[] indirection.
     Requires Nalloc >= Npart so that sorted_pos can hold Npart entries. */
  if (subbox.Nalloc < subbox.Npart)
    {
      printf("ERROR on Task %d: Nalloc (%u) < Npart (%u), cannot build direct lookup table.\n",
	     ThisTask, subbox.Nalloc, subbox.Npart);
      printf("       Please increase MaxMemPerParticle.\n");
      fflush(stdout);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

  memset(sorted_pos, -1, subbox.Npart * sizeof(int));
  for (i=0; i<(int)subbox.Nstored; i++)
    sorted_pos[frag_pos[i]] = i;

  tmp=MPI_Wtime()-tmp;
  cputime.sort+=tmp;
  if (!ThisTask)
    printf("[%s] Sorting done, total cputime = %14.6f\n",fdate(),tmp);

}


static int compare_search(const void * a,const void * b)
{
  if ( *((int *)a) == *((int *)b) )
    return 0;
  else
    if ( *((int *)a) < *((int *)b) )
      return -1;
    else
      return 1;
}

/* reorders frag_pos and frag after they have been index-sorted */
void reorder(int *ind, int n) 
{
  int i,oldf,next,tmp;
  product_data oldF,tmpF;

  memset(&oldF,0,sizeof(product_data));
  memset(&tmpF,0,sizeof(product_data));
  for (i=0; i<n; i++)
    {
      if (ind[i]!=i)
	{
	  oldf=frag_pos[i];
	  memcpy(&oldF,&frag[i],sizeof(product_data));
	  next=ind[i];
	  while (next != i) 
	    {
	      tmp=frag_pos[next];	    
	      frag_pos[next]=oldf;
	      oldf=tmp;
	      memcpy(&tmpF,&frag[next],sizeof(product_data));
	      memcpy(&frag[next],&oldF,sizeof(product_data));
	      memcpy(&oldF,&tmpF,sizeof(product_data));
	      tmp=ind[next];
	      ind[next]=next;
	      next=tmp;
	    }
	  frag_pos[i]=oldf;
	  memcpy(&frag[i],&oldF,sizeof(product_data));
	  ind[i]=next;
	}
    } 
}

/* reorders only frag_pos after it has been index-sorted */
void reorder_nofrag(int *ind, int n) 
{
  int i,oldf,next,tmp;

  for (i=0; i<n; i++)
    {
      if (ind[i]!=i)
	{
	  oldf=sorted_pos[i];
	  next=ind[i];
	  while (next != i) 
	    {
	      tmp=sorted_pos[next];
	      sorted_pos[next]=oldf;
	      oldf=tmp;
	      tmp=ind[next];
	      ind[next]=next;
	      next=tmp;
	    }
	  sorted_pos[i]=oldf;
	  ind[i]=next;
	}
    } 
}


int find_location(int i,int j,int k)
{
  /* Returns the fmax-order index of the particle at grid position (i,j,k),
     or -1 if the particle is not stored.
     Uses the direct lookup table built in sort_and_organize(). */

  return sorted_pos[COORD_TO_INDEX(i,j,k,subbox.Lgwbl)];
}


int count_peaks(int *ngood)
{

  /* counts the number of Fmax peaks down to the final redshift
     to know the total number of groups */

  int iz,i,j,k,nn,ngroups_tot,peak_cond,i1,j1,k1;

#ifdef DEBUG
  FILE *fd=fopen("peaks.dat","w");
#endif

  ngroups_tot=0;     /* number of groups, group 1 is the filament group */
  *ngood=0;          /* number of groups out of the safety boundary */

  for (iz=0; iz<subbox.Nstored; iz++)
    {
      /* position on the local box */
#ifdef CLASSIC_FRAGMENTATION
      INDEX_TO_COORD(iz,i,j,k,subbox.Lgwbl);
#else
      INDEX_TO_COORD(frag_pos[iz],i,j,k,subbox.Lgwbl);
#endif

      /* avoid borders */
      if ( !subbox.pbc[_x_] && (i==0 || i==subbox.Lgwbl[_x_]-1) ) continue;
      if ( !subbox.pbc[_y_] && (j==0 || j==subbox.Lgwbl[_y_]-1) ) continue;
      if ( !subbox.pbc[_z_] && (k==0 || k==subbox.Lgwbl[_z_]-1) ) continue;

      /* peak condition */
      peak_cond=1;
      for (nn=0; nn<6; nn++)
	{
	  switch (nn)
	    {
	    case 0:
	      i1=( subbox.pbc[_x_] && i==0 ? subbox.Lgwbl[_x_]-1 : i-1 );
	      j1=j;
	      k1=k;
	      break;
	    case 1:
	      i1=( subbox.pbc[_x_] && i==subbox.Lgwbl[_x_]-1 ? 0 : i+1 );
	      j1=j;
	      k1=k;
	      break;
	    case 2:
	      i1=i;
	      j1=( subbox.pbc[_y_] && j==0 ? subbox.Lgwbl[_y_]-1 : j-1 );
	      k1=k;
	      break;
	    case 3:
	      i1=i;
	      j1=( subbox.pbc[_y_] && j==subbox.Lgwbl[_y_]-1 ? 0 : j+1 );
	      k1=k;
	      break;
	    case 4:
	      i1=i;
	      j1=j;
	      k1=( subbox.pbc[_z_] && k==0 ? subbox.Lgwbl[_z_]-1 : k-1 );
	      break;
	    case 5:
	      i1=i;
	      j1=j;
	      k1=( subbox.pbc[_z_] && k==subbox.Lgwbl[_z_]-1 ? 0 : k+1 );
	      break;
	    }

#ifdef CLASSIC_FRAGMENTATION
	  peak_cond &= (frag[iz].Fmax > frag[COORD_TO_INDEX(i1,j1,k1,subbox.Lgwbl)].Fmax);
#else
	  /* looks for the neighbouring particle in the list */
	  int pos = find_location(i1,j1,k1);
	  if (pos>=0)
	    peak_cond &= (frag[iz].Fmax > frag[pos].Fmax);
#endif

	  if (!peak_cond)
	    break;

	}

      if (peak_cond)
	{
	  ngroups_tot++;
	  if ( i>=subbox.safe[_x_] && i<subbox.Lgwbl[_x_]-subbox.safe[_x_] &&
	       j>=subbox.safe[_y_] && j<subbox.Lgwbl[_y_]-subbox.safe[_y_] &&
	       k>=subbox.safe[_z_] && k<subbox.Lgwbl[_z_]-subbox.safe[_z_])
	    (*ngood)++;
#ifdef DEBUG
	  fprintf(fd," %2d %2d %2d   %12.10f\n",i,j,k,frag[iz].Fmax);
#endif
	}
    }

#ifdef DEBUG
  fclose(fd);
#endif

  return ngroups_tot;
}


int create_map()
{
  int i,j,k,i1,i2,j1,j2,k1,k2;

  /* sets to 1 all particles in the well-resolved region plus one row for each side (without PBCs) */
  memset(frag_map_update, 0, subbox.maplength*sizeof(unsigned int));
  if (!subbox.pbc[_x_])
    {
      i1=subbox.safe[_x_]-1;
      i2=subbox.Lgrid[_x_]+subbox.safe[_x_]+1;
    }
  else
    {
      i1=0;
      i2=subbox.Lgrid[_x_];
    }
  if (!subbox.pbc[_y_])
    {
      j1=subbox.safe[_y_]-1;
      j2=subbox.Lgrid[_y_]+subbox.safe[_y_]+1;
    }
  else
    {
      j1=0;
      j2=subbox.Lgrid[_y_];
    }
  if (!subbox.pbc[_z_])
    {
      k1=subbox.safe[_z_]-1;
      k2=subbox.Lgrid[_z_]+subbox.safe[_z_]+1;
    }
  else
    {
      k1=0;
      k2=subbox.Lgrid[_z_];
    }

  for (i=i1; i<i2; i++)
    for (j=j1; j<j2; j++)
      for (k=k1; k<k2; k++)
	set_mapup_bit(i,j,k);

  return 0;
}

void set_mapup_bit(int i, int j, int k)
{
  /* this operates on frag_map_update */
  unsigned int pos = COORD_TO_INDEX(i,j,k,subbox.Lgwbl);
  //i + (j + k*subbox.Lgwbl[_y_])*subbox.Lgwbl[_x_];
  frag_map_update[pos/UINTLEN]|=(1<<pos%UINTLEN);
}

int get_mapup_bit(unsigned int pos)
{
  /* this operates on frag_map_update */
  /* unsigned int pos = i + (j + k*subbox.Lgwbl[_y_])*subbox.Lgwbl[_x_]; */
  unsigned int rem = pos%UINTLEN;
  return (frag_map_update[pos/UINTLEN] & (1<<rem))>>rem;
}

int get_map_bit(unsigned int pos)
{
  /* this operates on frag_map */
  /* unsigned int pos = i + (j + k*subbox.Lgwbl[_y_])*subbox.Lgwbl[_x_]; */
  unsigned int rem = pos%UINTLEN;
  return (frag_map[pos/UINTLEN] & (1<<rem))>>rem;
}

int get_map_bit_coord(int i, int j, int k)
{
  /* this operates on frag_map */
  unsigned int pos = COORD_TO_INDEX(i,j,k,subbox.Lgwbl);
  //i + (j + k*subbox.Lgwbl[_y_])*subbox.Lgwbl[_x_];
  unsigned int rem = pos%UINTLEN;
  return (frag_map[pos/UINTLEN] & (1<<rem))>>rem;
}

void write_map(int turn)
{
  int i,j,k;
  unsigned int pos,rem;
  char fname[LBLENGTH];
  sprintf(fname,"map_task%d_turn%d.txt",ThisTask,turn);
  FILE *fd=fopen(fname,"w");
  /* this loop is not following memory... */
  for (k=0; k<subbox.Lgwbl[_z_]; k++)
    {
      fprintf(fd,"k=%d\n",k);
      for (j=0; j<subbox.Lgwbl[_y_]; j++)
	{
	  for (i=0; i<subbox.Lgwbl[_x_]; i++)
	    {
	      pos = COORD_TO_INDEX(i,j,k,subbox.Lgwbl); 
	      //i + (j + k*subbox.Lgwbl[_y_])*subbox.Lgwbl[_x_];
	      rem = pos%UINTLEN;
	      fprintf(fd,"%1d",(frag_map[pos/UINTLEN] & (1<<rem))>>rem);
	    }
	  fprintf(fd,"\n");
	}
    }
  fclose(fd);
  sprintf(fname,"mapup_task%d_turn%d.txt",ThisTask,turn);
  fd=fopen(fname,"w"); 
  for (k=0; k<subbox.Lgwbl[_z_]; k++)
    {
      fprintf(fd,"k=%d\n",k);
      for (j=0; j<subbox.Lgwbl[_y_]; j++)
	{
	  for (i=0; i<subbox.Lgwbl[_x_]; i++)
	    {
	      pos =  COORD_TO_INDEX(i,j,k,subbox.Lgwbl); 
	      //i + (j + k*subbox.Lgwbl[_y_])*subbox.Lgwbl[_x_];
	      rem = pos%UINTLEN;
	      fprintf(fd,"%1d",(frag_map_update[pos/UINTLEN] & (1<<rem))>>rem);
	    }
	  fprintf(fd,"\n");
	}
    }
  fclose(fd);
}



#ifdef RECOMPUTE_DISPLACEMENTS

int shift_all_displacements()
{
  /* Shifts all Vel to Vel_prev, at all orders */

  /* This shifts Vel_prev to Vel in the fft space */
  for (int i=0; i<MyGrids[0].total_local_size; i++)
    for (int ia=0; ia<3; ia++)
      {
	products[i].Vel_prev[ia]=products[i].Vel[ia];
#ifdef TWO_LPT
	products[i].Vel_2LPT_prev[ia]=products[i].Vel_2LPT[ia];
#ifdef THREE_LPT
	products[i].Vel_3LPT_1_prev[ia]=products[i].Vel_3LPT_1[ia];
	products[i].Vel_3LPT_2_prev[ia]=products[i].Vel_3LPT_2[ia];
#endif
#endif
      }
  return 0;
}

int recompute_group_velocities()
{
  /* Recompute average displacements of group velocities */
  int next,npart,i,ia;
  for (i=FILAMENT+1; i<=ngroups; i++)
    if (groups[i].point > 0)
      {
	for (ia=0; ia<3; ia++)
	  {
	    groups[i].Vel_prev[ia] = groups[i].Vel[ia] = 0;
#ifdef TWO_LPT
	    groups[i].Vel_prev[ia] = groups[i].Vel_2LPT[ia] = 0.;
#ifdef THREE_LPT
	    groups[i].Vel_3LPT_1_prev[ia] = groups[i].Vel_3LPT_1[ia] =
	      groups[i].Vel_3LPT_2_prev[ia] = groups[i].Vel_3LPT_2[ia] = 0;
#endif
#endif
	  }

	next=groups[i].point;
	for (npart=0; npart<groups[i].Mass; npart++)
	  {
	    for (ia=0; ia<3; ia++)
	      {
		groups[i].Vel[ia]+=frag[next].Vel[ia];
		groups[i].Vel_prev[ia]+=frag[next].Vel_prev[ia];
#ifdef TWO_LPT
		groups[i].Vel_2LPT[ia]+=frag[next].Vel_2LPT[ia];
		groups[i].Vel_2LPT_prev[ia]+=frag[next].Vel_2LPT_prev[ia];
#ifdef THREE_LPT
		groups[i].Vel_3LPT_1[ia]+=frag[next].Vel_3LPT_1[ia];
		groups[i].Vel_3LPT_1_prev[ia]+=frag[next].Vel_3LPT_1_prev[ia];
		groups[i].Vel_3LPT_2[ia]+=frag[next].Vel_3LPT_2[ia];
		groups[i].Vel_3LPT_2_prev[ia]+=frag[next].Vel_3LPT_2_prev[ia];
#endif
#endif
	      }
	    next=linking_list[next];
	  }
	for (ia=0; ia<3; ia++)
	  {
	    groups[i].Vel[ia] /= (double)groups[i].Mass;
	    groups[i].Vel_prev[ia] /= (double)groups[i].Mass;
#ifdef TWO_LPT
	    groups[i].Vel_2LPT[ia] /= (double)groups[i].Mass;
	    groups[i].Vel_2LPT_prev[ia] /= (double)groups[i].Mass;
#ifdef THREE_LPT
	    groups[i].Vel_3LPT_1[ia] /= (double)groups[i].Mass;
	    groups[i].Vel_3LPT_1_prev[ia] /= (double)groups[i].Mass;
	    groups[i].Vel_3LPT_2[ia] /= (double)groups[i].Mass;
	    groups[i].Vel_3LPT_2_prev[ia] /= (double)groups[i].Mass;
#endif
#endif
	  }
      }

  return 0;
}


#endif


double myz;
gsl_function Function;

double Integrand_MF(double logm, void *param)
{
  double m=exp(logm);
  return m * AnalyticMassFunction(m,myz);
}

double compute_Nhalos_in_PLC(double z1, double z2)
{
  /* analytic prediction of the number of halos in the PLC */

  double MinMass    = log(params.ParticleMass*params.MinHaloMass);
  double delta_z    = 0.01;
  double number     = 0;
  double solidangle = (1-cos( (params.PLCAperture>90. ? 90. : params.PLCAperture)  /180.*PI) )*2.*PI;
  
  double result, error, upper, lower;
  gsl_function Function;
  
  Function.function = &Integrand_MF;
  lower=z1;
  do
    {
      upper=lower+delta_z;
      if (upper>z2)
	upper=z2;
      myz = 0.5*(upper+lower);

      gsl_integration_qags(&Function, MinMass, 37.0, 0.0, TOLERANCE, NWINT, workspace, &result, &error);

      number += result * solidangle * (pow(ComovingDistance(upper),3.) -
				       pow(ComovingDistance(lower),3.)) /3.;
      lower+=delta_z;
    } 
  while (upper<z2);

  return number;
}

int size2Mb(double *s)
{
  *s /= MBYTE;
  int gb=0;
  if (*s>1024.)
    {
      *s /= 1024.;
      gb=1;
    }
  return gb;
}

int estimate_file_size(void)
{

  /* estimates the size of output files */

  /* only Task 0 needs to work here */
  if (ThisTask)
    return 0;

  /* this works for binary output */
  if (params.CatalogInAscii)
    return 0;

  double       result, error, total=0.0, size, number;
  int          gb;
  gsl_function Function;
  
  printf("ESTIMATED STORAGE REQUIREMENTS:\n");

  double MinMass=log(params.ParticleMass*params.MinHaloMass);
  Function.function = &Integrand_MF;
  for (int iout=0; iout<outputs.n; iout++)
    {

      myz=outputs.z[iout];
      gsl_integration_qags(&Function, MinMass, 37.0, 0.0, TOLERANCE, NWINT, workspace, &result, &error);
      number = result * pow(params.BoxSize_htrue,3.);
      size = number * sizeof(catalog_data);
      total+=size;
      gb=size2Mb(&size);
      printf("catalog, z=%6.4f, number of halos: %d, size: %f %s",outputs.z[iout],(int)number, size,(gb?"Gbyte":"Mbyte"));
      if (params.NumFiles>1)
	printf(" - each file will have a size of %f %s",size/(double)params.NumFiles,(gb?"Gbyte":"Mbyte"));
      printf("\n");
    }

  size = number * sizeof(catalog_data) * 1.4;
  total += size;
  gb=size2Mb(&size);
  printf("order-of-magnitude size of histories file: %f %s",1.4*size,(gb?"Gbyte":"Mbyte"));
  if (params.NumFiles>1)
    printf(" - each file will have a size of %f %s",size/(double)params.NumFiles,(gb?"Gbyte":"Mbyte"));
  printf("\n");

#ifdef PLC
  
  if (params.StartingzForPLC>0.)
    {
      number = compute_Nhalos_in_PLC(params.LastzForPLC, params.StartingzForPLC);
      size = number * sizeof(plc_write_data);
      total+=size;
      gb=size2Mb(&size);
      printf("past light cone, number of halos: %d, size: %f %s",(int)number, size,(gb?"Gbyte":"Mbyte"));
      if (params.NumFiles>1)
	printf(" - each file will have a size of %f %s",size/(double)params.NumFiles,(gb?"Gbyte":"Mbyte"));
      printf("\n");

    }
#endif

#ifdef LONGIDS
  double IDsize = (double)MyGrids[0].Ntotal * 8.;
#else
  double IDsize = (double)MyGrids[0].Ntotal * 4.;
#endif

  int nblo=3;
#ifndef TWO_LPT
  int nvel=3;
  nblo+=1;
#else
#ifndef THREE_LPT
  int nvel=6;
  nblo+=2;
#else
  int nvel=12;
  nblo+=4;
#endif
#endif

#ifdef ADD_RMAX_TO_SNAPSHOT
  nblo+=1;
#endif


  if (params.WriteTimelessSnapshot)
    {
      size=268. + IDsize + 6. + (nvel+2) * ((double)MyGrids[0].Ntotal * sizeof(float) + 6.) + nblo*40 + 6.;
#ifdef ADD_RMAX_TO_SNAPSHOT
      size+=((double)MyGrids[0].Ntotal * sizeof(float) + 6.);
#endif
      total+=size;
      gb=size2Mb(&size);
      printf("timeless snapshot size: %f %s", size, (gb?"Gbyte":"Mbyte"));
      if (params.NumFiles>1)
	printf(" - each file will have a size of %f %s",size/(double)params.NumFiles,(gb?"Gbyte":"Mbyte"));
      printf("\n");
    }

  gb=size2Mb(&total);

  printf("Total storage: %f %s\n",total,(gb?"Gbyte":"Mbyte"));

  return 0;
}


#ifdef USE_FASTFRAG

/* ============================================================
   FastFrag: 8-pass subvolume fragmentation
   Ported from src_fastfrag/fragment.c with debug I/O removed.
   ============================================================ */

/* Macro for periodic/non-periodic boundary coordinate clamping */
#define SET_PBC_FF(I,F,L) ( (F?((I)+(L))%(L):(I)) )

/* comparison function for sorting global group list by t_peak (descending) */
static int ff_index_compare_F(const void *a, const void *b)
{
  double ta = groups[*((const int *)a)].t_peak;
  double tb = groups[*((const int *)b)].t_peak;
  if (ta == tb) return 0;
  return (ta > tb) ? -1 : 1;
}

/**
 * @brief Count Fmax peaks inside a subvolume (excluding the 1-particle border).
 *
 * @param[in] v  Pointer to the initialised volume_data structure.
 * @return Number of local maxima of Fmax above outputs.Flast.
 */
int count_peaks_v(volume_data *v)
{
  int i, j, k, iz, nn, i1, j1, k1;
  int Npeaks = 0;

  for (iz = 0; iz < (int)v->Npart; iz++)
    {
      INDEX_TO_COORD(iz, i, j, k, v->GridSize);

      /* skip border particles — the border is the overlap region */
      if (i == 0 || i == v->GridSize[_x_] - 1) continue;
      if (j == 0 || j == v->GridSize[_y_] - 1) continue;
      if (k == 0 || k == v->GridSize[_z_] - 1) continue;

      /* skip particles that collapse too late */
      if (v->Frag[iz].Fmax < outputs.Flast) continue;

      int peak_cond = 1;
      for (nn = 0; nn < 6; nn++)
        {
          switch (nn)
            {
            case 0: i1=i-1; j1=j;   k1=k;   break;
            case 1: i1=i+1; j1=j;   k1=k;   break;
            case 2: i1=i;   j1=j-1; k1=k;   break;
            case 3: i1=i;   j1=j+1; k1=k;   break;
            case 4: i1=i;   j1=j;   k1=k-1; break;
            case 5: i1=i;   j1=j;   k1=k+1; break;
            default: i1=i; j1=j; k1=k; break;
            }
          peak_cond &= (v->Frag[iz].Fmax >
                        v->Frag[COORD_TO_INDEX(i1, j1, k1, v->GridSize)].Fmax);
          if (!peak_cond) break;
        }
      if (peak_cond) Npeaks++;
    }
  return Npeaks;
}


/**
 * @brief Convert a volume-local particle index to a subbox index.
 *
 * @param[in] pv        Volume-local flat index.
 * @param[in] my_volume Pointer to the volume_data structure.
 * @return Flat index in the subbox coordinate frame.
 */
static int volume2subbox(int pv, volume_data *my_volume)
{
  int iv, jv, kv;
  INDEX_TO_COORD(pv, iv, jv, kv, my_volume->GridSize);

  int is = SET_PBC_FF(iv + my_volume->Start[_x_],
                      subbox.pbc[_x_], subbox.Lgwbl[_x_]);
  int js = SET_PBC_FF(jv + my_volume->Start[_y_],
                      subbox.pbc[_y_], subbox.Lgwbl[_y_]);
  int ks = SET_PBC_FF(kv + my_volume->Start[_z_],
                      subbox.pbc[_z_], subbox.Lgwbl[_z_]);

  return COORD_TO_INDEX(is, js, ks, subbox.Lgwbl);
}


/**
 * @brief Initialise a subvolume from a slice of the global subbox data.
 *
 * The volume arrays are carved out of @p buf (pre-allocated by the caller).
 * The function fills Frag[] by copying from the global frag[] with periodic
 * or reflective boundary handling, and sets Npeaks / Ngroups.
 *
 * @param[in]  ic,jc,kc       Origin of the sub-cube in subbox coordinates.
 * @param[in]  sizex,y,z      Dimensions of the sub-cube (including 1-cell border).
 * @param[out] my_volume      Volume descriptor to populate.
 * @param[in]  buf            Pre-allocated memory buffer.
 * @param[in]  bufsz          Size of buf in bytes (for safety check).
 * @return 0 on success, 1 on error.
 */
int initialize_volume(int ic, int jc, int kc,
                      int sizex, int sizey, int sizez,
                      volume_data *my_volume,
                      char *buf, size_t bufsz)
{
  my_volume->Start[_x_] = ic;
  my_volume->Start[_y_] = jc;
  my_volume->Start[_z_] = kc;

  my_volume->GridSize[0] = sizex;
  my_volume->GridSize[1] = sizey;
  my_volume->GridSize[2] = sizez;
  my_volume->Npart = (unsigned int)sizex * sizey * sizez;

  /* Carve out array pointers from the buffer */
  size_t need = (size_t)my_volume->Npart *
                (sizeof(product_data) + 2 * sizeof(int) + sizeof(group_data));
  if (need > bufsz)
    {
      if (!ThisTask)
        printf("ERROR [initialize_volume]: buffer too small: need %zu, have %zu\n",
               need, bufsz);
      return 1;
    }

  memset(buf, 0, need);

  size_t off = 0;
  my_volume->Frag         = (product_data *)(buf + off);
  off += (size_t)my_volume->Npart * sizeof(product_data);
  my_volume->Group_ID     = (int *)(buf + off);
  off += (size_t)my_volume->Npart * sizeof(int);
  my_volume->Linking_list = (int *)(buf + off);
  off += (size_t)my_volume->Npart * sizeof(int);
  my_volume->Groups       = (group_data *)(buf + off);

  /* Copy frag data from subbox into volume, honouring PBCs.
   *
   * After sort_and_organize(), frag[] is in Fmax-descending order:
   *   frag[i]     = product data of the i-th highest-Fmax particle
   *   frag_pos[i] = grid position (subbox index) of that particle
   *   sorted_pos[grid_pos] = time-order index of particle at grid_pos (-1 if absent)
   *
   * To get the product data for a particle at subbox grid position posb, use
   *   sorted_pos[posb] as the index into frag[].
   * If sorted_pos[posb] == -1 the particle is outside the stored boundary: leave Fmax=0.
   */
  for (int iv = 0; iv < sizex; iv++)
    {
      int ib = SET_PBC_FF(iv + ic, subbox.pbc[_x_], subbox.Lgwbl[_x_]);
          /* Skip cells that fall outside the subbox (non-PBC edge tasks) */
          if (ib < 0 || ib >= subbox.Lgwbl[_x_]) continue;
          for (int jv = 0; jv < sizey; jv++)
            {
              int jb = SET_PBC_FF(jv + jc, subbox.pbc[_y_], subbox.Lgwbl[_y_]);
              if (jb < 0 || jb >= subbox.Lgwbl[_y_]) continue;
              for (int kv = 0; kv < sizez; kv++)
                {
                  int kb = SET_PBC_FF(kv + kc, subbox.pbc[_z_], subbox.Lgwbl[_z_]);
                  if (kb < 0 || kb >= subbox.Lgwbl[_z_]) continue;
                  int posv = COORD_TO_INDEX(iv, jv, kv, my_volume->GridSize);
                  int posb = COORD_TO_INDEX(ib, jb, kb, subbox.Lgwbl);
                  int ti   = sorted_pos[posb]; /* time-order index, or -1 */
                  if (ti >= 0)
                    memcpy(my_volume->Frag + posv, frag + ti, sizeof(product_data));
                  /* else: particle absent → Frag[posv] stays zeroed (Fmax=0) */
                }
            }
    }

  my_volume->Npeaks  = (unsigned int)count_peaks_v(my_volume);
  my_volume->Ngroups = my_volume->Npeaks + FILAMENT + 1;

  return 0;
}


/**
 * @brief Merge the group catalogue from a processed subvolume into the global one.
 *
 * For each halo in the volume catalogue that is safely far from the volume
 * border (distance / M^{1/3} > R_THR and distance > A_THR), its full merger
 * tree is transplanted into the global groups[] array.  Otherwise only the
 * Mass and distance fields are updated.
 *
 * @param[in,out] my_volume  The processed volume (Frag, Groups, Group_ID,
 *                            Linking_list already filled by build_groups_in_volume).
 * @return 0 on success.
 */
int merge_catalogs(volume_data *my_volume)
{
#define R_THR_FF 2
#define A_THR_FF 9

  int i, j, k, dist, dist2;

  /* Compute the minimum distance from the volume border for each halo */
  for (int pos = 0; pos < (int)my_volume->Npart; pos++)
    {
      if (my_volume->Group_ID[pos] > FILAMENT)
        {
          INDEX_TO_COORD(pos, i, j, k, my_volume->GridSize);
          dist = i;
          dist2 = my_volume->GridSize[_x_] - i - 1;
          if (dist2 < dist) dist = dist2;
          dist2 = j;
          if (dist2 < dist) dist = dist2;
          dist2 = my_volume->GridSize[_y_] - j - 1;
          if (dist2 < dist) dist = dist2;
          dist2 = k;
          if (dist2 < dist) dist = dist2;
          dist2 = my_volume->GridSize[_z_] - k - 1;
          if (dist2 < dist) dist = dist2;

          my_volume->Groups[my_volume->Group_ID[pos]].aux2 = dist;
        }
    }

  /* Sort the global group list in decreasing t_peak order.
   * Cached across calls: ngroups is constant after find_all_peaks() so the
   * order never changes.  Avoids O(N log N) sort on every merge_catalogs call
   * (critical for the ultimo loop which calls us ~48K times). */
  static int ff_sorted_ngroups = -1;
  if (ngroups != ff_sorted_ngroups)
    {
      for (int gi = 0; gi < ngroups; gi++)
        indices[gi] = gi;
      qsort((void *)indices, ngroups, sizeof(int), ff_index_compare_F);
      ff_sorted_ngroups = ngroups;
    }

  /* Match volume group list with global list; both sorted by t_peak */
  int p = 0;
  for (int gvol = FILAMENT + 1; gvol < (int)my_volume->Ngroups; gvol++)
    {
      /* Advance global pointer until we find the matching peak */
      while (p < ngroups &&
             groups[indices[p]].name != my_volume->Groups[gvol].name &&
             groups[indices[p]].t_peak >= my_volume->Groups[gvol].t_peak &&
             groups[indices[p]].t_peak >= outputs.Flast)
        p++;

      if (p < ngroups &&
          groups[indices[p]].name == my_volume->Groups[gvol].name)
        {
          /* Matching peak found — store global index in aux1 */
          my_volume->Groups[gvol].aux1 = indices[p];
        }
      else
        {
          /* Halo not in global list — this should not happen in a
             correctly constructed 8-pass tiling, but handle gracefully */
          if (ngroups < subbox.PredNpeaks)
            {
              my_volume->Groups[gvol].aux1 = ngroups;
              groups[ngroups].aux2  = 0;
              groups[ngroups].good  = 0;
              ngroups++;
            }
          else
            {
              /* No space left — silently skip */
              my_volume->Groups[gvol].aux1 = FILAMENT;
              continue;
            }
        }
    }

  /* Update the global catalogue from the volume catalogue */
  for (int gvol = FILAMENT + 1; gvol < (int)my_volume->Ngroups; gvol++)
    {
      int global = my_volume->Groups[gvol].aux1;

      /* Skip invalid matches */
      if (global <= FILAMENT || global >= ngroups)
        continue;

      /* Skip branches that have been merged into their main halo already */
      if (my_volume->Groups[gvol].halo_app != gvol)
        continue;

      /* Safety criterion: halo is well-resolved only if it is far
         enough from the volume border both in absolute and relative terms */
      int safe_halo = (my_volume->Groups[gvol].aux2 /
                       pow((double)my_volume->Groups[gvol].Mass, 1.0/3.0)
                       > R_THR_FF &&
                       my_volume->Groups[gvol].aux2 > A_THR_FF);

      if (safe_halo)
        {
          /* Transplant the full merger tree from volume to global catalogue */
          int next = gvol;
          do
            {
              int gnext = my_volume->Groups[next].aux1;
              /* Copy entire group_data structure */
              memcpy(groups + gnext, my_volume->Groups + next, sizeof(group_data));
              /* Fix linked-list pointers: translate volume indices to global */
              groups[gnext].ll =
                my_volume->Groups[my_volume->Groups[next].ll].aux1;
              groups[gnext].halo_app =
                my_volume->Groups[my_volume->Groups[next].halo_app].aux1;
              groups[gnext].merged_with =
                my_volume->Groups[my_volume->Groups[next].merged_with].aux1;
              groups[gnext].good = 1;

              next = my_volume->Groups[next].ll;
            }
          while (next != gvol);

          /* Rebuild particle linking list in subbox frame */
          int pv = my_volume->Groups[gvol].point;
          int ps = volume2subbox(pv, my_volume);
          groups[global].point = ps;
          do
            {
              if (pv == my_volume->Groups[gvol].bottom)
                groups[global].bottom = ps;

              /* Mark particle as processed.
               * group_ID is indexed by subbox grid position (ps).
               * frag[] is in time-order after sort_and_organize(): use
               * sorted_pos[ps] to reach the correct frag entry. */
              group_ID[ps] = global;
              {
                int ti = sorted_pos[ps];
                if (ti >= 0) frag[ti].Rmax = -1;
              }

              int pv_next = my_volume->Linking_list[pv];
              int ps_next = volume2subbox(pv_next, my_volume);
              linking_list[ps] = ps_next;

              pv = pv_next;
              ps = ps_next;
            }
          while (pv != my_volume->Groups[gvol].point);
        }
      else
        {
          /* Only update mass and border distance in the global entry */
          groups[global].Mass = my_volume->Groups[gvol].Mass;
          groups[global].aux2 = my_volume->Groups[gvol].aux2;
        }
    }

#undef R_THR_FF
#undef A_THR_FF
  return 0;
}


/**
 * @brief Pre-identify all Fmax peaks in the full subbox and initialise groups[].
 *
 * This is called once before the 8-pass loop.  It populates groups[] with
 * one entry per local maximum of Fmax (above outputs.Flast) and sets ngroups.
 * The group entries are minimal: t_peak, name, good=0, point, bottom, ll,
 * halo_app are set; positions/velocities are NOT set here (they are filled
 * by build_groups_in_volume + merge_catalogs).
 *
 * @return Total number of peaks found.
 */
int find_all_peaks(void)
{
  int iz, i, j, k, nn, i1, j1, k1, peak_cond;
  int npeaks_tot = 0;

  /* Initialise group list: index 0 and FILAMENT are reserved */
  groups[FILAMENT].Mass = 0;
  ngroups = FILAMENT + 1;

  /* After sort_and_organize(), frag[iz] is sorted by decreasing Fmax.
   * frag_pos[iz]     = subbox grid index of the iz-th sorted particle.
   * sorted_pos[grid] = time-order index of particle at grid pos (-1 if absent).
   * We iterate by time order (can break early when Fmax < Flast),
   * but use frag_pos[iz] for spatial coordinates and sorted_pos for neighbours. */

  for (iz = 0; iz < (int)subbox.Nstored; iz++)
    {
      if (frag[iz].Fmax < outputs.Flast)
        break; /* array is sorted descending — safe to break */

      int grid_iz = frag_pos[iz]; /* subbox grid index of this particle */
      INDEX_TO_COORD(grid_iz, i, j, k, subbox.Lgwbl);

      /* Skip particles at the subbox border */
      if (!subbox.pbc[_x_] && (i == 0 || i == subbox.Lgwbl[_x_] - 1)) continue;
      if (!subbox.pbc[_y_] && (j == 0 || j == subbox.Lgwbl[_y_] - 1)) continue;
      if (!subbox.pbc[_z_] && (k == 0 || k == subbox.Lgwbl[_z_] - 1)) continue;

      peak_cond = 1;
      for (nn = 0; nn < 6; nn++)
        {
          switch (nn)
            {
            case 0:
              i1 = (subbox.pbc[_x_] && i == 0 ? subbox.Lgwbl[_x_] - 1 : i - 1);
              j1 = j; k1 = k; break;
            case 1:
              i1 = (subbox.pbc[_x_] && i == subbox.Lgwbl[_x_] - 1 ? 0 : i + 1);
              j1 = j; k1 = k; break;
            case 2:
              i1 = i;
              j1 = (subbox.pbc[_y_] && j == 0 ? subbox.Lgwbl[_y_] - 1 : j - 1);
              k1 = k; break;
            case 3:
              i1 = i;
              j1 = (subbox.pbc[_y_] && j == subbox.Lgwbl[_y_] - 1 ? 0 : j + 1);
              k1 = k; break;
            case 4:
              i1 = i; j1 = j;
              k1 = (subbox.pbc[_z_] && k == 0 ? subbox.Lgwbl[_z_] - 1 : k - 1);
              break;
            case 5:
              i1 = i; j1 = j;
              k1 = (subbox.pbc[_z_] && k == subbox.Lgwbl[_z_] - 1 ? 0 : k + 1);
              break;
            default: i1 = i; j1 = j; k1 = k; break;
            }
          /* Neighbour lookup: sorted_pos[grid] → time-order index into frag[] */
          int neigh_grid = COORD_TO_INDEX(i1, j1, k1, subbox.Lgwbl);
          int neigh_ti   = sorted_pos[neigh_grid]; /* -1 if not stored */
          PRODFLOAT neigh_fmax = (neigh_ti >= 0) ? frag[neigh_ti].Fmax : (PRODFLOAT)0;
          peak_cond &= (frag[iz].Fmax > neigh_fmax);
          if (!peak_cond) break;
        }

      if (peak_cond)
        {
          if (ngroups >= subbox.PredNpeaks)
            {
              printf("ERROR Task %d [find_all_peaks]: ngroups %d exceeds PredNpeaks %d\n",
                     ThisTask, ngroups, subbox.PredNpeaks);
              fflush(stdout);
              return -1;
            }

          groups[ngroups].t_peak   = frag[iz].Fmax;
          groups[ngroups].t_merge  = -1;
          groups[ngroups].Mass     = 1;
          groups[ngroups].name     =
            COORD_TO_INDEX(
              (long long)((i + subbox.stabl[_x_] + MyGrids[0].GSglobal[_x_]) %
                          MyGrids[0].GSglobal[_x_]),
              (long long)((j + subbox.stabl[_y_] + MyGrids[0].GSglobal[_y_]) %
                          MyGrids[0].GSglobal[_y_]),
              (long long)((k + subbox.stabl[_z_] + MyGrids[0].GSglobal[_z_]) %
                          MyGrids[0].GSglobal[_z_]),
              MyGrids[0].GSglobal);
          groups[ngroups].good     = 0;
          groups[ngroups].point    = grid_iz; /* subbox grid index */
          groups[ngroups].bottom   = grid_iz;
          groups[ngroups].ll       = ngroups;
          groups[ngroups].halo_app = ngroups;
          groups[ngroups].aux2     = 0;

          if (params.MinHaloMass <= 1)
            groups[ngroups].t_appear = frag[iz].Fmax;
          else
            groups[ngroups].t_appear = -1;

          linking_list[iz] = iz;
          group_ID[iz]     = ngroups;

          ngroups++;
          npeaks_tot++;
        }
    }

  /* Zero out the FILAMENT and reserved entries */
  for (int gi = 0; gi <= FILAMENT; gi++)
    groups[gi].name = groups[gi].aux2 = 0;
  groups[FILAMENT].t_peak = 0;

  return npeaks_tot;
}


/**
 * @brief 8-pass subvolume fragmentation driver.
 *
 * Tiles the MPI subbox into Nsub^3 sub-cubes offset by half a tile in each
 * combination of x/y/z directions (8 passes total).  Each sub-cube is
 * initialised, fragmented independently with build_groups_in_volume(), and
 * merged back into the global catalogue with merge_catalogs().
 *
 * @return 0 on success, 1 on error.
 */
int fragment_fastfrag(void)
{
  /* 8 pass offsets: each coordinate is either 0 or size/2 */
  static const int pass_offsets[8][3] = {
    {0, 0, 0},
    {1, 0, 0},
    {0, 1, 0},
    {0, 0, 1},
    {1, 1, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 1}
  };

  int Nsub = 4;
  int size = subbox.Lgwbl[_x_] / Nsub;  /* sub-cube side (interior, without border) */

  if (size < 3)
    {
      if (!ThisTask)
        printf("[FastFrag] WARNING: subbox too small for Nsub=%d, forcing Nsub=1\n", Nsub);
      Nsub = 1;
      size = subbox.Lgwbl[_x_];
    }

  /* Per-dimension Nsub to cover non-cubic subboxes (e.g. 2D MPI decomposition) */
  int Nsuby = subbox.Lgwbl[_y_] / size; if (Nsuby < 1) Nsuby = 1;
  int Nsubz = subbox.Lgwbl[_z_] / size; if (Nsubz < 1) Nsubz = 1;

  /* Largest subbox dimension — used to size the "ultimo loop" volumes */
  int Lgwbl_max = subbox.Lgwbl[_x_];
  if (subbox.Lgwbl[_y_] > Lgwbl_max) Lgwbl_max = subbox.Lgwbl[_y_];
  if (subbox.Lgwbl[_z_] > Lgwbl_max) Lgwbl_max = subbox.Lgwbl[_z_];
  int Lgwbl_min = subbox.Lgwbl[_x_];
  if (subbox.Lgwbl[_y_] < Lgwbl_min) Lgwbl_min = subbox.Lgwbl[_y_];
  if (subbox.Lgwbl[_z_] < Lgwbl_min) Lgwbl_min = subbox.Lgwbl[_z_];

  /* Initial "ultimo loop" volume side: centered on the peak, minimum size
     that guarantees dist(peak, border) = (size_u0-1)/2 > 9 (= A_THR_FF).
     half-size = 10 > 9 => peak is always "safe" in one pass. */
  int size_u0 = 21;   /* 2*(9+1)+1 = 21; matches A_THR_FF=9 in merge_catalogs */

  /* Single reusable buffer, sized for max of: 8-pass volume and ultimo initial volume */
  size_t n_pass   = (size_t)(size + 2) * (size + 2) * (size + 2);
  size_t n_ultimo = (size_t)size_u0 * size_u0 * size_u0;
  size_t vol_n    = (n_pass > n_ultimo) ? n_pass : n_ultimo;
  size_t vol_sz   = vol_n * (sizeof(product_data) + 2 * sizeof(int) +
                              sizeof(group_data));
  char  *vol_buf  = (char *)malloc(vol_sz);
  if (!vol_buf)
    {
      printf("ERROR Task %d [fragment_fastfrag]: malloc failed for volume buffer (%zu bytes)\n",
             ThisTask, vol_sz);
      fflush(stdout);
      return 1;
    }

  if (!ThisTask)
    printf("[%s] FastFrag: Nsub=%d size=%d Nsubyz=%d/%d ultimo_size=%d buf %zu MB\n",
           fdate(), Nsub, size, Nsuby, Nsubz, size_u0, vol_sz / (1024 * 1024));

  /* Pre-identify all peaks in the subbox and initialise groups[] */
  int Npeaks = find_all_peaks();
  if (Npeaks < 0)
    {
      free(vol_buf);
      return 1;
    }

  if (!ThisTask)
    printf("[%s] FastFrag: found %d peaks in subbox\n", fdate(), Npeaks);

  /* 8-pass loop — use per-dimension Nsub to cover non-cubic subboxes */
  for (int pass = 0; pass < 8; pass++)
    {
      int ox = pass_offsets[pass][0] * (size / 2);
      int oy = pass_offsets[pass][1] * (size / 2);
      int oz = pass_offsets[pass][2] * (size / 2);

      for (int is = 0; is < Nsub;  is++)
        for (int js = 0; js < Nsuby; js++)
          for (int ks = 0; ks < Nsubz; ks++)
            {
              volume_data my_volume;

              int ic = is * size - 1 + ox;
              int jc = js * size - 1 + oy;
              int kc = ks * size - 1 + oz;

              if (initialize_volume(ic, jc, kc,
                                    size + 2, size + 2, size + 2,
                                    &my_volume, vol_buf, vol_sz))
                {
                  free(vol_buf);
                  return 1;
                }

              if (build_groups_in_volume(&my_volume))
                {
                  free(vol_buf);
                  return 1;
                }

              if (merge_catalogs(&my_volume))
                {
                  free(vol_buf);
                  return 1;
                }
            }

      if (!ThisTask)
        printf("[%s] FastFrag: pass %d/%d done\n", fdate(), pass + 1, 8);
    }

  /* ------------------------------------------------------------------
   * Ultimo loop: resolve halos that were never safely interior to
   * any of the 8-pass volumes (i.e. groups with good==0 after all passes).
   * For each such halo, build a volume centred on its peak and process it.
   * Analogous to the "ultimo loop" in src_fastfrag/fragment.c.
   * ------------------------------------------------------------------ */
  {
    int n_resolved = 0;

    for (int g = FILAMENT + 1; g < ngroups; g++)
      {
        if (groups[g].good)            continue;   /* already done   */
        if (groups[g].Mass <= 1)       continue;   /* single-particle, skip */
        if (groups[g].halo_app != g)   continue;   /* merged branch  */

        int ib, jb, kb;
        INDEX_TO_COORD(groups[g].point, ib, jb, kb, subbox.Lgwbl);

        int size_u = size_u0;

        do
          {
            /* Grow buffer if needed */
            size_t need_u = (size_t)size_u * size_u * size_u *
                            (sizeof(product_data) + 2 * sizeof(int) +
                             sizeof(group_data));
            if (need_u > vol_sz)
              {
                char *new_buf = (char *)realloc(vol_buf, need_u);
                if (!new_buf)
                  {
                    printf("ERROR Task %d [fastfrag ultimo]: realloc failed "
                           "for %zu bytes\n", ThisTask, need_u);
                    free(vol_buf);
                    return 1;
                  }
                vol_buf = new_buf;
                vol_sz  = need_u;
              }

            volume_data my_volume;

            if (initialize_volume(ib - size_u / 2,
                                  jb - size_u / 2,
                                  kb - size_u / 2,
                                  size_u, size_u, size_u,
                                  &my_volume, vol_buf, vol_sz))
              {
                free(vol_buf);
                return 1;
              }

            if (build_groups_in_volume(&my_volume))
              {
                free(vol_buf);
                return 1;
              }

            if (merge_catalogs(&my_volume))
              {
                free(vol_buf);
                return 1;
              }

            if (!groups[g].good)
              size_u = (int)(size_u * 1.2);
          }
        while (!groups[g].good && size_u <= Lgwbl_min);

        if (groups[g].good)
          n_resolved++;
      }

    if (!ThisTask)
      printf("[%s] FastFrag: ultimo loop resolved %d additional halos\n",
             fdate(), n_resolved);
  }

  free(vol_buf);

  /* ------------------------------------------------------------------
   * Statistics and catalog output
   * ------------------------------------------------------------------ */

  /* Count good halos across all MPI tasks */
  {
    unsigned long long good_halos = 0, all_good_halos = 0;
    unsigned long long n_peaks    = (unsigned long long)Npeaks;
    unsigned long long all_peaks  = 0;
    int ig1;

    for (ig1 = FILAMENT + 1; ig1 <= ngroups; ig1++)
      if (groups[ig1].point >= 0 && groups[ig1].good)
        good_halos++;

    MPI_Reduce(&n_peaks,    &all_peaks,      1, MPI_UNSIGNED_LONG_LONG,
               MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&good_halos, &all_good_halos, 1, MPI_UNSIGNED_LONG_LONG,
               MPI_SUM, 0, MPI_COMM_WORLD);

    if (!ThisTask)
      {
        printf("Total number of peaks:                 %llu\n", all_peaks);
        printf("Total number of good halos:            %llu\n", all_good_halos);
        printf("\n");
      }
  }

  /* Write output catalogs for all requested redshifts */
  {
    double cputmp;
    int iout;

    for (iout = 0; iout < outputs.n; iout++)
      {
        cputmp = MPI_Wtime();

        if (!ThisTask)
          printf("[%s] Writing output at z=%f (FastFrag)\n",
                 fdate(), outputs.z[iout]);

        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);

        if (write_catalog(iout))
          return 1;

        if (compute_mf(iout))
          return 1;

        if (iout == outputs.n - 1)
          if (write_histories())
            return 1;

        cputime.io += MPI_Wtime() - cputmp;
      }
  }

  return 0;
}

#endif /* USE_FASTFRAG */

