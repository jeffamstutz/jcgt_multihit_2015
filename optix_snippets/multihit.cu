// ======================================================================== //
// Copyright 2015 Jefferson Amstutz                                         //
// Copyright 2015 SURVICE Engineering Company                               //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include <optix_world.h>
#include "random.h"
#include "commonStructs.h"

#include "RayPayload.h"

using namespace optix;

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );

rtDeclareVariable(PerRayData_multihit, prd_mh, rtPayload, );

#if MHTK_SOA_LAYOUT
rtBuffer<float, 3> hits_T;
rtBuffer<int,   3> hits_PrimID;
rtBuffer<int,   3> hits_GeomID;
rtBuffer<float, 3> hits_Ngx;
rtBuffer<float, 3> hits_Ngy;
rtBuffer<float, 3> hits_Ngz;
#else
rtBuffer<Hitpoint, 3> hits_buffer;
#endif
RT_PROGRAM void any_hit_multihit()
{
  if (prd_mh.numHits >= HITPOINT_BUFFER_SIZE)
  {
    prd_mh.hitBufferOverflow = true;
    rtTerminateRay();
    return;
  }

#if MHTK_SORTING
# if !MHTK_POST_SORTING
#  if MHTK_SOA_LAYOUT
#   if REVERSED_ACCESS
  uint3 i = make_uint3(prd_mh.numHits, launch_index.y, launch_index.x);
#   else
  uint3 i = make_uint3(launch_index.x, launch_index.y, prd_mh.numHits);
#   endif

#   if REVERSED_ACCESS
  for (; i.x > 0; --i.x)
  {
    uint3 j = make_uint3(i.x-1, i.y, i.z);
#   else
  for (; i.z > 0; --i.z)
  {
    uint3 j = make_uint3(i.x, i.y, i.z-1);
#   endif
    if (hits_T[j] > t_hit)
    {
      hits_T[i]      = hits_T[j];
      hits_PrimID[i] = hits_PrimID[j];
      hits_GeomID[i] = hits_GeomID[j];
      hits_Ngx[i]    = hits_Ngx[j];
      hits_Ngy[i]    = hits_Ngy[j];
      hits_Ngz[i]    = hits_Ngz[j];
#   if VIZ_DATA
      prd_mh.numSwaps++;
#   endif
    }
    else
      break;
  }

  hits_T[i]      = t_hit;
  hits_Ngx[i]    = geometric_normal.x;
  hits_Ngy[i]    = geometric_normal.y;
  hits_Ngz[i]    = geometric_normal.z;
  hits_PrimID[i] = 0;
  hits_PrimID[i] = 0;
  prd_mh.numHits++;
#  else
  uint3 i = make_uint3(launch_index.x, launch_index.y, prd_mh.numHits);

  for (; i.z > 0; --i.z)
  {
    uint3 j = make_uint3(i.x, i.y, i.z-1);
    if (hits_buffer[j].t > t_hit)
    {
      hits_buffer[i] = hits_buffer[j];
#   if VIZ_DATA
      prd_mh.numSwaps++;
#   endif
    }
    else
      break;
  }

  Hitpoint &hit = hits_buffer[i];
  hit.t      = t_hit;
  hit.Ng[0]  = geometric_normal.x;
  hit.Ng[1]  = geometric_normal.y;
  hit.Ng[2]  = geometric_normal.z;
  hit.primID = 0;
  hit.geomID = 0;
  prd_mh.numHits++;
#  endif
# else
#  if MHTK_SOA_LAYOUT
#    if REVERSED_ACCESS
  uint3 index = make_uint3(prd_mh.numHits++, launch_index.y, launch_index.x);
#    else
  uint3 index = make_uint3(launch_index.x, launch_index.y, prd_mh.numHits++);
#    endif
  hits_T[index]      = t_hit;
  hits_Ngx[index]    = geometric_normal.x;
  hits_Ngy[index]    = geometric_normal.y;
  hits_Ngz[index]    = geometric_normal.z;
  hits_PrimID[index] = 0;
  hits_GeomID[index] = 0;
#  else
#   if REVERSED_ACCESS
  uint3 index = make_uint3(prd_mh.numHits++, launch_index.y, launch_index.x);
#   else
  uint3 index = make_uint3(launch_index.x, launch_index.y, prd_mh.numHits++);
#   endif
  Hitpoint &hit = hits_buffer[index];
  hit.t = t_hit;
  hit.Ng[0] = geometric_normal.x;
  hit.Ng[1] = geometric_normal.y;
  hit.Ng[2] = geometric_normal.z;
  hit.primID = 0;
  hit.geomID = 0;
#  endif
# endif
#else
# if MHTK_SOA_LAYOUT
#  if REVERSED_ACCESS
  uint3 index = make_uint3(prd_mh.numHits++, launch_index.y, launch_index.x);
#  else
  uint3 index = make_uint3(launch_index.x, launch_index.y, prd_mh.numHits++);
#  endif
  hits_T[index]      = t_hit;
  hits_Ngx[index]    = geometric_normal.x;
  hits_Ngy[index]    = geometric_normal.y;
  hits_Ngz[index]    = geometric_normal.z;
  hits_PrimID[index] = 0;
  hits_GeomID[index] = 0;
# else
#  if REVERSED_ACCESS
  uint3 index = make_uint3(prd_mh.numHits++, launch_index.y, launch_index.x);
#  else
  uint3 index = make_uint3(launch_index.x, launch_index.y, prd_mh.numHits++);
#  endif
  Hitpoint &hit = hits_buffer[index];
  hit.t = t_hit;
  hit.Ng[0] = geometric_normal.x;
  hit.Ng[1] = geometric_normal.y;
  hit.Ng[2] = geometric_normal.z;
  hit.primID = 0;
  hit.geomID = 0;
# endif
#endif

  rtIgnoreIntersection();
}

// ----------------------------------------------------------


