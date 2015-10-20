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
#include "helpers.h"

using namespace optix;

#include "RayPayload.h"

rtDeclareVariable(float3, eye, , );
rtDeclareVariable(float3, U, , );
rtDeclareVariable(float3, V, , );
rtDeclareVariable(float3, W, , );
rtDeclareVariable(float3, bad_color, , );
rtDeclareVariable(float, scene_epsilon, , );
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(unsigned int, multihit_ray_type, , );
rtBuffer<uchar4, 2> output_buffer;
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

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );
rtDeclareVariable(float, time_view_scale, , ) = 1e-6f;

static __host__ __device__ __inline__
optix::float3 make_grad_color(const int i, const int max)
{
  float f = i;
  if (i == 0)
    return make_float3(0.f);                                   // black
  else if (i <= max/4)
    return make_float3(0.f, 0.f, f/(max/4));                   // bluish
  else if (i <= max/2)
    return make_float3(0.f, f/(max/4) - 1.f, 2.f - f/(max/4)); // cyan/green
  else if (i <= 3*max/4)
    return make_float3(f/(max/4)-2.f, 1.f, 0.f);               // green/yellow
  else if (i <= max)
    return make_float3(1.f, 4.0-f/(max/4), 0.f);               // orange
  else
    return make_float3(1.f, 0.0, 0.f);                         // red
}


RT_PROGRAM void pinhole_camera_mh()
{
  float2 d = make_float2(launch_index) / make_float2(launch_dim) * 2.f - 1.f;
  float3 ray_origin = eye;
  float3 ray_direction = normalize(d.x*U + d.y*V + W);

  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, multihit_ray_type,
                                   scene_epsilon, RT_DEFAULT_MAX);

  PerRayData_multihit prd;
  prd.hitBufferOverflow = false;
  prd.numHits  = 0;
  prd.numSwaps = 0;

  rtTrace(top_object, ray, prd);

#if MHTK_SORTING
# if MHTK_POST_SORTING
#  if MHTK_SOA_LAYOUT
  if (prd.numHits > 0)
  {
#    if REVERSED_ACCESS
    uint3 i = make_uint3(0u, launch_index.y, launch_index.x);
    for (; i.x < prd.numHits - 1; ++(i.x))
    {
#    else
    uint3 i = make_uint3(launch_index.x, launch_index.y, 0u);
    for (; i.z < prd.numHits - 1; ++(i.z))
    {
#    endif
      float d = hits_T[i];
      uint3 pos = i;
      bool doSwap = false;
#    if REVERSED_ACCESS
      uint3 j = make_uint3(i.x+1, i.y, i.z);
      for (; j.x < prd.numHits; ++(j.x))
      {
#    else
      uint3 j = make_uint3(i.x, i.y, i.z+1);
      for (; j.z < prd.numHits; ++(j.z))
      {
#    endif
        if(hits_T[j] < d)
        {
          doSwap = true;
          pos = j;
          d = hits_T[j];
        }
      }

      if (doSwap)
      {
        Hitpoint tmp;
        tmp.t      = hits_T[i];
        tmp.primID = hits_PrimID[i];
        tmp.geomID = hits_GeomID[i];
        tmp.Ng[0]  = hits_Ngx[i];
        tmp.Ng[1]  = hits_Ngy[i];
        tmp.Ng[2]  = hits_Ngz[i];

        hits_T[i]      = hits_T[pos];
        hits_PrimID[i] = hits_PrimID[pos];
        hits_GeomID[i] = hits_GeomID[pos];
        hits_Ngx[i]    = hits_Ngx[pos];
        hits_Ngy[i]    = hits_Ngy[pos];
        hits_Ngz[i]    = hits_Ngz[pos];

        hits_T[pos]      = tmp.t;
        hits_PrimID[pos] = tmp.primID;
        hits_GeomID[pos] = tmp.geomID;
        hits_Ngx[pos]    = tmp.Ng[0];
        hits_Ngy[pos]    = tmp.Ng[1];
        hits_Ngz[pos]    = tmp.Ng[2];
#if VIZ_DATA
        prd.numSwaps++;
#endif
      }
    }
  }
#  else
  if (prd.numHits > 0)
  {
    uint3 i = make_uint3(launch_index.x, launch_index.y, 0u);
    /* sort the hitpoints (only for measuring performance) */
    for (; i.z < prd.numHits - 1; ++(i.z))
    {
      float d = hits_buffer[i].t;
      uint3 pos = i;
      bool doSwap = false;
      uint3 j = make_uint3(i.x, i.y, i.z+1);
      for (; j.z < prd.numHits; ++(j.z))
      {
        if(hits_buffer[j].t < d)
        {
          doSwap = true;
          pos = j;
          d = hits_buffer[j].t;
        }
      }

      if (doSwap)
      {
        Hitpoint tmp     = hits_buffer[i];
        hits_buffer[i]   = hits_buffer[pos];
        hits_buffer[pos] = tmp;
#if VIZ_DATA
        prd.numSwaps++;
#endif
      }
    }
  }
#  endif
# endif
#endif

#if VIZ_DATA
# if VIZ_SWAPS
  output_buffer[launch_index] = make_color(make_grad_color(prd.numSwaps, 128));
# else
  output_buffer[launch_index] = make_color(make_grad_color(prd.numHits, 128));
# endif
#else
  if (!prd.hitBufferOverflow)
    output_buffer[launch_index] = make_color(make_float3(prd.numHits / 10.f));
  else
    output_buffer[launch_index] = make_color(make_float3(1.f, 0.f, 0.f));
#endif
}

RT_PROGRAM void exception()
{
  const unsigned int code = rtGetExceptionCode();
  rtPrintf( "Caught exception 0x%X at launch index (%d,%d)\n", code, launch_index.x, launch_index.y );
  output_buffer[launch_index] = make_color( bad_color );
}
