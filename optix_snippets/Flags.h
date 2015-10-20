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

#pragma once

// '1' for SOA layout
#define MHTK_SOA_LAYOUT (0)

// '1' for indexing [HIT][HEIGHT][WIDTH]
// '0' for indexing [WIDTH][HEIGHT][HIT]
#define REVERSED_ACCESS (0)

// '1' to turn on sorting
#define MHTK_SORTING (1)

#if MHTK_SORTING
// '1' for sorting after intersection
# define MHTK_POST_SORTING (0)
#endif

#define VIZ_DATA (1)

#if VIZ_DATA
// '1' to show # swaps
# define VIZ_SWAPS (1)
#endif

#define HITPOINT_BUFFER_SIZE 200
