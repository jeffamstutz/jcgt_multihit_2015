## ======================================================================== ##
## Copyright 2015 SURVICE Engineering Company                               ##
##                                                                          ##
## Licensed under the Apache License, Version 2.0 (the "License");          ##
## you may not use this file except in compliance with the License.         ##
## You may obtain a copy of the License at                                  ##
##                                                                          ##
##     http://www.apache.org/licenses/LICENSE-2.0                           ##
##                                                                          ##
## Unless required by applicable law or agreed to in writing, software      ##
## distributed under the License is distributed on an "AS IS" BASIS,        ##
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. ##
## See the License for the specific language governing permissions and      ##
## limitations under the License.                                           ##
## ======================================================================== ##

SET(OSPRAY_MODULE_MHTK OFF CACHE BOOL "Build 'MHTK' module")

IF (OSPRAY_MODULE_MHTK)
  CONFIGURE_OSPRAY()

  IF (NOT OSPRAY_INTERSECTION_FILTER)
    MESSAGE(SEND_ERROR
            "OSPRAY_INTERSECTION_FILTER must be enabled to use the MHTK module."
    )
  ENDIF()

  SET(OSPRAY_MODULE_MHTK_SORTING ON CACHE BOOL "Sort MHTK intersections")

  IF (OSPRAY_MODULE_MHTK_SORTING)
    ADD_DEFINITIONS_ISPC(-DMHTK_SORTING)
  ENDIF()

  INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR})
  INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/ospray/include)

  INCLUDE_DIRECTORIES_ISPC(${PROJECT_SOURCE_DIR})
  INCLUDE_DIRECTORIES_ISPC(${EMBREE_DIR}/include)

  OSPRAY_ADD_LIBRARY(ospray_module_mhtk${OSPRAY_LIB_SUFFIX} SHARED
    xray_renderer.cpp
    xray_renderer.ispc
  )

  TARGET_LINK_LIBRARIES(ospray_module_mhtk${OSPRAY_LIB_SUFFIX}
    ospray${OSPRAY_LIB_SUFFIX}
  )
ENDIF()
