if (CUDA_FOUND)
  add_subdirectory (cuda)
endif (CUDA_FOUND)

add_library (neural SHARED
  cache.h
  encoder.h
  factory.h
  loader.h
  network.h
  writer.h
  cache.cc
  encoder.cc
  factory.cc
  loader.cc
  network_mux.cc
  network_random.cc
  writer.cc
)

target_link_libraries (neural
  utils
  ${CMAKE_THREAD_LIBS_INIT}
  ${Boost_LIBRARIES}
)

if (CUDNN_FOUND)
  target_link_libraries(neural
    neural_cudnn
  )
endif(CUDNN_FOUND)

if (TENSORRT_FOUND)
  target_link_libraries(neural
    neural_tensorrt
  )
endif (TENSORRT_FOUND)