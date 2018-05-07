if (CUDNN_FOUND)
    cuda_add_library (neural_cudnn SHARED
        network_cudnn.cu
    )

    target_link_libraries(neural_cudnn
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDNN_LIBRARY}
        utils
    )
endif (CUDNN_FOUND)

if (TENSORRT_FOUND)
endif (TENSORRT_FOUND)
