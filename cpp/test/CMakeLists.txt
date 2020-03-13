cmake_minimum_required(VERSION 3.2)

add_subdirectory("${PROJECT_SOURCE_DIR}/test/extern/googletest" "extern/googletest")

mark_as_advanced(
    BUILD_GMOCK BUILD_GTEST BUILD_SHARED_LIBS
    gmock_build_tests gtest_build_samples gtest_build_tests
    gtest_disable_pthreads gtest_force_shared_crt gtest_hide_internal_symbols
)

macro(package_add_test TESTNAME)
    add_executable(${TESTNAME} ${ARGN})
    target_link_libraries(${TESTNAME} contact_modes gtest gmock gtest_main dw)
    add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})
    set_target_properties(${TESTNAME} PROPERTIES FOLDER test)
endmacro()

macro(package_add_cu_test TESTNAME)
    add_executable(${TESTNAME} ${ARGN})
    # target_link_libraries(${TESTNAME} gtest gmock gtest_main dw)
    add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})
    # set_target_properties(${TESTNAME} PROPERTIES FOLDER test)
    set_target_properties(${TESTNAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(${TESTNAME} PRIVATE itbl gtest gmock gtest_main dw)
    target_compile_options(${TESTNAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-relaxed-constexpr
    --relocatable-device-code=true
    >)
endmacro()

package_add_test(test_arrangements test_arrangements.cpp)