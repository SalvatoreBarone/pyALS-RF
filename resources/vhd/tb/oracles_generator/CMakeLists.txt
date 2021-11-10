project(oracles)
cmake_minimum_required(VERSION 3.3)
set (CMAKE_CXX_STANDARD 11)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY                     ${CMAKE_BINARY_DIR}/libs)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY                     ${CMAKE_BINARY_DIR}/libs)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY                      ${CMAKE_BINARY_DIR}/bin)
set(ORACLES_DESTINATION_DIRECTORY                  ${CMAKE_CURRENT_LIST_DIR}/..)

add_executable(tb_basic_comparator                        tb_basic_comparator.c)
add_executable(tb_parallel_comparator                  tb_parallel_comparator.c)
add_executable(tb_decision_box_int                        tb_decision_box_int.c)
add_executable(tb_decision_box_fp                         tb_decision_box_fp.cc)
add_executable(tb_combiner                                        tb_combiner.c)

add_custom_target(
    oracle_files ALL
    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    COMMAND echo "Please, wait. This may take a while..."
    COMMAND ./tb_basic_comparator    > ${ORACLES_DESTINATION_DIRECTORY}/tb_basic_comparator.txt 
    COMMAND ./tb_parallel_comparator > ${ORACLES_DESTINATION_DIRECTORY}/tb_parallel_comparator.txt 
    COMMAND ./tb_decision_box_int    > ${ORACLES_DESTINATION_DIRECTORY}/tb_decision_box_int.txt 
    COMMAND ./tb_decision_box_fp     > ${ORACLES_DESTINATION_DIRECTORY}/tb_decision_box_fp.txt 
    COMMAND ./tb_combiner            > ${ORACLES_DESTINATION_DIRECTORY}/tb_combiner.txt 
    DEPENDS tb_basic_comparator tb_parallel_comparator tb_decision_box_int tb_decision_box_fp tb_combiner
    VERBATIM)