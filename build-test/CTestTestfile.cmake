# CMake generated Testfile for 
# Source directory: /home/andrew/migration/metaphor
# Build directory: /home/andrew/migration/metaphor/build-test
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[metaphor_tests]=] "/home/andrew/.local/bin/c3c" "test" "metaphor")
set_tests_properties([=[metaphor_tests]=] PROPERTIES  WORKING_DIRECTORY "/home/andrew/migration/metaphor" _BACKTRACE_TRIPLES "/home/andrew/migration/metaphor/CMakeLists.txt;399;add_test;/home/andrew/migration/metaphor/CMakeLists.txt;0;")
