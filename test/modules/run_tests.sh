#!/bin/bash
TEST_TYPE=all

FPGACONVNET_HLS_PATH=$PWD/../../fpgaconvnet/hls

while getopts ":m:n:cseih" opt; do
    case ${opt} in
        m )
            MODULE=$OPTARG
            ;;
        n )
            TEST_NUM=$OPTARG
            ;;
        c )
            # c simulation
            TEST_TYPE=sim
            ;;
        s )
            # synthesis
            TEST_TYPE=synth
            ;;
        e )
            # co-simulation
            TEST_TYPE=cosim
            ;;
        i )
            # implementation
            TEST_TYPE=impl
            ;;
        h )
            echo "USAGE: run_test.sh [-m (module)] [-n (test number)] [-c,-s,-e,-i]"
            echo "  -c = C simulation"
            echo "  -s = Synthesis"
            echo "  -e = Co-simulation"
            echo "  -i = Implementation"
            exit
            ;;
    esac
done
shift $((OPTIND -1))

function run_test {
    echo "RUNNING TEST ${1}"
    # GENERATE INPUTS
    mkdir -p data/test_${1}
    python gen_data.py -c config/config_${1}.json -o $PWD/data/test_${1} -h tb
    # RUN TEST
    vitis-run --mode hls --tcl ../run_module_hls.tcl -- "_ -num ${1} -type ${TEST_TYPE} -name ${MODULE}"

}

# move to folder
cd $MODULE

if [ $TEST_NUM ]; then

    # run the test
    run_test $TEST_NUM

    # generate reports
    python ../report.py -m $MODULE -n $TEST_NUM

else

    # number of tests
    NUM_TEST="$(ls config/ -1U | wc -l)"
    # iterate over tests
    for i in $( seq 0 $(($NUM_TEST-1)) ); do

        # run the test
        run_test $i

    done

    # generate reports
    python ../report.py -m $MODULE

fi
