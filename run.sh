#! /bin/bash

wait_press() {

    echo 
    echo "Press ENTER key to continue..."

    while [ true ] ; do
        read -s -N 1 -t 1 key
        if [[ $key == $'\x0a' ]] ; then
            echo 
            break        
        fi
    done
}

# Autoraders

wait_press

python autograder.py -q q1


wait_press

python pacman.py --frameTime 0.02 -p ReflexAgent -l testClassic
python pacman.py --frameTime 0.02 -p ReflexAgent -k 1
python pacman.py --frameTime 0.02 -p ReflexAgent -k 2


wait_press

python autograder.py -q q2



wait_press

for i in {1..5}; do python pacman.py --frameTime 0.02 -p MinimaxAgent -l trappedClassic -a depth=3 ; done



wait_press

for i in {1..5}; do python pacman.py --frameTime 0.02 -p MinimaxAgent -l minimaxClassic -a depth=3 ; done



wait_press

python autograder.py -q q3



wait_press

python pacman.py --frameTime 0.02 -p AlphaBetaAgent -a depth=3 -l smallClassic



wait_press

for i in {1..5}; do python pacman.py --frameTime 0.02 -p AlphaBetaAgent -l trappedClassic -a depth=3 -q -n 10 ; done



wait_press

python autograder.py -q q4



wait_press

for i in {1..5}; do python pacman.py -p AlphaBetaAgent -l trappedClassic -a depth=3 -q -n 10 ; done



wait_press

python pacman.py -p ExpectimaxAgent -l trappedClassic -a depth=3 -q -n 10