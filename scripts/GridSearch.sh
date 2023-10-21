#!/usr/bin/env bash
m3=0
for seed in 42
do
    for lr in 3e-5 5e-5
    do
        for lambda1 in 0.01 0.015 0.02 0.03 0.04 0.05
        do
            for alpha in  0.1 0.2 0.3 0.4 0.6 0.7 0.8
            do
                for beta in 0.1 0.2 0.3 0.4 0.6 0.7 0.8
                do
                    
                    k=$(echo "$m1+$m2" | bc)
                    if [ `echo "$k>=1.0" | bc` -eq 1 ]
                    then
                        continue
                    fi
                    m3=$(echo "1.0-$k" | bc)
                    
                    sh scripts/train.sh 2 $lambda1 wikievents $seed $lr $alpha $ $beta $m3
                    sh scripts/test.sh 2 $lambda1 wikievents $seed $lr $alpha $beta $m3
                done
            done
        done
    done
done
