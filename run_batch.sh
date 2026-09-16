#!/bin/bash
requested_tasks=(25)
timeliness=(0.09)
acc=(65)
link_rate=(7000000)
images=(1100)
CPU=(4500)
UGS_CPU=(2000)


for n in "${requested_tasks[@]}"; do
    for i in "${images[@]}"; do
        for a in "${acc[@]}"; do
            for c in "${CPU[@]}"; do
                for l in "${timeliness[@]}"; do
                    for r in "${link_rate[@]}"; do
                        for u in "${UGS_CPU[@]}"; do
                            echo n: $n lat: $l acc: $a num-images: $i erramnt: .07 set_cpu: $c set_band: $r
                            python -m examples.AnglovaScenarioDAVC --ra -n $n -error band --lat $l --acc $a --num-images $i -erramnt .07 -set_cpu $c -set_ugs_cpu $u -set_band $r --num_seq_sub 3
                        done
                    done
                done
            done
        done
    done
done