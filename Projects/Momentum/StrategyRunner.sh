#!/bin/bash

function get_running_n () {
  screen_out=$(sudo -u batyrkhan screen -ls)
  screen_det=$(echo "$screen_out" | grep Detached)
  screen_att=$(echo "$screen_out" | grep Attached)

  det_n=$(echo "$screen_det" | wc -l)
  att_n=$(echo "$screen_att" | wc -l)

  if [ -z "$screen_det" ]; then
    det_n=0
  fi

  if [ -z "$screen_att" ]; then
    att_n=0
  fi

  screen_n=$((det_n+att_n))
  echo "$screen_n"
}

echo "asd $n"


# A shell script to print each number five times. 75 - 155
for param1 in $(seq 75 10 165)
do
    for param2 in $(seq 1 1 10)  # 20 - 100
    do
          echo -n "$param1 $param2 "
        sudo -u batyrkhan screen -d -m -S "$param1$param2" ./Strategy.py dots_n "$param1" rebalance_window "$param2"
        sleep 5
        n=$(get_running_n)
        if [ $n -ge 8 ]
        then
          while [ $n -ge 1 ];
          do
            sleep 60
            n=$(get_running_n)
          done

          swapoff -a; swapon -a
          sync; echo 1 > /proc/sys/vm/drop_caches
          sync; echo 2 > /proc/sys/vm/drop_caches
          sync; echo 2 > /proc/sys/vm/drop_caches
          sync; echo 3 > /proc/sys/vm/drop_caches
        fi
    done

  echo "" #### print the new line ###
done
