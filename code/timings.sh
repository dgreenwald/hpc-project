echo "HPC PROJECT TIMINGS" > timings.out


echo "" >> timings.out
echo "NSIM TIMINGS" >> timings.out
for nsim in 125 250 500 1000 2000 4000 6000 8000 12000 16000; do

    echo "" >> timings.out
    echo "NSIM = $nsim" >> timings.out
    echo "" >> timings.out

    ./solve 1000 1000 $nsim 1200 1 >> timings.out

done

echo "" >> timings.out
echo "NT TIMINGS" >> timings.out
for nt in 150 300 600 1200 2400 4800 7200 9600 14400 19200; do

    echo "" >> timings.out
    echo "NT = $nt" >> timings.out
    echo "" >> timings.out
    ./solve 1000 1000 5000 $nt 1 >> timings.out

done

echo "" >> timings.out
echo "NX TIMINGS" >> timings.out
for nx in 125 250 500 1000 2000 4000 6000 8000 12000 16000; do

    echo "" >> timings.out
    echo "NX = $nx" >> timings.out
    echo "" >> timings.out

    ./solve $nx 1000 100 120 1 >> timings.out

done

echo "" >> timings.out
echo "NQ TIMINGS" >> timings.out
for nq in 125 250 500 1000 2000 4000 6000 8000 12000 16000; do

    echo "" >> timings.out
    echo "NQ = $nq" >> timings.out
    echo "" >> timings.out

    ./solve 1000 $nq 100 120 1 >> timings.out

done
