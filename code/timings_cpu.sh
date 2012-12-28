echo "HPC CPU PROJECT TIMINGS" > timings_cpu.out

echo "" >> timings_cpu.out
echo "NX TIMINGS" >> timings_cpu.out
for nx in 125 250 500 1000 2000 4000 8000 16000; do

    echo "" >> timings_cpu.out
    echo "NX = $nx" >> timings_cpu.out
    echo "" >> timings_cpu.out

    ./solve $nx 1000 100 120 0 >> timings_cpu.out

done

echo "" >> timings_cpu.out
echo "NQ TIMINGS" >> timings_cpu.out
for nq in 125 250 500 1000 2000 4000 8000 16000; do

    echo "" >> timings_cpu.out
    echo "NQ = $nq" >> timings_cpu.out
    echo "" >> timings_cpu.out

    ./solve 1000 $nq 100 120 0 >> timings_cpu.out

done

echo "" >> timings_cpu.out
echo "NSIM TIMINGS" >> timings_cpu.out
for nsim in 125 250 500 1000 2000 4000 6000 8000 12000 16000; do

    echo "" >> timings_cpu.out
    echo "NSIM = $nsim" >> timings_cpu.out
    echo "" >> timings_cpu.out

    ./solve 1000 1000 $nsim 1200 0 >> timings_cpu.out

done

echo "" >> timings_cpu.out
echo "NT TIMINGS" >> timings_cpu.out
for nt in 150 300 600 1200 2400 4800 7200 9600 14400 19200; do

    echo "" >> timings_cpu.out
    echo "NT = $nt" >> timings_cpu.out
    echo "" >> timings_cpu.out
    ./solve 1000 1000 5000 $nt 0 >> timings_cpu.out

done