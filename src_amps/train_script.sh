gpu=0

for D in `seq 10 10 100`; do
    python train_mnist.py --gpu $gpu --D $D --epochs 50 --step_size 10 --save-model --sample-image
done

for D in `seq 2 2 10`; do
    python train_sk.py --gpu $gpu --D $D
done

for D in 2 5 10; do
    for m in 2 4 8 16 32 64 128 256 512 1024; do
        for seed in `seq 1 1 10`; do
            python train_randvec.py --gpu $gpu --n 20 --m $m --D $D --seed $seed
        done
    done
done

for D in 2 5 10; do
    for n in `seq 10 10 100`; do
        for seed in `seq 1 1 10`; do
            python train_randvec.py --gpu $gpu --n $n --m 100 --D $D --seed $seed
        done
    done
done