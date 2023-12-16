kernels=(atax bicg doitgen-red gemm-p gemver trmm-opt bicg-large doitgen gesummv 2mm symm-opt syrk mvt trmm symm correlation gemm-p-large fdtd-2d-large covariance)
kernels=(atax bicg doitgen-red gemm-p gemver trmm-opt bicg-medium doitgen gesummv 2mm 3mm symm syrk mvt trmm correlation fdtd-2d adi atax-medium gesummv-medium mvt-medium jacobi-1d jacobi-2d heat-3d)
echo $HOSTNAME
if [ $HOSTNAME == 'u22-atefehSZ' ]
then
    kernels=(aes gemm-blocked gemm-ncubed spmv-ellpack stencil nw stencil-3d)
    kernels=(2mm syrk mvt trmm symm correlation mvt-medium)
    kernels=(gemm-p-large covariance correlation)
    benchmark=poly
elif [ $HOSTNAME == 'u12-atefehSZ' ]
then
    kernels=(doitgen 3mm fdtd-2d jacobi-1d gemver gesummv gemm-p adi seidel-2d)
    kernels=(gemm-p)
    benchmark=poly
elif [ $HOSTNAME == 'u15-atefehSZ' ]
then
    kernels=(doitgen 3mm fdtd-2d jacobi-1d gemver)
    kernels=(syrk symm symm-opt)
    benchmark=poly
elif [ $HOSTNAME == 'u7-atefehSZ' ]
then
    kernels=(bicg 3mm doitgen-red gemm-p gemver trmm-opt doitgen gesummv atax-medium bicg-medium)
    kernels=(mvt syrk syr2k jacobi-2d heat-3d)
    kernels=(gemm-p trmm)
    benchmark=poly
elif [ $HOSTNAME == 'u5-atefehSZ' ]
then
    kernels=(aes gemm-blocked gemm-ncubed spmv-ellpack stencil nw stencil-3d)
    kernels=(fdtd-2d-large gemver)
    benchmark=poly
elif [ $HOSTNAME == 'jc8' ]
then
    # kernels=(aes gemm-blocked gemm-ncubed spmv-ellpack stencil nw stencil-3d)
    kernels=(syr2k)
    benchmark=poly
fi

for kernel in  ${kernels[@]}  
do
    if [ $HOSTNAME == 'jc8' ]
    then
        python3 parallel_run_tool_dse-third.py --version 'v18' --kernel $kernel --benchmark $benchmark --root-dir ../ --redis-port "7777" --server $HOSTNAME
    elif [ $HOSTNAME == 'u5-atefehSZ' ]
    then
        python3 parallel_run_tool_dse.py --version 'v20' --kernel $kernel --benchmark $benchmark --root-dir ../ --redis-port "7777"
    else
        python3 parallel_run_tool_dse.py --version 'v20' --kernel $kernel --benchmark $benchmark --root-dir ../ --redis-port "7777" 
    fi
done
