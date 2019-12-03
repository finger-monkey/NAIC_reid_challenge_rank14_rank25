# exec_file
exec_file=$1

for config_file in {"lr001","lr002","lr003","lr004","lr005","lr006","lr007","lr008","lr009"}; do
    echo "${exec_file}"
    bash "${exec_file}" ${config_file}
done
