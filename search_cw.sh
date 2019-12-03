# exec_file
exec_file=$1

for config_file in {"cw001","cw002","cw003","cw004","cw005","cw006","cw007","cw008","cw009","cw010"}; do
    echo "${exec_file}"
    bash "${exec_file}" ${config_file}
done
