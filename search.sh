# exec_file
exec_file=$1

for config_file in {"margin001","margin002","margin003","margin004","margin005"}; do
    echo "${exec_file}"
    bash "${exec_file}" ${config_file}
done
