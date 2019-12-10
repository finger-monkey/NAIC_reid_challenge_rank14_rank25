# exec_file
exec_file=$1

for config_file in {"400_cw_001","400_cw_002","400_cw_003","400_cw_004","400_cw_005"}; do
  echo "${exec_file}"
  bash "${exec_file}" ${config_file}
done