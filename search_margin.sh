# exec_file
exec_file=$1

for config_file in {"400_margin_001","400_margin_002","400_margin_003","400_margin_004"}; do
  echo "${exec_file}"
  bash "${exec_file}" ${config_file}
done