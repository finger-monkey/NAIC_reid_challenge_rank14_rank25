# exec_file
exec_file=$1

for config_file in {"400_lr_001","400_lr_002","400_lr_003","400_lr_004"}; do
  echo "${exec_file}"
  bash "${exec_file}" ${config_file}
done