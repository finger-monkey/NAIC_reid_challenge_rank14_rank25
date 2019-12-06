# exec_file
exec_file=$1

for config_file in {"pa001","pa002","pa003","pa004","pa005","pa006","pa007","pa008","pa009","pa010"}; do
  echo "${exec_file}"
  bash "${exec_file}" ${config_file}
done
