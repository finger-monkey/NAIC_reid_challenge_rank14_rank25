# exec_file
exec_file=$1

for config_file in {"mcw001","mcw002","mcw003","mcw004","mcw005","mcw006","mcw007","mcw008","mcw009","mcw010","mcw011"}; do
  echo "${exec_file}"
  bash "${exec_file}" ${config_file}
done
