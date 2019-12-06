# exec_file
exec_file=$1

for config_file in {"cl001","cl002","cl003","cl004","cl005","cl006","cl007","cl008","cl009","cl010","cl011","cl012","cl013","cl014","cl015"}; do
  echo "${exec_file}"
  bash "${exec_file}" ${config_file}
done
